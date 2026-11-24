"""Domain logic for recovering missing hourly Wh values from period totals.

The clerk submits, per meter + UTC day, a full replacement set of period
segments.  Every segment says: sum of hourly Wh values over the whole hours
``[start, end)`` equals ``total``.  Segments may overlap.  Original readings
pin individual hourly variables.

Recovering a value is a uniqueness question over all non-negative integer
assignments satisfying every equality.  We solve it exactly by turning the
system into difference constraints on prefix sums and running all-pairs
shortest paths (Floyd-Warshall on the <=25 prefix nodes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .errors import ValidationProblem
from .hours import parse_hour

HOURS_PER_DAY = 24

# Canonical provenance markers (literal mapping of 原报 / 恢复 / 待补).
STATUS_ORIGINAL = "original"
STATUS_RECOVERED = "recovered"
STATUS_PENDING = "pending"

_DATE_RE = re.compile(r"^(\d{4})[-/.]?(\d{2})[-/.]?(\d{2})$")
_CLOCK_RE = re.compile(r"^(\d{1,2})(?::?(\d{2}))?(?::\d{2}(?:\.\d+)?)?$")
_RANGE_KEY_RE = re.compile(
    r"^\s*(\d{1,2}(?::\d{2})?)\s*(?:-|–|—|~|to|\.\.)\s*(\d{1,2}(?::\d{2})?)\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Day / hour indexing
# --------------------------------------------------------------------------- #


def parse_day(value: Any) -> str:
    """Return canonical ``YYYY-MM-DD`` (UTC) for a date/datetime value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationProblem("date_required", "date (UTC day) is required.")
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Unix timestamp (seconds / milliseconds) -> its UTC day.
        number = int(value)
        if abs(number) >= 10_000_000_000:
            number //= 1000
        if abs(number) >= 100_000:
            return datetime.fromtimestamp(number, tz=timezone.utc).strftime("%Y-%m-%d")
        # YYYYMMDD as an 8-digit integer.
        text = str(abs(number)).zfill(8)
        match = _DATE_RE.fullmatch(text)
        if match:
            return _valid_day(match)
    text = str(value).strip()
    match = _DATE_RE.fullmatch(text)
    if match:
        return _valid_day(match)
    # Fall back to the general hour parser (full ISO timestamps etc.).
    hour = parse_hour(value)
    return hour[:10]


def _valid_day(match: re.Match[str]) -> str:
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, tzinfo=timezone.utc).strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValidationProblem("invalid_date", f"Invalid UTC date {match.string!r}.") from exc


def day_start_dt(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def day_start_hour(day: str) -> str:
    return f"{day}T00:00:00Z"


def hour_for_index(day: str, index: int) -> str:
    return (day_start_dt(day) + timedelta(hours=index)).strftime("%Y-%m-%dT%H:00:00Z")


def index_for_hour(day: str, hour_text: str) -> int:
    parsed = parse_hour(hour_text)
    dt = datetime.strptime(parsed, "%Y-%m-%dT%H:00:00Z").replace(tzinfo=timezone.utc)
    delta = dt - day_start_dt(day)
    index = int(delta.total_seconds() // 3600)
    return index


def endpoint_to_index(day: str, value: Any, *, edge: str) -> int:
    """Resolve a segment start/end boundary to a 0..24 prefix index."""
    if isinstance(value, bool):
        raise ValidationProblem(
            "invalid_segment_boundary",
            f"Segment {edge} must be a whole UTC hour or 0..24 hour index.",
        )
    # Small integers are hour-of-day indexes; large numbers are timestamps.
    if isinstance(value, int):
        if 0 <= value <= HOURS_PER_DAY:
            return value
        return index_for_hour(day, parse_hour(value))
    if isinstance(value, float) and value.is_integer():
        return endpoint_to_index(day, int(value), edge=edge)
    if isinstance(value, datetime):
        parsed = parse_hour(value)
        return index_for_hour(day, parsed)

    text = str(value).strip()
    if not text:
        raise ValidationProblem("invalid_segment_boundary", f"Segment {edge} is required.")

    # Pure-digit clock forms: "9", "09", "0900" (never an 8/10-digit date key).
    if text.isdigit() and len(text) <= 4:
        if len(text) <= 2:
            hour, minute = int(text), 0
        elif len(text) == 3:
            hour, minute = int(text[0]), int(text[1:])
        else:
            hour, minute = int(text[:2]), int(text[2:])
        if minute != 0:
            raise ValidationProblem(
                "hour_not_whole",
                f"Segment {edge} {text!r} must be a whole hour (minutes must be zero).",
            )
        if not 0 <= hour <= HOURS_PER_DAY:
            raise ValidationProblem(
                "segment_out_of_day",
                f"Segment {edge} hour {hour} is outside the UTC day (0..24).",
            )
        return hour

    # "HH" / "HH:MM" / "HH:MM:SS" clock text.
    clock = _CLOCK_RE.fullmatch(text)
    if clock and ":" in text:
        parts = text.split(":")
        try:
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        except ValueError as exc:
            raise ValidationProblem("invalid_segment_boundary", f"Invalid {edge} hour {text!r}.") from exc
        if minute != 0:
            raise ValidationProblem(
                "hour_not_whole",
                f"Segment {edge} {text!r} must be a whole hour (minutes must be zero).",
            )
        if not 0 <= hour <= HOURS_PER_DAY:
            raise ValidationProblem(
                "segment_out_of_day",
                f"Segment {edge} hour {hour} is outside the UTC day (0..24).",
            )
        return hour

    # Anything else is an absolute ISO/compact timestamp.
    return index_for_hour(day, parse_hour(value))


# --------------------------------------------------------------------------- #
# Segment parsing
# --------------------------------------------------------------------------- #

START_KEYS = (
    "start", "start_hour", "startHour", "start_time", "startTime", "start_at",
    "startAt", "from", "from_hour", "fromHour", "begin", "begin_hour",
    "beginHour", "beginning", "start_index", "startIndex", "s",
)
END_KEYS = (
    "end", "end_hour", "endHour", "end_time", "endTime", "end_at", "endAt",
    "to", "to_hour", "toHour", "finish", "finish_hour", "stop", "stop_hour",
    "stopHour", "end_index", "endIndex", "e",
)
TOTAL_KEYS = (
    "total", "totals", "total_wh", "totalWh", "total_amount", "totalAmount",
    "total_value", "totalValue", "period_total", "periodTotal",
    "period_total_wh", "period_amount", "periodAmount", "sum", "sum_wh",
    "amount", "amount_wh", "wh", "value", "watt_hours", "wattHours",
    "energy_wh", "energyWh", "energy", "usage", "consumption", "qty",
    "quantity", "reading", "reading_value", "count",
)
SEGMENT_CONTAINER_KEYS = (
    "segments", "periods", "intervals", "ranges", "blocks", "slots", "parts",
    "entries", "rows", "items", "list", "period_totals", "interval_totals",
    "segment_totals", "totals", "duty_totals", "duty_records", "records",
    "handovers", "handover", "shift_totals", "shift_readings", "data",
)
DATE_KEYS = (
    "date", "day", "utc_date", "utcDate", "date_utc", "day_utc",
    "report_date", "reportDate", "reading_date", "target_date", "dt",
)


def _first(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None and data[key] != "":
            return data[key]
    return None


def parse_segment(day: str, raw: Any, *, seq: int = 0) -> dict[str, Any]:
    """Validate and normalize one segment into ``{start_index,end_index,amount}``."""
    if isinstance(raw, (list, tuple)):
        if len(raw) == 2 and isinstance(raw[1], (int, float)) and not isinstance(raw[1], bool):
            # Could be [start, end] with amount elsewhere? Treat 3-array only.
            start_raw, end_raw = raw[0], raw[1]
            amount_raw = None
        elif len(raw) >= 3:
            start_raw, end_raw, amount_raw = raw[0], raw[1], raw[2]
        else:
            raise ValidationProblem(
                "invalid_segment",
                "Array segments must be [start_hour, end_hour, total_wh].",
                {"segment": raw, "seq": seq},
            )
    elif isinstance(raw, dict):
        start_raw = _first(raw, START_KEYS)
        end_raw = _first(raw, END_KEYS)
        amount_raw = _first(raw, TOTAL_KEYS)
        # {"hours": [s, e], "total": T} / {"range": "0-3", ...}
        hours = raw.get("hours") or raw.get("hour_range") or raw.get("range")
        if (start_raw is None or end_raw is None) and isinstance(hours, (list, tuple)) and len(hours) >= 2:
            start_raw = start_raw if start_raw is not None else hours[0]
            end_raw = end_raw if end_raw is not None else hours[1]
        if (start_raw is None or end_raw is None) and isinstance(hours, str):
            match = _RANGE_KEY_RE.fullmatch(hours)
            if match:
                start_raw, end_raw = match.group(1), match.group(2)
    elif isinstance(raw, str):
        match = _RANGE_KEY_RE.fullmatch(raw)
        if not match:
            raise ValidationProblem("invalid_segment", f"Unrecognized segment {raw!r}.")
        start_raw, end_raw, amount_raw = match.group(1), match.group(2), None
    else:
        raise ValidationProblem("invalid_segment", "Each period segment must be an object.", {"seq": seq})

    # String range key form {"0-3": 600} is handled by parse_segments mapping.
    if start_raw is None or end_raw is None:
        raise ValidationProblem(
            "segment_bounds_required",
            "Each segment requires start and end whole hours.",
            {"segment": raw, "seq": seq},
        )
    if amount_raw is None:
        raise ValidationProblem(
            "segment_total_required",
            "Each segment requires a non-negative integer total in Wh.",
            {"segment": raw, "seq": seq},
        )

    from .payload import AMOUNT_KEYS, amount_from, first_present as _fp

    # Tolerate nested value envelopes such as {"reading": {"amount": 100}}.
    if isinstance(amount_raw, dict):
        nested = _fp(amount_raw, AMOUNT_KEYS)
        amount_raw = nested if nested is not None else amount_raw.get("total")

    amount = amount_from(amount_raw)
    start_index = endpoint_to_index(day, start_raw, edge="start")
    end_index = endpoint_to_index(day, end_raw, edge="end")

    if end_index <= start_index:
        raise ValidationProblem(
            "invalid_segment_range",
            "Segment end must be after start; coverage is [start, end).",
            {"start_index": start_index, "end_index": end_index},
        )
    if not (0 <= start_index and end_index <= HOURS_PER_DAY):
        raise ValidationProblem(
            "segment_out_of_day",
            "Every covered hour must belong to the reported UTC day.",
            {
                "start_index": start_index,
                "end_index": end_index,
                "allowed": [0, HOURS_PER_DAY],
            },
        )

    return {
        "seq": seq,
        "start_index": start_index,
        "end_index": end_index,
        "amount": amount,
        "start_hour": hour_for_index(day, start_index),
        "end_hour": hour_for_index(day, end_index),
        "covered_indices": list(range(start_index, end_index)),
        "covered_hours": [hour_for_index(day, i) for i in range(start_index, end_index)],
    }


def parse_segments(day: str, raw_segments: Any) -> list[dict[str, Any]]:
    """Accept lists, range-key mappings, and single segment objects."""
    if raw_segments is None:
        return []

    # Mapping form: {"0-3": 600, "3-6": 400} or {"00:00-02:00": {"total": 10}}.
    if isinstance(raw_segments, dict):
        items: list[Any] = []
        for key, value in raw_segments.items():
            match = _RANGE_KEY_RE.fullmatch(str(key))
            if match:
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("start", match.group(1))
                    entry.setdefault("end", match.group(2))
                    items.append(entry)
                else:
                    items.append({"start": match.group(1), "end": match.group(2), "total": value})
            else:
                # Nested container such as {"segments": [...]} already handled
                # by caller; a plain dict segment is parsed as one segment.
                items.append(raw_segments)
                break
        raw_segments = items

    if not isinstance(raw_segments, (list, tuple)):
        # A single segment object.
        raw_segments = [raw_segments]

    parsed = [parse_segment(day, raw, seq=i) for i, raw in enumerate(raw_segments)]
    return parsed


def extract_segments_payload(body: dict[str, Any]) -> Any:
    """Find the segments container under any common envelope key."""
    found = _first(body, SEGMENT_CONTAINER_KEYS)
    if found is not None:
        # {"period_totals": {"segments": [...]}} nested envelope.
        if isinstance(found, dict):
            nested = _first(found, SEGMENT_CONTAINER_KEYS)
            if nested is not None:
                return nested
            # Maybe mapping of range -> total.
            return found
        return found
    # Single flat segment in the body itself.
    if _first(body, START_KEYS) is not None and _first(body, END_KEYS) is not None:
        return [body]
    return None


# --------------------------------------------------------------------------- #
# Difference-constraint solver
# --------------------------------------------------------------------------- #


@dataclass
class DayAnalysis:
    feasible: bool
    #: Unique value for each hour when known (reported or uniquely recovered).
    values: list[int | None]
    #: "original" | "recovered" | "pending" per hour.
    statuses: list[str]
    #: Tight integer lower/upper bounds on each hourly variable.
    lower: list[int | None]
    upper: list[int | None]
    conflicts: list[dict[str, Any]]


def analyze_day(
    reported: dict[int, int],
    segments: Iterable[tuple[int, int, int]] | Iterable[dict[str, Any]],
    *,
    n: int = HOURS_PER_DAY,
) -> DayAnalysis:
    """Compute feasibility and unique recoverable values for one meter-day."""
    norm_segments: list[tuple[int, int, int]] = []
    for seg in segments:
        if isinstance(seg, dict):
            norm_segments.append((int(seg["start_index"]), int(seg["end_index"]), int(seg["amount"])))
        else:
            norm_segments.append((int(seg[0]), int(seg[1]), int(seg[2])))

    inf = float("inf")
    size = n + 1
    dist: list[list[Any]] = [[inf] * size for _ in range(size)]
    for i in range(size):
        dist[i][i] = 0

    def add_edge(u: int, v: int, weight: int) -> None:
        # Keep finite weights as arbitrary-precision ints; ``inf`` is only the
        # unreachable sentinel.
        if weight < dist[u][v]:
            dist[u][v] = weight

    # x_h >= 0  <=>  p_h - p_{h+1} <= 0  -> edge (h+1) -> h weight 0.
    for h in range(n):
        add_edge(h + 1, h, 0)

    # Original reports pin x_h = amount.
    for h, amount in reported.items():
        if not 0 <= h < n:
            continue
        add_edge(h, h + 1, int(amount))
        add_edge(h + 1, h, -int(amount))

    # Period totals pin p_end - p_start = total.
    for start, end, total in norm_segments:
        add_edge(start, end, int(total))
        add_edge(end, start, -int(total))

    # Floyd-Warshall.
    for k in range(size):
        row_k = dist[k]
        for i in range(size):
            dik = dist[i][k]
            if dik == inf:
                continue
            row_i = dist[i]
            for j in range(size):
                dkj = row_k[j]
                if dkj == inf:
                    continue
                candidate = dik + dkj
                if candidate < row_i[j]:
                    row_i[j] = candidate

    negative_nodes = [i for i in range(size) if dist[i][i] < 0]
    feasible = not negative_nodes

    lower: list[int | None] = [None] * n
    upper: list[int | None] = [None] * n
    values: list[int | None] = [None] * n
    statuses: list[str] = [STATUS_PENDING] * n
    conflicts: list[dict[str, Any]] = []

    if not feasible:
        conflicts.append(
            {
                "code": "constraint_system_infeasible",
                "message": (
                    "Original readings and period totals have no common "
                    "non-negative integer allocation."
                ),
                "prefix_nodes": negative_nodes,
            }
        )
        # In conflict mode only original reports are supplied.
        for h in range(n):
            if h in reported:
                values[h] = int(reported[h])
                statuses[h] = STATUS_ORIGINAL
        return DayAnalysis(False, values, statuses, lower, upper, conflicts)

    for h in range(n):
        low = -dist[h + 1][h]
        high = dist[h][h + 1]
        low_int = int(low) if low != inf else None
        high_int = int(high) if high != inf else None
        lower[h] = low_int
        upper[h] = high_int

        if h in reported:
            values[h] = int(reported[h])
            statuses[h] = STATUS_ORIGINAL
            continue

        if high_int is not None and low_int is not None and low_int == high_int:
            values[h] = high_int
            statuses[h] = STATUS_RECOVERED
        else:
            values[h] = None
            statuses[h] = STATUS_PENDING

    return DayAnalysis(True, values, statuses, lower, upper, [])
