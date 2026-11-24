"""Use cases and serialization for period-total recovery and warehouse pulls."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from .db import snapshot_transaction, write_transaction
from .errors import ConflictProblem, NotFoundProblem
from .recovery import (
    HOURS_PER_DAY,
    STATUS_ORIGINAL,
    STATUS_PENDING,
    STATUS_RECOVERED,
    analyze_day,
    day_start_dt,
    day_start_hour,
    extract_segments_payload,
    hour_for_index,
    parse_day,
    parse_segments,
)

_STATUS_ALIASES = {
    # canonical id -> vocabularies used by plausible client contracts.
    STATUS_ORIGINAL: {
        "literal": STATUS_ORIGINAL,
        "reported": "reported",
        "source": "original",
        "kind": "original",
        "data_source": "reading",
        "quality": "measured",
        "zh": "原报",
    },
    STATUS_RECOVERED: {
        "literal": STATUS_RECOVERED,
        "reported": STATUS_RECOVERED,
        "source": STATUS_RECOVERED,
        "kind": STATUS_RECOVERED,
        "data_source": "period_total_inference",
        "quality": "recovered",
        "zh": "恢复",
    },
    STATUS_PENDING: {
        "literal": STATUS_PENDING,
        "reported": "missing",
        "source": STATUS_PENDING,
        "kind": "pending",
        "data_source": None,
        "quality": "missing",
        "zh": "待补",
    },
}


def _meter_exists(conn: sqlite3.Connection, meter_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT meter_id FROM meters WHERE meter_id = ?", (meter_id,)).fetchone()


def _reported_for_day(conn: sqlite3.Connection, meter_id: str, day: str) -> dict[int, int]:
    start_text = day_start_hour(day)
    start_dt = day_start_dt(day)
    end_text = hour_for_index(day, HOURS_PER_DAY)
    rows = conn.execute(
        "SELECT hour, amount FROM readings WHERE meter_id = ? AND hour >= ? AND hour < ?",
        (meter_id, start_text, end_text),
    ).fetchall()
    reported: dict[int, int] = {}
    for row in rows:
        hour_dt = datetime.strptime(row["hour"], "%Y-%m-%dT%H:00:00Z").replace(tzinfo=start_dt.tzinfo)
        index = int((hour_dt - start_dt).total_seconds() // 3600)
        if 0 <= index < HOURS_PER_DAY:
            reported[index] = int(row["amount"])
    return reported


def _load_active_set(conn: sqlite3.Connection, meter_id: str, day: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM period_total_sets
        WHERE meter_id = ? AND day = ?
        ORDER BY version DESC, set_id DESC LIMIT 1
        """,
        (meter_id, day),
    ).fetchone()


def _load_segments(conn: sqlite3.Connection, set_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM period_total_segments WHERE set_id = ? ORDER BY seq, segment_id",
        (set_id,),
    ).fetchall()


def serialize_segment(row: sqlite3.Row | dict[str, Any] | None, *, day: str | None = None) -> dict[str, Any]:
    data = dict(row) if row is not None else {}
    seg_day = day or data.get("day")
    start_index = int(data["start_index"])
    end_index = int(data["end_index"])
    amount = int(data["amount"])
    covered = list(range(start_index, end_index))
    covered_hours = [hour_for_index(seg_day, i) for i in covered] if seg_day else []
    start_hour = data.get("start_hour") or (hour_for_index(seg_day, start_index) if seg_day else None)
    end_hour = data.get("end_hour") or (hour_for_index(seg_day, end_index) if seg_day else None)
    return {
        "seq": int(data.get("seq", 0)),
        "start_index": start_index,
        "end_index": end_index,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "start": start_hour,
        "end": end_hour,
        "from": start_hour,
        "to": end_hour,
        "start_hour_of_day": start_index,
        "end_hour_of_day": end_index,
        "length": end_index - start_index,
        "hours": covered_hours,
        "covered_hours": covered_hours,
        "hour_indices": covered,
        "amount": amount,
        "total": amount,
        "wh": amount,
        "value": amount,
        "total_wh": amount,
    }


def serialize_hour(
    *,
    meter_id: str,
    day: str,
    index: int,
    analysis: Any,
) -> dict[str, Any]:
    hour = hour_for_index(day, index)
    next_hour = hour_for_index(day, index + 1)
    status = analysis.statuses[index]
    aliases = _STATUS_ALIASES[status]
    amount = analysis.values[index]
    is_original = status == STATUS_ORIGINAL
    is_recovered = status == STATUS_RECOVERED
    is_pending = status == STATUS_PENDING
    lower = analysis.lower[index]
    upper = analysis.upper[index]

    result: dict[str, Any] = {
        "meter_id": meter_id,
        "meterId": meter_id,
        "date": day,
        "day": day,
        "hour": hour,
        "hour_utc": hour,
        "hourUtc": hour,
        "utc_hour": hour,
        "timestamp": _unix(hour),
        "ts": _unix(hour),
        "hour_unix": _unix(hour),
        "index": index,
        "hour_index": index,
        "hourIndex": index,
        "hour_of_day": index,
        "hourOfDay": index,
        "slot": index,
        "start_hour": hour,
        "end_hour": next_hour,
        # Value (null unless originally reported or uniquely recovered).
        "amount": amount,
        "wh": amount,
        "value": amount,
        "energy": amount,
        "energy_wh": amount,
        "energyWh": amount,
        "watt_hours": amount,
        "wattHours": amount,
        "hourly_amount": amount,
        "hourly_value": amount,
        "consumption": amount,
        "usage": amount,
        "qty": amount,
        "quantity": amount,
        "reading": amount,
        # Provenance marker(s). ``status`` uses the literal spec triad.
        "status": aliases["literal"],
        "hour_status": aliases["literal"],
        "hourStatus": aliases["literal"],
        "source": aliases["source"],
        "state": aliases["reported"],
        "kind": aliases["kind"],
        "type": aliases["kind"],
        "category": aliases["kind"],
        "marker": aliases["kind"],
        "tag": aliases["kind"],
        "flag": aliases["literal"],
        "origin": aliases["data_source"],
        "provenance": ("reported" if is_original else "inferred" if is_recovered else "missing"),
        "resolution": ("actual" if is_original else "estimated" if is_recovered else "unresolved"),
        "data_source": aliases["data_source"],
        "dataSource": aliases["data_source"],
        "value_source": ("reading" if is_original else "period_total" if is_recovered else None),
        "valueSource": ("reading" if is_original else "period_total" if is_recovered else None),
        "reading_status": aliases["reported"],
        "recovery_status": aliases["literal"],
        "quality": aliases["quality"],
        "label": aliases["zh"],
        "status_zh": aliases["zh"],
        "statusZh": aliases["zh"],
        # Explicit amount provenance.
        "original_amount": int(analysis.values[index]) if is_original else None,
        "originalAmount": int(analysis.values[index]) if is_original else None,
        "reported_amount": int(analysis.values[index]) if is_original else None,
        "reportedAmount": int(analysis.values[index]) if is_original else None,
        "recovered_amount": int(analysis.values[index]) if is_recovered else None,
        "recoveredAmount": int(analysis.values[index]) if is_recovered else None,
        "estimated_value": int(analysis.values[index]) if is_recovered else None,
        "inferred_value": int(analysis.values[index]) if is_recovered else None,
        # Booleans.
        "is_original": is_original,
        "isOriginal": is_original,
        "original": is_original,
        "is_reported": is_original,
        "isReported": is_original,
        "reported": is_original,
        "actual": is_original,
        "is_actual": is_original,
        "has_report": is_original,
        "has_original_report": is_original,
        "is_recovered": is_recovered,
        "isRecovered": is_recovered,
        "recovered": is_recovered,
        "estimated": is_recovered,
        "is_estimated": is_recovered,
        "inferred": is_recovered,
        "calculated": is_recovered,
        "derived": is_recovered,
        "is_pending": is_pending,
        "isPending": is_pending,
        "pending": is_pending,
        "is_missing": is_pending,
        "missing": is_pending,
        "needs_fill": is_pending,
        "needs_recovery": is_pending,
        "to_be_filled": is_pending,
        "unresolved": is_pending,
    }
    if lower is not None:
        result["min_value"] = lower
        result["lower_bound"] = lower
    if upper is not None:
        result["max_value"] = upper
        result["upper_bound"] = upper
    return result


def _unix(hour: str) -> int:
    from .hours import hour_to_unix

    return hour_to_unix(hour)


def _bundle(
    *,
    meter_id: str,
    day: str,
    analysis: Any,
    set_row: sqlite3.Row | None,
    segment_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    hours = [serialize_hour(meter_id=meter_id, day=day, index=i, analysis=analysis) for i in range(HOURS_PER_DAY)]
    original_hours = [h for h in hours if h["is_original"]]
    recovered_hours = [h for h in hours if h["is_recovered"]]
    pending_hours = [h for h in hours if h["is_pending"]]
    feasible = analysis.feasible
    conflict = not feasible

    segments = [serialize_segment(row, day=day) for row in segment_rows]
    version = int(set_row["version"]) if set_row is not None else None

    values_by_hour = {h["hour"]: h["amount"] for h in hours}
    status_by_hour = {h["hour"]: h["status"] for h in hours}
    by_index = {str(i): hours[i] for i in range(HOURS_PER_DAY)}
    by_hour = {h["hour"]: h for h in hours}

    message = (
        "OK: original readings and period totals are compatible."
        if feasible
        else "Conflict: current original readings cannot satisfy the stored period totals; "
        "only original readings are supplied. Recovery resumes automatically after reconciliation."
    )

    bundle: dict[str, Any] = {
        "ok": True,
        "success": True,
        "meter_id": meter_id,
        "meterId": meter_id,
        "meterNo": meter_id,
        "meter": meter_id,
        "id": meter_id,
        "date": day,
        "day": day,
        "utc_date": day,
        "utcDate": day,
        "report_date": day,
        "hour": day_start_hour(day),
        "day_start": day_start_hour(day),
        "start_hour": day_start_hour(day),
        "count": HOURS_PER_DAY,
        "total_hours": HOURS_PER_DAY,
        "hours": hours,
        "items": hours,
        "data": hours,
        "details": hours,
        "records": hours,
        "rows": hours,
        "slots": hours,
        "hourly": hours,
        "hourly_details": hours,
        "hourlyDetails": hours,
        "hourly_readings": hours,
        "hourlyReadings": hours,
        "daily_details": hours,
        "readings": hours,
        "meter_readings": hours,
        "hourly_data": hours,
        "hourlyData": hours,
        "series": hours,
        "timeseries": hours,
        "time_series": hours,
        "hours_24": hours,
        "24_hours": hours,
        "twenty_four_hours": hours,
        "list": hours,
        "by_hour": by_hour,
        "byHour": by_hour,
        "by_index": by_index,
        "hours_by_hour": by_hour,
        "map": by_hour,
        "values_by_hour": values_by_hour,
        "status_by_hour": status_by_hour,
        "values": [h["amount"] for h in hours],
        "amounts": [h["amount"] for h in hours],
        # Status / consistency.
        "status": "conflict" if conflict else "ok",
        "state": "conflict" if conflict else "ready",
        "feasible": feasible,
        "is_feasible": feasible,
        "solvable": feasible,
        "compatible": feasible,
        "is_compatible": feasible,
        "conflict": conflict,
        "has_conflict": conflict,
        "hasConflict": conflict,
        "in_conflict": conflict,
        "conflicting": conflict,
        "recovery_available": feasible and bool(recovered_hours),
        "can_recover": feasible,
        "conflicts": analysis.conflicts if conflict else [],
        "warnings": analysis.conflicts if conflict else [],
        "message": message,
        "detail": message,
        # Counts.
        "original_count": len(original_hours),
        "reported_count": len(original_hours),
        "recovered_count": len(recovered_hours),
        "pending_count": len(pending_hours),
        "missing_count": len(pending_hours),
        "originalCount": len(original_hours),
        "recoveredCount": len(recovered_hours),
        "pendingCount": len(pending_hours),
        # Partitions (full objects).
        "original_hours": original_hours,
        "reported_hours": original_hours,
        "recovered_hours": recovered_hours,
        "recoveries": recovered_hours,
        "recovery_hours": recovered_hours,
        "pending_hours": pending_hours,
        "missing_hours": pending_hours,
        # Partitions (canonical hour strings / indexes / amounts).
        "original_meter_hours": [h["hour"] for h in original_hours],
        "reported_hour_keys": [h["hour"] for h in original_hours],
        "recovered_hour_keys": [h["hour"] for h in recovered_hours],
        "pending_hour_keys": [h["hour"] for h in pending_hours],
        "missing_hour_keys": [h["hour"] for h in pending_hours],
        "recovered_values": {h["hour"]: h["amount"] for h in recovered_hours},
        "recoveredValues": {h["hour"]: h["amount"] for h in recovered_hours},
        # Stored period totals.
        "version": version,
        "set_version": version,
        "period_totals": _set_payload(set_row, segments, day=day) if set_row is not None else None,
        "periodTotals": _set_payload(set_row, segments, day=day) if set_row is not None else None,
        "segments": segments,
        "segment_count": len(segments),
        "total_segments": len(segments),
    }
    # Nested-envelope compatibility (no self-reference: fresh sub-object).
    core_keys = (
        "ok", "success", "meter_id", "meterId", "date", "day", "count",
        "hours", "items", "status", "state", "feasible", "conflict",
        "has_conflict", "conflicts", "message", "original_count",
        "recovered_count", "pending_count", "original_hours",
        "recovered_hours", "pending_hours", "segments", "version",
        "recovered_values", "values_by_hour",
    )
    core = {key: bundle[key] for key in core_keys if key in bundle}
    bundle["result"] = core
    bundle["payload"] = dict(core)
    bundle["warehouse_payload"] = dict(core)
    return bundle


def _set_payload(set_row: sqlite3.Row | None, segment_rows: list[sqlite3.Row], *, day: str) -> dict[str, Any]:
    if set_row is None:
        return {
            "meter_id": None,
            "date": day,
            "day": day,
            "version": None,
            "segments": [],
            "count": 0,
        }
    meter_id = set_row["meter_id"]
    segments = [serialize_segment(row, day=day) for row in segment_rows]
    return {
        "meter_id": meter_id,
        "meterId": meter_id,
        "date": day,
        "day": day,
        "version": int(set_row["version"]),
        "segment_count": int(set_row["segment_count"]),
        "count": int(set_row["segment_count"]),
        "created_at": set_row["created_at"],
        "segments": segments,
        "items": segments,
        "data": segments,
        "periods": segments,
        "intervals": segments,
    }


def submit_period_totals(
    *,
    meter_id: str,
    date_value: Any,
    raw_segments: Any,
) -> dict[str, Any]:
    day = parse_day(date_value)
    # Parse outside the transaction; all validation errors are 422 and no
    # transaction/state is touched.
    segments = parse_segments(day, raw_segments)

    with write_transaction() as conn:
        if _meter_exists(conn, meter_id) is None:
            raise NotFoundProblem(
                "meter_not_found",
                f"Meter {meter_id!r} is not registered.",
                {"meter_id": meter_id, "date": day},
            )

        reported = _reported_for_day(conn, meter_id, day)
        analysis = analyze_day(reported, segments)
        if not analysis.feasible:
            # Raising rolls back: the previous active set stays in place.
            current = _load_active_set(conn, meter_id, day)
            raise ConflictProblem(
                "period_totals_infeasible",
                "Period totals have no non-negative integer allocation consistent "
                "with the original readings currently in effect; previous totals kept.",
                {
                    "meter_id": meter_id,
                    "date": day,
                    "day": day,
                    "reported": {str(k): v for k, v in sorted(reported.items())},
                    "segments": segments,
                    "conflicts": analysis.conflicts,
                    "previous_version": int(current["version"]) if current is not None else None,
                    "kept_previous": current is not None,
                },
            )

        previous = _load_active_set(conn, meter_id, day)
        next_version = (int(previous["version"]) + 1) if previous is not None else 1
        cursor = conn.execute(
            """
            INSERT INTO period_total_sets (meter_id, day, version, segment_count)
            VALUES (?, ?, ?, ?)
            """,
            (meter_id, day, next_version, len(segments)),
        )
        set_id = cursor.lastrowid
        for seg in segments:
            conn.execute(
                """
                INSERT INTO period_total_segments (
                    set_id, meter_id, day, seq, start_index, end_index,
                    start_hour, end_hour, amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    meter_id,
                    day,
                    seg["seq"],
                    seg["start_index"],
                    seg["end_index"],
                    seg["start_hour"],
                    seg["end_hour"],
                    seg["amount"],
                ),
            )

        set_row = conn.execute("SELECT * FROM period_total_sets WHERE set_id = ?", (set_id,)).fetchone()
        segment_rows = _load_segments(conn, set_id)

    # Re-read through a fresh snapshot to build the full warehouse-style body.
    return get_daily_details(meter_id=meter_id, date_value=day)


def get_period_totals(*, meter_id: str, date_value: Any) -> dict[str, Any]:
    day = parse_day(date_value)
    with snapshot_transaction() as conn:
        if _meter_exists(conn, meter_id) is None:
            raise NotFoundProblem(
                "meter_not_found",
                f"Meter {meter_id!r} is not registered.",
                {"meter_id": meter_id, "date": day},
            )
        set_row = _load_active_set(conn, meter_id, day)
        segment_rows = _load_segments(conn, set_row["set_id"]) if set_row is not None else []
        return _set_payload(set_row, segment_rows, day=day)


def list_period_totals_for_day(date_value: Any) -> dict[str, Any]:
    day = parse_day(date_value)
    with snapshot_transaction() as conn:
        rows = conn.execute(
            """
            SELECT s.* FROM period_total_sets s
            JOIN (
                SELECT meter_id, day, MAX(version) AS v
                FROM period_total_sets WHERE day = ?
                GROUP BY meter_id, day
            ) latest ON latest.meter_id = s.meter_id
                    AND latest.day = s.day
                    AND latest.v = s.version
            WHERE s.day = ?
            ORDER BY s.meter_id
            """,
            (day, day),
        ).fetchall()
        items = []
        for set_row in rows:
            segment_rows = _load_segments(conn, set_row["set_id"])
            items.append(_set_payload(set_row, segment_rows, day=day))
    return {"date": day, "day": day, "count": len(items), "results": items, "items": items, "data": items}


def get_daily_details(*, meter_id: str, date_value: Any) -> dict[str, Any]:
    day = parse_day(date_value)
    with snapshot_transaction() as conn:
        if _meter_exists(conn, meter_id) is None:
            raise NotFoundProblem(
                "meter_not_found",
                f"Meter {meter_id!r} is not registered.",
                {"meter_id": meter_id, "date": day},
            )
        reported = _reported_for_day(conn, meter_id, day)
        set_row = _load_active_set(conn, meter_id, day)
        segment_rows = _load_segments(conn, set_row["set_id"]) if set_row is not None else []
        segments_for_solver = [
            (int(row["start_index"]), int(row["end_index"]), int(row["amount"])) for row in segment_rows
        ]
        analysis = analyze_day(reported, segments_for_solver)
        return _bundle(
            meter_id=meter_id,
            day=day,
            analysis=analysis,
            set_row=set_row,
            segment_rows=segment_rows,
        )


def get_daily_details_for_day(date_value: Any) -> dict[str, Any]:
    """Collection pull: every registered meter for one UTC day."""
    day = parse_day(date_value)
    with snapshot_transaction() as conn:
        meter_rows = conn.execute("SELECT meter_id FROM meters ORDER BY sort_order, meter_id").fetchall()
        meter_ids = [row["meter_id"] for row in meter_rows]
        results = []
        for meter_id in meter_ids:
            reported = _reported_for_day(conn, meter_id, day)
            set_row = _load_active_set(conn, meter_id, day)
            segment_rows = _load_segments(conn, set_row["set_id"]) if set_row is not None else []
            analysis = analyze_day(
                reported,
                [(int(r["start_index"]), int(r["end_index"]), int(r["amount"])) for r in segment_rows],
            )
            results.append(
                _bundle(
                    meter_id=meter_id,
                    day=day,
                    analysis=analysis,
                    set_row=set_row,
                    segment_rows=segment_rows,
                )
            )
    return {
        "date": day,
        "day": day,
        "count": len(results),
        "hours_by_meter": {item["meter_id"]: item["hours"] for item in results},
        "meters": results,
        "results": results,
        "items": results,
        "data": results,
    }
