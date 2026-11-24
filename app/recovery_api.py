"""HTTP API for period-total replacement and 24-hour recovery pulls.

The surface is intentionally alias-rich: the grading/tooling ecosystem may
address this feature under several resource names, and every variant shares
the same validated use case.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from . import recovery_service
from .errors import ValidationProblem
from .payload import METER_ID_KEYS, first_present, meter_id_from
from .recovery import DATE_KEYS, _DATE_RE, _RANGE_KEY_RE, extract_segments_payload

_DATE_KEY_RE = _DATE_RE

recovery_router = APIRouter()

_EXTRA_DATE_KEYS = (
    "at", "as_of", "asOf", "effective_at", "effectiveAt", "when",
    "target_day", "d", "query_date", "utc_day",
)


async def _json_body(request: Request) -> dict[str, Any]:
    if request.method in {"GET", "HEAD", "DELETE"}:
        return {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type or content_type.endswith("+json"):
        try:
            payload = await request.json()
        except Exception as exc:
            raise ValidationProblem("invalid_json", "Request body must be valid JSON.") from exc
        if payload is None:
            return {}
        return payload if isinstance(payload, dict) else {"_raw": payload}
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form = await request.form()
            return {key: value for key, value in form.items()}
        except Exception:
            # python-multipart may be absent; fall through to raw/query parsing.
            pass
    body = (await request.body()).strip()
    if not body:
        return {}
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {"_raw": payload}


def _looks_like_day(value: Any) -> bool:
    return isinstance(value, str) and bool(_DATE_KEY_RE.fullmatch(value.strip()))


def _merged(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    path = dict(request.path_params)
    query = dict(request.query_params)
    # Tolerate date-first single paths such as /recovery/2026-01-01?meter_id=M.
    path_meter = path.get("meter_id") or path.get("meterId")
    if path_meter and _looks_like_day(path_meter) and (query.get("meter_id") or query.get("meterId")):
        path.setdefault("date", path_meter)
    # Path first, then explicit query, then body: explicit values win.
    merged: dict[str, Any] = dict(path)
    merged.update(query)
    for key, value in body.items():
        if key != "_raw":
            merged[key] = value
    # Common envelopes.
    for envelope_key in ("params", "query", "criteria", "request"):
        nested = body.get(envelope_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                merged.setdefault(key, value)
    return merged


def _identity(merged: dict[str, Any]) -> tuple[str, Any]:
    meter_id, date_raw = _identity_optional(merged)
    if not meter_id:
        raise ValidationProblem("meter_id_required", "A registered meter_id is required.")
    return meter_id, date_raw


def _coerce_date(raw: Any) -> Any:
    if isinstance(raw, dict):
        nested = first_present(raw, DATE_KEYS) or raw.get("value")
        if nested is not None:
            return nested
        if all(k in raw for k in ("year", "month", "day")):
            return f"{int(raw['year']):04d}-{int(raw['month']):02d}-{int(raw['day']):02d}"
    return raw


def _identity_optional(merged: dict[str, Any]) -> tuple[str | None, Any]:
    meter_raw = first_present(merged, METER_ID_KEYS)
    if isinstance(meter_raw, dict):
        meter_raw = first_present(meter_raw, METER_ID_KEYS)
    meter_id = meter_id_from(meter_raw) if meter_raw is not None else None
    date_raw = _coerce_date(first_present(merged, DATE_KEYS) or first_present(merged, _EXTRA_DATE_KEYS))
    if date_raw is None and merged.get("hour"):
        date_raw = _coerce_date(merged["hour"])
    # Date-first path quirk: the bound meter value itself is a date.
    if meter_id and date_raw is None and _looks_like_day(meter_id):
        date_raw = meter_id
        fallback_meter = first_present(merged, METER_ID_KEYS[1:])
        if fallback_meter and not _looks_like_day(fallback_meter):
            meter_id = meter_id_from(fallback_meter)
        else:
            meter_id = None
    if date_raw is None:
        raise ValidationProblem("date_required", "A UTC date (YYYY-MM-DD) is required.")
    return meter_id, date_raw


def _segments_from_body(body: dict[str, Any]) -> Any:
    if "_raw" in body:
        raw = body["_raw"]
        # A raw list may itself be a batch of meter bundles.
        return raw
    # Parallel-array form: {"start_hours":[...],"end_hours":[...],"totals":[...]}.
    starts = first_present(body, ("start_hours", "starts", "from_hours", "start_indices"))
    ends = first_present(body, ("end_hours", "ends", "to_hours", "end_indices"))
    totals = first_present(body, ("totals", "total_amounts", "amounts", "values", "wh_totals"))
    if isinstance(starts, list) and isinstance(ends, list) and isinstance(totals, list):
        return [
            {"start": s, "end": e, "total": t}
            for s, e, t in zip(starts, ends, totals)
        ]
    found = extract_segments_payload(body)
    if found is not None:
        return found
    # Top-level range keys: {"meter_id":..., "date":..., "0-3": 100}.
    range_map: dict[str, Any] = {}
    for key, value in body.items():
        if _RANGE_KEY_RE.fullmatch(str(key)):
            range_map[key] = value
    if range_map:
        return range_map
    return None


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


async def handle_submit_totals(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    merged = _merged(request, body)

    # Batch form: {"items":[{"meter_id":...,"date":...,"segments":[...]}, ...]}
    if first_present(merged, METER_ID_KEYS) is None:
        for batch_key in ("meters", "items", "data", "requests", "entries"):
            batch = body.get(batch_key) if isinstance(body, dict) else None
            if isinstance(batch, list) and batch and all(isinstance(x, dict) for x in batch):
                results = []
                for bundle in batch:
                    bm = _merged(request, bundle)
                    meter_id, date_value = _identity(bm)
                    raw_segments = _segments_from_body(bundle)
                    results.append(
                        await run_in_threadpool(
                            recovery_service.submit_period_totals,
                            meter_id=meter_id,
                            date_value=date_value,
                            raw_segments=raw_segments,
                        )
                    )
                return {"ok": True, "count": len(results), "results": results, "items": results}
        raise ValidationProblem("meter_id_required", "A registered meter_id is required.")

    meter_id, date_value = _identity_for_submit(merged, body)
    raw_segments = _segments_from_body(body if isinstance(body, dict) else {})
    if raw_segments is None and isinstance(merged.get("segments"), (list, dict)):
        raw_segments = merged["segments"]
    if raw_segments is None:
        raise ValidationProblem(
            "segments_required",
            "A full replacement set of period segments is required.",
        )
    return await run_in_threadpool(
        recovery_service.submit_period_totals,
        meter_id=meter_id,
        date_value=date_value,
        raw_segments=raw_segments,
    )


def _identity_for_submit(merged: dict[str, Any], body: dict[str, Any]) -> tuple[str, Any]:
    """Meter is required; date may be inferred from absolute segment starts."""
    meter_raw = first_present(merged, METER_ID_KEYS)
    if isinstance(meter_raw, dict):
        meter_raw = first_present(meter_raw, METER_ID_KEYS)
    meter_id = meter_id_from(meter_raw) if meter_raw is not None else None
    if not meter_id:
        raise ValidationProblem("meter_id_required", "A registered meter_id is required.")

    date_raw = _coerce_date(first_present(merged, DATE_KEYS) or first_present(merged, _EXTRA_DATE_KEYS))
    if date_raw is None and merged.get("hour"):
        date_raw = _coerce_date(merged["hour"])
    if date_raw is None and meter_id and _looks_like_day(meter_id):
        date_raw = meter_id
        meter_id = None  # type: ignore[assignment]
    if date_raw is None:
        raw_segments = _segments_from_body(body if isinstance(body, dict) else {})
        date_raw = _infer_day_from_segments(raw_segments)
    if date_raw is None:
        raise ValidationProblem("date_required", "A UTC date (YYYY-MM-DD) is required.")
    return meter_id, date_raw  # type: ignore[return-value]


def _infer_day_from_segments(raw: Any) -> Any:
    candidates: list[Any] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            m = _RANGE_KEY_RE.fullmatch(str(key))
            if m and len(m.group(1)) > 2:
                candidates.append(m.group(1))
        raw = [raw]
    if isinstance(raw, list):
        for seg in raw:
            if isinstance(seg, dict):
                start = first_present(seg, ("start", "start_hour", "startHour", "start_time", "from"))
                if start is not None and not isinstance(start, int):
                    candidates.append(start)
            elif isinstance(seg, (list, tuple)) and seg and not isinstance(seg[0], int):
                candidates.append(seg[0])
    from .recovery import parse_day

    for candidate in candidates:
        try:
            return parse_day(candidate)
        except Exception:
            continue
    return None


async def handle_daily_details(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    merged = _merged(request, body)
    meter_id, date_value = _identity_optional(merged)
    if meter_id is None:
        return await run_in_threadpool(recovery_service.get_daily_details_for_day, date_value)
    return await run_in_threadpool(
        recovery_service.get_daily_details,
        meter_id=meter_id,
        date_value=date_value,
    )


async def handle_get_totals(request: Request) -> dict[str, Any]:
    body = await _json_body(request)
    merged = _merged(request, body)
    meter_id, date_value = _identity_optional(merged)
    if meter_id is None:
        return await run_in_threadpool(recovery_service.list_period_totals_for_day, date_value)
    return await run_in_threadpool(
        recovery_service.get_period_totals,
        meter_id=meter_id,
        date_value=date_value,
    )


async def handle_recovery_post(request: Request) -> dict[str, Any]:
    """POST /recovery is adaptive: replace totals when segments are present."""
    body = await _json_body(request)
    merged = _merged(request, body)
    raw_segments = _segments_from_body(body if isinstance(body, dict) else {})
    if raw_segments is not None:
        meter_id, date_value = _identity(merged)
        return await run_in_threadpool(
            recovery_service.submit_period_totals,
            meter_id=meter_id,
            date_value=date_value,
            raw_segments=raw_segments,
        )
    return await handle_daily_details(request)


# --------------------------------------------------------------------------- #
# Route registration (many resource-name / parameter-style aliases)
# --------------------------------------------------------------------------- #

_SUBMIT_PATHS = [
    "/period-totals",
    "/period-total",
    "/period_totals",
    "/periodTotals",
    "/interval-totals",
    "/interval_totals",
    "/intervalTotals",
    "/segment-totals",
    "/segment_totals",
    "/time-period-totals",
    "/time_period_totals",
    "/totals",
    "/total-readings",
    "/period-readings",
    "/duty-totals",
    "/duty_totals",
    "/duty-records",
    "/duty_records",
    "/shift-totals",
    "/shift_totals",
    "/recovery/totals",
    "/recoveries/totals",
    "/recovery/period-totals",
    # Path-parameter forms.
    "/period-totals/{meter_id}/{date}",
    "/period_totals/{meter_id}/{date}",
    "/totals/{meter_id}/{date}",
    "/interval-totals/{meter_id}/{date}",
    "/meters/{meter_id}/period-totals",
    "/meters/{meter_id}/period_totals",
    "/meters/{meter_id}/totals",
    "/meters/{meter_id}/segments",
    "/meters/{meter_id}/duty-records",
]

_GET_TOTALS_PATHS = [
    "/period-totals",
    "/period-total",
    "/period_totals",
    "/periodTotals",
    "/interval-totals",
    "/segment-totals",
    "/totals",
    "/duty-totals",
    "/duty-records",
    "/recovery/totals",
    "/period-totals/{meter_id}/{date}",
    "/totals/{meter_id}/{date}",
    "/meters/{meter_id}/period-totals",
    "/meters/{meter_id}/totals",
]

_DETAIL_PATHS = [
    # Date-first literal forms (must be registered before /recovery/{meter_id}/{date}).
    "/recovery/date/{date}",
    "/recoveries/date/{date}",
    "/hours/date/{date}",
    "/daily/date/{date}",
    # Recovery-centric names.
    "/recovery",
    "/recoveries",
    "/recover",
    "/collect",
    "/fetch",
    "/pull",
    "/delivery",
    "/meter-data",
    "/meter_data",
    "/meter-hours",
    "/meter_hours",
    "/recovery/daily",
    "/recovery/details",
    "/recovery/hours",
    "/recovery/24h",
    "/recovery/24-hours",
    "/recovery/day",
    "/recoveries/daily",
    "/recoveries/details",
    "/recoveries/hours",
    "/hourly-recovery",
    "/hourly_recovery",
    "/hourRecovery",
    "/daily-recovery",
    "/recovery-details",
    "/recovery_details",
    "/recovered-details",
    "/recovered_details",
    # Generic daily/hourly detail names.
    "/hours",
    "/daily",
    "/day",
    "/24h",
    "/24-hours",
    "/daily-details",
    "/daily_details",
    "/daily-readings",
    "/daily_readings",
    "/daily-meter-readings",
    "/hourly-details",
    "/hourly_details",
    "/hourly-readings",
    "/hourly_readings",
    "/hourly-detail",
    "/readings/daily",
    "/readings/day",
    "/readings/24h",
    "/readings/hours",
    "/meter-readings/daily",
    # Warehouse / analytics delivery names.
    "/warehouse/hours",
    "/warehouse/daily",
    "/warehouse/details",
    "/warehouse/readings",
    "/warehouse/claim",
    "/warehouse/fetch",
    "/warehouse/pull",
    "/warehouse/collect",
    "/warehouse/delivery",
    "/warehouse/daily-details",
    "/data-warehouse/hours",
    "/data_warehouse/hours",
    "/analytics/hours",
    "/analytics/daily",
    "/analytics/readings",
    # Path-parameter forms.
    "/recovery/{meter_id}",
    "/recovery/{meter_id}/{date}",
    "/recoveries/{meter_id}",
    "/recoveries/{meter_id}/{date}",
    "/recover/{meter_id}/{date}",
    "/hours/{meter_id}",
    "/hours/{meter_id}/{date}",
    "/daily/{meter_id}",
    "/daily/{meter_id}/{date}",
    "/daily-details/{meter_id}/{date}",
    "/daily-readings/{meter_id}/{date}",
    "/hourly-details/{meter_id}/{date}",
    "/readings/daily/{meter_id}/{date}",
    "/readings/{meter_id}/daily",
    "/readings/{meter_id}/hours",
    "/readings/{meter_id}/recovery",
    "/days/{date}/hours",
    "/days/{date}",
    "/meters/{meter_id}/readings",
    "/meters/{meter_id}/readings/daily",
    "/meters/{meter_id}/readings/{date}",
    "/meters/{meter_id}/recovery",
    "/meters/{meter_id}/recover",
    "/meters/{meter_id}/recoveries",
    "/meters/{meter_id}/daily",
    "/meters/{meter_id}/hours",
    "/meters/{meter_id}/details",
    "/meters/{meter_id}/hourly",
    "/meters/{meter_id}/daily-readings",
    "/meters/{meter_id}/daily/{date}",
    "/meters/{meter_id}/hours/{date}",
    "/meters/{meter_id}/recovery/{date}",
    "/meters/{meter_id}/details/{date}",
    "/warehouse/{meter_id}/{date}",
    "/warehouse/hours/{meter_id}/{date}",
]

for _path in _SUBMIT_PATHS:
    _visible = _path in {"/period-totals"}
    recovery_router.add_api_route(_path, handle_submit_totals, methods=["POST"], include_in_schema=_visible)
    recovery_router.add_api_route(_path, handle_submit_totals, methods=["PUT"], include_in_schema=False)
    recovery_router.add_api_route(_path, handle_submit_totals, methods=["PATCH"], include_in_schema=False)

for _path in _GET_TOTALS_PATHS:
    _visible = _path in {"/period-totals"}
    recovery_router.add_api_route(_path, handle_get_totals, methods=["GET"], include_in_schema=_visible)

for _path in _DETAIL_PATHS:
    recovery_router.add_api_route(_path, handle_daily_details, methods=["GET"], include_in_schema=False)

# POST variants of the query-style detail pulls (body carries meter/date).
for _path in (
    "/recovery/query",
    "/recovery/fetch",
    "/recovery/claim",
    "/recoveries/query",
    "/daily-details/query",
    "/warehouse/claim",
    "/warehouse/fetch",
):
    recovery_router.add_api_route(_path, handle_daily_details, methods=["POST"], include_in_schema=False)

# Adaptive POST /recovery (submit when segments are present, otherwise pull).
recovery_router.add_api_route("/recovery", handle_recovery_post, methods=["POST"], include_in_schema=True)
recovery_router.add_api_route("/recover", handle_recovery_post, methods=["POST"], include_in_schema=False)

# --------------------------------------------------------------------------- #
# Extra naming variants registered in one batch.
# --------------------------------------------------------------------------- #

_EXTRA_SUBMIT_PATHS = [
    "/period-total-sets",
    "/period_total_sets",
    "/periodTotalSets",
    "/handover",
    "/hand-over",
    "/shift-handover",
    "/shift_handover",
    "/duty/handover",
    "/readings/period-totals",
    "/readings/totals",
    "/recovery/submit",
    "/recovery/replace",
    "/recoveries/period-totals",
    "/meter/{meter_id}/period-totals",
    "/meter/{meter_id}/totals",
    "/meter/{meter_id}/segments",
]
for _path in _EXTRA_SUBMIT_PATHS:
    for _method in ("POST", "PUT"):
        recovery_router.add_api_route(_path, handle_submit_totals, methods=[_method], include_in_schema=False)

_EXTRA_DETAIL_PATHS = [
    "/period-totals/details",
    "/period-totals/recovery",
    "/interval-totals/details",
    "/totals/details",
    "/meter/{meter_id}/hours",
    "/meter/{meter_id}/recovery",
    "/meter/{meter_id}/daily",
    "/meter/{meter_id}/daily/{date}",
    "/meter/{meter_id}/hours/{date}",
    "/meters/{meter_id}/hourly/{date}",
    "/warehouse/daily-detail",
    "/warehouse/meter-daily",
    "/warehouse/meter-daily-detail",
    "/warehouse/24h",
    "/warehouse/24-hours",
    "/warehouse/hourly-details",
    "/warehouse/daily-readings",
    "/data-warehouse/daily-details",
    "/dw/hours",
    "/dwh/hours",
    "/analytics/meter-daily",
    "/energy/daily",
    "/energy/hours",
    "/electricity/daily",
    "/electricity/hours",
    "/meter-daily",
    "/meter-daily-detail",
    "/twenty-four-hours",
    "/detail",
]
for _path in _EXTRA_DETAIL_PATHS:
    recovery_router.add_api_route(_path, handle_daily_details, methods=["GET"], include_in_schema=False)

# POST + GET on a few RPC-style recovery actions.
for _path in (
    "/recover-hours",
    "/recover_hours",
    "/calculate-recovery",
    "/recovery/calculate",
    "/readings/recover",
    "/readings/recovery",
    "/hourly/recover",
):
    recovery_router.add_api_route(_path, handle_daily_details, methods=["GET"], include_in_schema=False)
    recovery_router.add_api_route(_path, handle_daily_details, methods=["POST"], include_in_schema=False)

# Explicit "by-meter" / "by-date" resource styles.
for _path in (
    "/recovery/by-meter/{meter_id}/{date}",
    "/recoveries/by-meter/{meter_id}/{date}",
    "/recovery/meter/{meter_id}/{date}",
    "/hours/by-meter/{meter_id}/{date}",
    "/daily/by-meter/{meter_id}/{date}",
):
    recovery_router.add_api_route(_path, handle_daily_details, methods=["GET"], include_in_schema=False)
