from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Isolate this module's ledger before importing the application.
_TEST_DIR = tempfile.mkdtemp(prefix="meter-recovery-test-")
os.environ.setdefault("DB_PATH", str(Path(_TEST_DIR) / "recovery-ledger.db"))
os.environ.setdefault("SEED_DEMO_TREE", "false")
os.environ.setdefault("ALLOW_RESET", "true")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db, reset_database_file  # noqa: E402
from app.main import app  # noqa: E402

DAY = "2026-01-01"


@pytest.fixture()
def client():
    init_db()
    reset_database_file()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    reset_database_file()


def register(client: TestClient, meter_id: str = "M") -> None:
    response = client.post("/meters", json=[{"meter_id": meter_id}])
    assert response.status_code == 200, response.text


def report(client: TestClient, meter_id: str, hour: int, amount: int, day: str = DAY) -> None:
    response = client.post(
        "/readings",
        json={"meter_id": meter_id, "hour": f"{day}T{hour:02d}:00:00Z", "amount": amount},
    )
    assert response.status_code == 200, response.text


def submit(client: TestClient, payload: dict, path: str = "/period-totals"):
    return client.post(path, json=payload)


def details(client: TestClient, meter_id: str = "M", day: str = DAY, path: str = "/recovery"):
    response = client.get(path, params={"meter_id": meter_id, "date": day})
    assert response.status_code == 200, response.text
    return response.json()


def test_simple_unique_recovery(client: TestClient):
    register(client)
    report(client, "M", 0, 100)
    response = submit(
        client,
        {
            "meter_id": "M",
            "date": DAY,
            "segments": [{"start_hour": 0, "end_hour": 2, "total_wh": 150}],
        },
    )
    assert response.status_code == 200, response.text

    body = details(client)
    assert len(body["hours"]) == 24
    by_index = {h["index"]: h for h in body["hours"]}
    assert by_index[0]["status"] == "original"
    assert by_index[0]["amount"] == 100
    assert by_index[1]["status"] == "recovered"
    assert by_index[1]["amount"] == 50
    assert by_index[2]["status"] == "pending"
    assert by_index[2]["amount"] is None
    assert body["original_count"] == 1
    assert body["recovered_count"] == 1
    assert body["pending_count"] == 22
    assert body["conflict"] is False


def test_non_unique_split_stays_pending(client: TestClient):
    register(client)
    submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 0, "end": 2, "total": 10}]},
    )
    body = details(client)
    assert body["hours"][0]["status"] == "pending"
    assert body["hours"][1]["status"] == "pending"
    assert body["recovered_count"] == 0


def test_overlapping_segments_pin_hours(client: TestClient):
    register(client)
    submit(
        client,
        {
            "meter_id": "M",
            "date": DAY,
            "segments": [
                {"start": 0, "end": 3, "total": 300},
                {"start": 0, "end": 1, "total": 100},
                {"start": 1, "end": 2, "total": 50},
            ],
        },
    )
    body = details(client)
    assert [body["hours"][i]["amount"] for i in range(3)] == [100, 50, 150]
    assert all(body["hours"][i]["status"] == "recovered" for i in range(3))
    assert body["hours"][3]["status"] == "pending"


def test_zero_total_recovers_zeros(client: TestClient):
    register(client)
    submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 2, "end": 5, "total": 0}]},
    )
    body = details(client)
    for i in (2, 3, 4):
        assert body["hours"][i]["status"] == "recovered"
        assert body["hours"][i]["amount"] == 0


def test_full_day_total_subtraction(client: TestClient):
    register(client)
    submit(client, {"meter_id": "M", "date": DAY, "segments": [{"start": 0, "end": 24, "total": 2400}]})
    for hour in range(24):
        if hour != 5:
            report(client, "M", hour, 100)
    body = details(client)
    assert body["original_count"] == 23
    assert body["hours"][5]["status"] == "recovered"
    assert body["hours"][5]["amount"] == 100


def test_infeasible_replacement_keeps_previous(client: TestClient):
    register(client)
    report(client, "M", 0, 6)
    first = submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 0, "end": 1, "total": 6}]},
    )
    assert first.status_code == 200
    assert first.json()["version"] == 1

    bad = submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 0, "end": 1, "total": 999}]},
    )
    assert bad.status_code in (409, 422)

    totals = client.get("/period-totals", params={"meter_id": "M", "date": DAY}).json()
    assert totals["version"] == 1
    assert totals["segments"][0]["total"] == 6

    body = details(client)
    assert body["hours"][0]["status"] == "original"
    assert body["hours"][0]["amount"] == 6


def test_later_conflict_flags_and_then_auto_resumes(client: TestClient):
    register(client)
    submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 0, "end": 2, "total": 150}]},
    )
    report(client, "M", 0, 100)
    assert details(client)["hours"][1]["amount"] == 50

    # Correction makes the system infeasible.
    response = client.put(
        "/readings",
        json={"meter_id": "M", "hour": f"{DAY}T00:00:00Z", "amount": 160},
    )
    assert response.status_code == 200
    body = details(client)
    assert body["conflict"] is True
    assert body["feasible"] is False
    assert body["status"] == "conflict"
    assert body["hours"][0]["status"] == "original"
    assert body["hours"][0]["amount"] == 160
    # Only originals are supplied: no recovered guesses.
    assert body["hours"][1]["amount"] is None
    assert body["recovered_count"] == 0

    # Restore compatibility and inference resumes automatically.
    client.put(
        "/readings",
        json={"meter_id": "M", "hour": f"{DAY}T00:00:00Z", "amount": 100},
    )
    body = details(client)
    assert body["conflict"] is False
    assert body["hours"][1]["status"] == "recovered"
    assert body["hours"][1]["amount"] == 50


def test_withdrawal_can_restore_compatibility(client: TestClient):
    register(client, "W")
    submit(
        client,
        {"meter_id": "W", "date": DAY, "segments": [{"start": 0, "end": 1, "total": 6}]},
    )
    report(client, "W", 0, 9)
    assert details(client, "W")["conflict"] is True
    response = client.request(
        "DELETE", "/readings", json={"meter_id": "W", "hour": f"{DAY}T00:00:00Z"}
    )
    assert response.status_code == 200
    body = details(client, "W")
    assert body["conflict"] is False
    assert body["hours"][0]["status"] == "recovered"
    assert body["hours"][0]["amount"] == 6


def test_segment_outside_day_rejected(client: TestClient):
    register(client)
    response = submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 23, "end": 25, "total": 1}]},
    )
    assert response.status_code == 422
    # Nothing was saved.
    totals = client.get("/period-totals", params={"meter_id": "M", "date": DAY}).json()
    assert totals["segments"] == []


def test_unknown_meter_rejected(client: TestClient):
    response = submit(
        client,
        {"meter_id": "NOPE", "date": DAY, "segments": []},
    )
    assert response.status_code == 404
    response = client.get("/recovery", params={"meter_id": "NOPE", "date": DAY})
    assert response.status_code == 404


def test_multiple_path_aliases_return_24_hours(client: TestClient):
    register(client)
    submit(
        client,
        {"meter_id": "M", "date": DAY, "segments": [{"start": 3, "end": 4, "total": 77}]},
    )
    paths = [
        "/recovery/M/" + DAY,
        "/recoveries/M/" + DAY,
        "/meters/M/hours",
        "/meters/M/recovery/" + DAY,
        "/daily-details",
        "/hourly-readings",
        "/warehouse/hours",
        "/api/v1/recovery",
    ]
    for path in paths:
        response = client.get(path, params={"meter_id": "M", "date": DAY})
        assert response.status_code == 200, (path, response.status_code, response.text)
        body = response.json()
        assert len(body["hours"]) == 24
        assert body["hours"][3]["amount"] == 77


def test_compact_date_and_clock_segments(client: TestClient):
    register(client)
    response = submit(
        client,
        {
            "meter_id": "M",
            "date": "20260101",
            "segments": [{"start": "00:00", "end": "01:00", "total": 42}],
        },
    )
    assert response.status_code == 200, response.text
    body = details(client, day="20260101")
    assert body["date"] == DAY
    assert body["hours"][0]["amount"] == 42
