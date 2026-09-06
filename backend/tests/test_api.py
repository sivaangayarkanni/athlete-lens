from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_analyze_guest():
    with TestClient(app) as client:
        r = client.post(
            "/api/analyze",
            json={
                "age": 19,
                "sex": "F",
                "sport": "Athletics",
                "sprint_100m_s": 13.4,
                "vertical_jump_cm": 44,
                "rpe": 6,
                "sleep_hours": 7.5,
                "sessions_last_7": 4,
                "rest_days_last_7": 2,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["performance_index"] <= 100
        assert body["injury_risk"] in {"low", "moderate", "high"}
        assert body["recommendations"]


def test_athletes_seeded():
    with TestClient(app) as client:
        r = client.get("/api/athletes")
        assert r.status_code == 200
        assert len(r.json()) >= 1
