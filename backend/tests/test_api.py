from datetime import date, timedelta

from fastapi.testclient import TestClient

from backend.app.injury.catalog import REGIONS
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


# --------------------------------------------------------------------------- extended suite
VALID_ATHLETE = dict(name="Test Runner", sport="Athletics", sex="M", age=20, height_cm=175, weight_kg=65)


def _session(athlete_id, **kw):
    body = dict(athlete_id=athlete_id, session_date=date.today().isoformat(), duration_min=70, rpe=6)
    body.update(kw)
    return body


def test_root_health_alias_and_ui(client):
    assert client.get("/health").json()["ok"] is True
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Athlete Lens" in r.text
    js = client.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "text/css" in client.get("/styles.css").headers["content-type"]
    # unknown client-side route falls back to the SPA
    assert "text/html" in client.get("/athletes/3").headers["content-type"]


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert "Unknown API route" in r.json()["detail"]


def test_seed_roster_shape(client):
    athletes = client.get("/api/athletes").json()
    assert len(athletes) >= 8
    seeded = {a["name"]: a for a in athletes}
    assert "Meenakshi R" in seeded
    m = seeded["Meenakshi R"]
    assert m["session_count"] >= 30  # ~8 weeks of daily history
    assert m["latest_risk"] in {"low", "moderate", "high"}
    assert m["top_region"] in REGIONS
    assert 0 < m["latest_injury_probability"] < 1
    assert 0 <= m["latest_readiness"] <= 100


def test_athlete_filters(client):
    kabaddi = client.get("/api/athletes", params={"sport": "Kabaddi"}).json()
    assert kabaddi and all(a["sport"] == "Kabaddi" for a in kabaddi)
    assert client.get("/api/athletes", params={"q": "meenak"}).json()[0]["name"] == "Meenakshi R"
    assert client.get("/api/athletes", params={"risk": "extreme"}).status_code == 422
    assert client.get("/api/athletes", params={"sport": "Cricket"}).status_code == 422


def test_stats(client):
    s = client.get("/api/stats").json()
    assert s["athlete_count"] >= 8
    assert s["session_count"] >= 8 * 30
    assert sum(s["risk_bands"].values()) == s["athlete_count"]
    assert s["high_risk"] == s["risk_bands"]["high"]
    assert 0 < s["avg_readiness"] <= 100
    assert len(s["recent"]) <= 8 and s["recent"]
    assert s["watchlist"][0]["injury_risk"] in {"high", "moderate", "low"}


def test_athlete_crud_and_session_flow(client):
    r = client.post("/api/athletes", json=VALID_ATHLETE)
    assert r.status_code == 201, r.text
    athlete = r.json()
    aid = athlete["id"]
    assert athlete["session_count"] == 0 and athlete["latest_risk"] is None

    detail = client.get(f"/api/athletes/{aid}").json()
    assert detail["sessions"] == [] and detail["latest"] is None

    for d in (3, 2, 0):
        r = client.post("/api/sessions", json=_session(aid, session_date=(date.today() - timedelta(days=d)).isoformat()))
        assert r.status_code == 201, r.text
    logged = r.json()
    assert logged["session"]["athlete_id"] == aid
    assert logged["analysis"]["injury_risk"] in {"low", "moderate", "high"}
    assert len(logged["analysis"]["plan_72h"]) == 3

    detail = client.get(f"/api/athletes/{aid}").json()
    assert detail["athlete"]["session_count"] == 3
    assert len(detail["trend"]) == 3
    last = detail["trend"][-1]
    assert last["srpe_load"] == 70 * 6
    assert last["acute_7d"] == 3 * 420
    assert last["chronic_28d"] == 3 * 420 / 4
    assert last["acwr"] is None  # < 21 days of history: ratio withheld
    assert detail["latest"]["readiness_band"] in {"green", "amber", "red"}

    r = client.patch(f"/api/athletes/{aid}", json={"previous_injuries": 4, "city": "Theni"})
    assert r.status_code == 200 and r.json()["city"] == "Theni"

    sid = detail["sessions"][0]["id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert len(client.get(f"/api/athletes/{aid}/sessions").json()) == 2
    assert client.delete(f"/api/sessions/{sid}").status_code == 404

    assert client.delete(f"/api/athletes/{aid}").status_code == 204
    assert client.get(f"/api/athletes/{aid}").status_code == 404


def test_validation_errors_are_readable(client):
    r = client.post("/api/athletes", json={**VALID_ATHLETE, "age": 5})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"].startswith("age:")
    assert body["errors"][0]["field"] == "age"
    assert client.post("/api/athletes", json={**VALID_ATHLETE, "sport": "Cricket"}).status_code == 422
    assert client.post("/api/analyze", json={"sleep_hours": 30}).status_code == 422


def test_session_rules(client):
    assert client.post("/api/sessions", json=_session(999999)).status_code == 404
    future = (date.today() + timedelta(days=2)).isoformat()
    r = client.post("/api/sessions", json=_session(1, session_date=future))
    assert r.status_code == 422 and "future" in r.json()["detail"]
    assert client.post("/api/sessions", json=_session(1, duration_min=5)).status_code == 422


def test_seeded_trend_reports_acwr(client):
    t = client.get("/api/athletes/1").json()["trend"]
    assert len(t) >= 30
    assert t[0]["acwr"] is None and t[-1]["acwr"] is not None and t[-1]["acwr"] > 0


def test_what_if_uses_athlete_baseline(client):
    base = client.post("/api/analyze", json={"athlete_id": 8}).json()
    rested = client.post("/api/analyze", json={"athlete_id": 8, "sleep_hours": 9, "rest_days_last_7": 3, "rpe": 4,
                                               "sessions_last_7": 3, "wellness": 9, "load_change_pct": -30,
                                               "extra_rest_days": 2}).json()
    assert rested["injury_probability"] < base["injury_probability"]
    assert rested["readiness_score"] > base["readiness_score"]
    assert client.post("/api/analyze", json={"athlete_id": 424242}).status_code == 404


def test_analysis_is_deterministic_and_sensitive(client):
    good = dict(sleep_hours=8.5, rpe=5, rest_days_last_7=3, sessions_last_7=4, wellness=9)
    bad = dict(sleep_hours=4.5, rpe=9.5, rest_days_last_7=0, sessions_last_7=9, wellness=3, duration_min=140,
               chronic_sessions_per_week=4, chronic_duration_min=60, matches_last_7=2)
    a = client.post("/api/analyze", json=good).json()
    assert a == client.post("/api/analyze", json=good).json()
    b = client.post("/api/analyze", json=bad).json()
    assert b["injury_risk"] in {"moderate", "high"}
    assert b["injury_probability"] > 1.5 * a["injury_probability"]
    assert b["injury"]["load"]["acwr"] > 1.5 and b["injury"]["load"]["sleep_debt_7"] > 15
    assert b["overtraining"] is True
    assert b["plan_72h"][0]["intensity"] == "rest"
    assert a["readiness_score"] > b["readiness_score"]


def test_csv_upload(client):
    csv_text = (
        "name,sport,sex,age,height_cm,weight_kg,session_date,duration_min,rpe,sleep_hours\n"
        f"Csv Athlete,Hockey,F,18,160,52,{date.today().isoformat()},60,6,7.5\n"
        f"Csv Athlete,Hockey,F,18,160,52,{date.today().isoformat()},45,4,8\n"
        "Bad Age,Hockey,F,4,160,52,2026-01-01,60,6,7\n"
        "Bad Date,Hockey,F,18,160,52,not-a-date,60,6,7\n"
    )
    r = client.post("/api/upload-csv", files={"file": ("squad.csv", csv_text, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {**body, "rows": 4, "athletes_created": 1, "sessions_created": 2}
    assert [e["line"] for e in body["errors"]] == [4, 5]
    match = client.get("/api/athletes", params={"q": "Csv Athlete"}).json()
    assert len(match) == 1 and match[0]["session_count"] == 2


def test_csv_upload_rejects_bad_files(client):
    r = client.post("/api/upload-csv", files={"file": ("x.txt", "a,b\n1,2", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/upload-csv", files={"file": ("x.csv", "name,sport\nA,Hockey\n", "text/csv")})
    assert r.status_code == 400 and "Missing required columns" in r.json()["detail"]


def test_sample_csv_imports(client):
    with open("data/sample_athletes.csv", "rb") as f:
        r = client.post("/api/upload-csv", files={"file": ("sample_athletes.csv", f, "text/csv")})
    assert r.status_code == 200
    assert r.json()["sessions_created"] == 3 and r.json()["errors"] == []


def test_model_card_has_real_metrics(client):
    card = client.get("/api/model").json()
    perf = card["metrics"]["performance"]
    inj = card["metrics"]["injury_any_grouped_cv"]
    assert card["training_data"]["performance"]["cohort_size"] == 2400
    assert card["training_data"]["performance"]["test_rows"] == 480
    assert perf["holdout_mae"] < perf["baseline_mean_mae"]
    assert inj["model"]["roc_auc"] > inj["baseline_sport_rate"]["roc_auc"]
    assert card["training_data"]["injury"]["real"]["licence"] == "CC0 1.0"
    assert card["injury_features"] > 50
    assert card["limitations"] and any("SIMULATED" in item for item in card["limitations"])
    assert client.get("/api/sports").json()[0] == "Athletics"
