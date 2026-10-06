"""Per-region injury model: features, simulator, engine (explanations, what-if), metrics and API."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text

from backend.app.database import migrate
from backend.app.injury.catalog import INJURY_TYPES, PREVENTION, REGIONS, SPORT_REGION_PRIOR, SPORTS
from backend.app.injury.engine import SCENARIOS, empty_daily, engine
from backend.app.injury.features import FEATURES, FIDX, GROUPS, daily_features
from backend.app.injury.simulate import sample_profile, simulate_athlete
from backend.app.services import apply_what_if, guest_daily

PROFILE = {"age": 20, "sex": "M", "sport": "Football", "height_cm": 175, "weight_kg": 68, "years_training": 5,
           "growth_cm": 0, "nordic": 0, "adductor": 0}
METRICS = json.loads((Path(__file__).resolve().parents[1] / "app" / "artifacts" / "injury_metrics.json").read_text())


def constant_calendar(n=42, load_per_day=400.0):
    d = empty_daily(n)
    d["load"][:] = load_per_day
    d["duration"][:] = load_per_day / 5
    d["rpe"][:] = 5.0
    d["session"][:] = 1
    d["sleep"][:] = 8.0
    d["wellness"][:] = 7.0
    return d


# --------------------------------------------------------------------------- catalog
def test_catalog_covers_every_region_and_sport():
    assert len(REGIONS) == 12 and len(INJURY_TYPES) == 6
    for s in SPORTS:
        assert set(SPORT_REGION_PRIOR[s]) == set(REGIONS)
    for r in REGIONS:
        assert PREVENTION[r]["prevention"] and PREVENTION[r]["return_to_play"]
    assert "Nordic" in " ".join(PREVENTION["hamstring"]["prevention"])
    assert "Copenhagen" in " ".join(PREVENTION["groin"]["prevention"])


# --------------------------------------------------------------------------- features
def test_feature_vector_shape_and_groups():
    X = daily_features(constant_calendar(), PROFILE)
    assert X.shape == (42, len(FEATURES))
    assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)
    assert X[-1, FIDX["sport_Football"]] == 1 and X[-1, FIDX["sport_Kabaddi"]] == 0
    assert abs(sum(X[-1, FIDX[f"sport_prior_{r}"]] for r in REGIONS) - 1) < 1e-9


def test_rolling_load_features_on_constant_load():
    x = daily_features(constant_calendar(n=160), PROFILE)[-1]   # long enough for the 28-day EWMA to converge
    assert x[FIDX["acute_7"]] == pytest.approx(7 * 400)
    assert x[FIDX["chronic_28w"]] == pytest.approx(7 * 400)
    assert x[FIDX["acwr_ra"]] == pytest.approx(1.0)
    assert x[FIDX["acwr_ewma"]] == pytest.approx(1.0, abs=0.02)
    assert x[FIDX["load_wow"]] == pytest.approx(0.0)
    assert x[FIDX["rest_days_7"]] == 0 and x[FIDX["sessions_7"]] == 7
    assert x[FIDX["sleep_debt_7"]] == pytest.approx(0.0)


def test_monotony_strain_and_spike():
    d = constant_calendar()
    d["load"][-7:] = [600, 0, 600, 0, 600, 0, 600]   # alternating week, +2400 vs 2800? -> 2400
    d["session"][-7:] = [1, 0, 1, 0, 1, 0, 1]
    x = daily_features(d, PROFILE)[-1]
    loads = np.array([600, 0, 600, 0, 600, 0, 600], float)
    mono = loads.mean() / loads.std()
    assert x[FIDX["monotony_7"]] == pytest.approx(mono, rel=0.02)
    assert x[FIDX["strain_7"]] == pytest.approx(loads.sum() * mono, rel=0.02)
    assert x[FIDX["rest_days_7"]] == 3
    spike = constant_calendar()
    spike["load"][-7:] = 800
    xs = daily_features(spike, PROFILE)[-1]
    assert xs[FIDX["acwr_ra"]] > 1.5 and xs[FIDX["load_wow"]] == pytest.approx(1.0)


def test_features_have_no_lookahead():
    d = constant_calendar()
    X1 = daily_features(d, PROFILE)
    d2 = {k: v.copy() for k, v in d.items()}
    d2["load"][30:] *= 3
    d2["sleep"][30:] = 4
    X2 = daily_features(d2, PROFILE)
    np.testing.assert_allclose(X1[:30], X2[:30], equal_nan=True)


def test_injury_history_features():
    X = daily_features(constant_calendar(), PROFILE, injuries=[(10, "hamstring", 24)], prior_regions={"knee": 1})
    assert X[5, FIDX["prev_hamstring"]] == 0 and X[-1, FIDX["prev_hamstring"]] == 1
    assert X[-1, FIDX["prev_knee"]] == 1 and X[-1, FIDX["prev_any"]] == 2
    assert X[-1, FIDX["days_since_injury"]] == 41 - 10


# --------------------------------------------------------------------------- simulator
def test_simulator_is_deterministic_and_plan_overrides_apply():
    prof = sample_profile(np.random.default_rng(3), "Kabaddi")
    a = simulate_athlete(np.random.default_rng(5), prof, season_days=56)
    b = simulate_athlete(np.random.default_rng(5), prof, season_days=56)
    np.testing.assert_array_equal(a["daily"]["load"], b["daily"]["load"])
    assert a["injuries"] == b["injuries"]
    calm = simulate_athlete(np.random.default_rng(9), prof, season_days=28, allow_injuries=False)
    spiky = simulate_athlete(np.random.default_rng(9), prof, season_days=28, allow_injuries=False, plan={"week_overrides": {-2: 2.0}})
    assert calm["injuries"] == [] and spiky["injuries"] == []
    # week index -2 is the last full week of the calendar ("this week" for the seeder)
    assert spiky["daily"]["load"][-7:].sum() > 1.4 * calm["daily"]["load"][-7:].sum()
    np.testing.assert_allclose(spiky["daily"]["load"][:-7], calm["daily"]["load"][:-7])


# --------------------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def athlete_row():
    engine.load()
    d = constant_calendar()
    d["load"][-7:] *= 1.8
    d["sleep"][-7:] = 6.0
    return d, daily_features(d, PROFILE, prior_regions={"hamstring": 1})[-1]


def test_predict_outputs_calibrated_probabilities(athlete_row):
    _, x = athlete_row
    p = engine.predict(x)
    assert set(p) == set(REGIONS) | {"any"}
    assert all(0 < float(v[0]) < 1 for v in p.values())
    s = engine.summarize(x, "Football")
    assert len(s["regions"]) == 12 and len(s["top_regions"]) == 3
    assert s["overall"]["band"] in {"low", "moderate", "high"}
    assert abs(sum(t["share"] for t in s["type_mix"]) - 1) < 0.01
    for r in s["regions"]:
        assert abs(sum(r["type_probs"].values()) - 1) < 0.01


def test_explanations_are_additive(athlete_row):
    _, x = athlete_row
    ex = engine.explain(x, "Football", ["any", "hamstring", "knee"])
    p = engine.predict(x)
    for t, e in ex.items():
        total = sum(d["contribution"] for d in e["drivers"])
        assert e["reference_probability"] + total == pytest.approx(float(p[t][0]), abs=2e-4)
        assert e["probability"] == pytest.approx(float(p[t][0]), abs=2e-4)


def test_counterfactuals_rerun_the_model(athlete_row):
    d, _ = athlete_row
    cf = engine.counterfactuals(d, PROFILE, [], {"hamstring": 1}, ["any", "hamstring"])
    keys = {c["scenario"] for c in cf}
    assert {"load_minus_15", "load_minus_30", "sleep_8h", "nordic", "adductor"} <= keys <= set(SCENARIOS)
    assert "skip_match" not in keys   # no match in the calendar
    for c in cf:
        for ch in c["changes"]:
            assert 0 < ch["after"] < 1 and 0 < ch["before"] < 1
    nordic = next(c for c in cf if c["scenario"] == "nordic")
    ham = next(ch for ch in nordic["changes"] if ch["target"] == "hamstring")
    assert ham["after"] <= ham["before"]   # Nordic programme never raises hamstring risk
    cf2 = engine.counterfactuals(d, {**PROFILE, "nordic": 1}, [], {}, ["any"])
    assert "nordic" not in {c["scenario"] for c in cf2}


def test_real_data_cross_check(athlete_row):
    d, _ = athlete_row
    d = {k: v.copy() for k, v in d.items()}
    d["distance"][:] = 8.0
    rc = engine.real_check(d)
    assert 0 < rc["daily_probability"] < 1 and "Lövdal" in rc["note"]


def test_guest_calendar_and_what_if_transforms():
    p = dict(duration_min=60, rpe=6, distance_km=5, sessions_last_7=5, sleep_hours=7, wellness=7, resting_hr=60,
             sprint_100m_s=13, vertical_jump_cm=40, matches_last_7=1, chronic_sessions_per_week=3)
    d = guest_daily(p)
    assert len(d["load"]) == 28
    assert d["session"][-7:].sum() == 5 and d["session"][-1] == 1 and d["match"][-7:].sum() == 1
    assert d["session"][:7].sum() == 3
    cut = apply_what_if(d, load_change_pct=-30)
    assert cut["load"][-7:].sum() == pytest.approx(0.7 * d["load"][-7:].sum())
    assert cut["load"][:21].sum() == pytest.approx(d["load"][:21].sum())
    rest = apply_what_if(d, extra_rest_days=1)
    assert rest["session"][-7:].sum() == 4 and rest["session"][-1] == 1


# --------------------------------------------------------------------------- recorded metrics
def test_metrics_file_beats_baselines():
    sim = METRICS["simulated"]
    any_cv = sim["grouped_cv"]["any"]
    assert any_cv["model"]["roc_auc"] > max(0.55, any_cv["baseline_sport_rate"]["roc_auc"])
    assert any_cv["model"]["brier"] < any_cv["baseline_prevalence"]["brier"]
    assert any_cv["oracle_true_hazard"]["roc_auc"] > any_cv["model"]["roc_auc"]   # ceiling sits above the model
    better = sum(sim["grouped_cv"][r]["model"]["roc_auc"] > sim["grouped_cv"][r]["baseline_sport_rate"]["roc_auc"] for r in REGIONS)
    assert better >= 8
    assert sim["temporal_unseen_athletes"]["any"]["model"]["roc_auc"] > 0.55
    assert METRICS["real"]["source"]["licence"] == "CC0 1.0" and METRICS["real"]["source"]["athletes"] == 74
    assert METRICS["features"] == FEATURES


# --------------------------------------------------------------------------- API
def test_regions_endpoint(client):
    r = client.get("/api/regions").json()
    assert [x["region"] for x in r] == REGIONS and all(x["prevention"] for x in r)
    assert len(client.get("/api/injury-types").json()) == 6


def test_athlete_injury_risk(client):
    r = client.get("/api/athletes/1/injury-risk")
    assert r.status_code == 200, r.text
    s = r.json()
    assert len(s["regions"]) == 12 and len(s["top_regions"]) == 3
    assert {"acwr", "monotony", "strain", "acute_7"} <= set(s["load"])
    assert "any" in s["drivers"] and all(t in s["drivers"] for t in s["top_regions"])
    assert s["real_data"]["cohort_daily_rate"] > 0
    assert client.get("/api/athletes/1/injury-risk", params={"explain": "none"}).json()["drivers"] is None
    assert client.get("/api/athletes/1/injury-risk", params={"explain": "bogus"}).status_code == 422


def test_region_detail(client):
    r = client.get("/api/athletes/1/injury-risk/hamstring")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["region"] == "hamstring" and 1 <= d["rank"] <= 12
    assert d["drivers"]["drivers"] and d["counterfactuals"]
    assert len(d["trend"]["dates"]) == len(d["trend"]["probability"]) >= 30
    assert "General guidance" in d["guidance"]["disclaimer"]
    assert d["previous_injuries_before_tracking"] == 1   # Meenakshi: previous hamstring strain
    assert client.get("/api/athletes/1/injury-risk/elbow").status_code == 422
    assert client.get("/api/athletes/99999/injury-risk/knee").status_code == 404


def test_risk_trend_and_what_if(client):
    t = client.get("/api/athletes/2/risk-trend").json()
    n = len(t["dates"])
    assert n >= 50 and len(t["overall"]) == n and all(len(v) == n for v in t["regions"].values())
    for k in ("ewma_acute", "ewma_chronic", "acwr", "monotony", "strain", "daily_load"):
        assert len(t["load"][k]) == n
    w = client.get("/api/athletes/2/what-if").json()
    assert w["scenarios"] and w["top_regions"]


def test_squad_heatmap(client):
    h = client.get("/api/squad/heatmap").json()
    assert [r["region"] for r in h["regions"]] == REGIONS
    assert len(h["athletes"]) >= 8
    row = h["athletes"][0]
    assert set(row["probabilities"]) == set(REGIONS) and set(row["relative_risk"]) == set(REGIONS)
    assert row["overall"] >= h["athletes"][-1]["overall"]
    k = client.get("/api/squad/heatmap", params={"sport": "Kabaddi"}).json()["athletes"]
    assert k and all(a["sport"] == "Kabaddi" for a in k)


def test_injury_log_crud_rescores(client):
    a = client.post("/api/athletes", json=dict(name="Log Tester", sport="Football", sex="M", age=21, height_cm=176, weight_kg=70)).json()
    aid = a["id"]
    assert client.get(f"/api/athletes/{aid}/injury-risk").status_code == 409   # no sessions yet
    for k in range(10, -1, -1):
        client.post("/api/sessions", json=dict(athlete_id=aid, session_date=(date.today() - timedelta(days=k)).isoformat(),
                                               duration_min=75, rpe=6, distance_km=6))
    before = {r["region"]: r["probability"] for r in client.get(f"/api/athletes/{aid}/injury-risk").json()["regions"]}
    r = client.post(f"/api/athletes/{aid}/injuries", json=dict(region="hamstring", injury_type="muscle_strain",
                                                              onset_date=(date.today() - timedelta(days=6)).isoformat(),
                                                              return_date=(date.today() - timedelta(days=1)).isoformat()))
    assert r.status_code == 201, r.text
    iid = r.json()["id"]
    after = {r["region"]: r["probability"] for r in client.get(f"/api/athletes/{aid}/injury-risk").json()["regions"]}
    assert after["hamstring"] > before["hamstring"]
    assert client.get(f"/api/athletes/{aid}").json()["athlete"]["previous_injuries"] == 1
    assert len(client.get(f"/api/athletes/{aid}/injuries").json()) == 1
    bad = dict(region="knee", injury_type="ligament_sprain", onset_date=date.today().isoformat(),
               return_date=(date.today() - timedelta(days=3)).isoformat())
    assert client.post(f"/api/athletes/{aid}/injuries", json=bad).status_code == 422
    future = dict(region="knee", injury_type="ligament_sprain", onset_date=(date.today() + timedelta(days=3)).isoformat())
    assert client.post(f"/api/athletes/{aid}/injuries", json=future).status_code == 422
    assert client.delete(f"/api/injuries/{iid}").status_code == 204
    assert client.delete(f"/api/injuries/{iid}").status_code == 404
    restored = {r["region"]: r["probability"] for r in client.get(f"/api/athletes/{aid}/injury-risk").json()["regions"]}
    assert restored["hamstring"] == pytest.approx(before["hamstring"], abs=1e-4)


def test_profile_fields_roundtrip_and_rescore(client):
    a = client.post("/api/athletes", json=dict(name="Profile Tester", sport="Hockey", sex="F", age=15, height_cm=160,
                                               weight_kg=50, injury_history="knee, ankle", growth_cm=4)).json()
    assert a["injury_history"] == ["knee", "ankle"] and a["growth_cm"] == 4 and a["nordic_program"] is False
    client.post("/api/sessions", json=dict(athlete_id=a["id"], session_date=date.today().isoformat(), duration_min=60, rpe=6))
    h0 = client.get(f"/api/athletes/{a['id']}/injury-risk").json()
    r = client.patch(f"/api/athletes/{a['id']}", json={"nordic_program": True, "injury_history": ["knee", "ankle", "hamstring"]})
    assert r.status_code == 200 and r.json()["nordic_program"] is True and "hamstring" in r.json()["injury_history"]
    h1 = client.get(f"/api/athletes/{a['id']}/injury-risk").json()
    assert h1["overall"]["probability"] != h0["overall"]["probability"] or h1["regions"] != h0["regions"]
    assert h1["load"]["history_padded"] is True
    assert client.post("/api/athletes", json=dict(name="Bad Region", sport="Hockey", age=15, height_cm=160, weight_kg=50,
                                                  injury_history=["elbow"])).status_code == 422


def test_analyze_returns_region_summary(client):
    body = client.post("/api/analyze", json={"sport": "Kabaddi", "matches_last_7": 2, "injury_history": "knee"}).json()
    inj = body["injury"]
    assert len(inj["regions"]) == 12 and inj["load"]["synthetic_history"] is True
    assert body["injury_probability"] == inj["overall"]["probability"]
    assert inj["drivers"]["any"]["drivers"]
    lighter = client.post("/api/analyze", json={"sport": "Kabaddi", "matches_last_7": 2, "injury_history": "knee",
                                                "load_change_pct": -40}).json()["injury"]
    assert lighter["load"]["acute_7"] < inj["load"]["acute_7"]
    athlete = client.post("/api/analyze", json={"athlete_id": 2, "load_change_pct": -30}).json()["injury"]
    assert athlete["load"]["synthetic_history"] is False


def test_injury_model_card(client):
    m = client.get("/api/injury-model").json()
    assert m["simulated"]["grouped_cv"]["hamstring"]["calibration"]["bins"]
    assert m["simulated"]["top_k_region"]["top3"]["model"] > m["simulated"]["top_k_region"]["top3"]["global_prior"]
    assert m["real"]["app_features"]["grouped_cv"]["roc_auc"] > 0.5
    assert any("10.34894/UWU9PV" in (s.get("doi") or "") for s in m["sources"])
    assert m["limitations"] and m["design"]
    h = client.get("/health").json()
    assert h["model"]["injury_ready"] is True and h["model"]["injury_regions"] == 12


def test_additive_migration(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/old.db")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE athletes (id INTEGER PRIMARY KEY, name TEXT)"))
        c.execute(text("CREATE TABLE sessions (id INTEGER PRIMARY KEY, athlete_id INTEGER)"))
        c.execute(text("CREATE TABLE predictions (id INTEGER PRIMARY KEY)"))
        c.execute(text("INSERT INTO athletes (name) VALUES ('Old Row')"))
    added = migrate(eng)
    assert "athletes.injury_history" in added and "predictions.region_risks" in added and "sessions.session_type" in added
    cols = {c["name"] for c in inspect(eng).get_columns("athletes")}
    assert {"injury_history", "growth_cm", "nordic_program", "adductor_program"} <= cols
    assert migrate(eng) == []   # idempotent
    with eng.begin() as c:
        assert c.execute(text("SELECT name, nordic_program FROM athletes")).one() == ("Old Row", 0)
