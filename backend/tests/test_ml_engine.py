import numpy as np

from backend.app.ml_engine import FEATURE_NAMES, AthleteLensModel, enrich, vectorize


def test_enrich_and_vectorize():
    row = enrich(dict(age=18, sex="F", sport="Cricket", years_training=2, height_cm=160, weight_kg=51.2,
                      previous_injuries=0, duration_min=70, distance_km=5, sprint_100m_s=14, vertical_jump_cm=40,
                      resting_hr=60, session_hr_avg=150, rpe=6, sleep_hours=7, wellness=7, sessions_last_7=7,
                      rest_days_last_7=1))
    assert row["sport"] == "Athletics"  # unknown sport falls back
    assert round(row["bmi"], 1) == 20.0
    assert row["acute_load"] == 70 * 6
    vec = vectorize(row)
    assert len(vec) == len(FEATURE_NAMES)
    assert sum(vec[-8:]) == 1.0


def test_training_from_scratch_writes_artifacts(tmp_path):
    model = AthleteLensModel(model_dir=tmp_path)
    meta = model.train(n=400)
    assert (tmp_path / "injury_rf.joblib").exists()
    assert meta["train_rows"] == 320 and meta["test_rows"] == 80
    reloaded = AthleteLensModel(model_dir=tmp_path)
    assert reloaded._try_load(tmp_path)
    x = np.zeros((1, len(FEATURE_NAMES)))
    assert np.allclose(model.clf.predict_proba(x), reloaded.clf.predict_proba(x))


def test_stale_artifacts_are_ignored(tmp_path):
    import joblib

    model = AthleteLensModel(model_dir=tmp_path)
    model.train(n=300)
    meta = joblib.load(tmp_path / "meta.joblib")
    meta["sklearn_version"] = "0.0.1"
    joblib.dump(meta, tmp_path / "meta.joblib")
    assert AthleteLensModel(model_dir=tmp_path)._try_load(tmp_path) is False
