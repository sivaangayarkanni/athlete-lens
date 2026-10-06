"""Train + evaluate the injury models and write artifacts.

    python -m backend.app.injury.train            # full run (~3,000 simulated athletes + real data)
    python -m backend.app.injury.train --quick    # small smoke run

Outputs (backend/app/artifacts):
    injury_models.joblib   calibrated per-region + overall + injury-type + real-data models
    injury_metrics.json    every reported number (metrics, calibration, importance, baselines)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import realdata
from .catalog import INJURY_TYPES, REGIONS, SPORTS
from .features import FEATURES, FIDX, GROUPS
from .simulate import BURN_IN, simulate_cohort

ART = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 7


def hgb(n_rows: int) -> HistGradientBoostingClassifier:
    """Shallow, heavily regularised boosting.

    Rows from one athlete are strongly correlated (overlapping 7-day windows, constant profile),
    so deeper trees with small leaves memorise which athletes got injured. Large leaves (3% of the
    rows) and 4-leaf trees generalised best to unseen athletes in grouped CV (see README).
    """
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=max(100, int(0.03 * n_rows)),
        l2_regularization=10.0, early_stopping=False, random_state=0)


def type_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=100,
                                          l2_regularization=10.0, early_stopping=False, random_state=0)


def holdout_calibrated(X, y, groups, method, rare, seed=0):
    """Evaluation helper: fit HGB on 80% of the training athletes, calibrate on the other 20%."""
    ath = np.unique(groups)
    rng = np.random.default_rng(seed)
    cal_ath = set(rng.choice(ath, size=max(1, len(ath) // 5), replace=False).tolist())
    is_cal = np.array([g in cal_ath for g in groups])
    m = hgb(int((~is_cal).sum())).fit(X[~is_cal], y[~is_cal])
    s = m.predict_proba(X[is_cal])[:, 1]
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip").fit(s, y[is_cal])
        return lambda Z: cal.predict(m.predict_proba(Z)[:, 1])
    lr = LogisticRegression(C=1e4).fit(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1), y[is_cal])

    def pred(Z):
        q = np.clip(m.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        return lr.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]
    return pred


class _PredWrap:
    def __init__(self, f):
        self.f = f

    def predict_proba(self, Z):
        p = self.f(Z)
        return np.column_stack([1 - p, p])


def calibrated(est, method, splits):
    return CalibratedClassifierCV(est, method=method, cv=splits, ensemble=False)


def _fit_cal(X, y, groups, method, rare, inner_folds=3):
    """Fit HGB, then calibrate on grouped out-of-fold predictions (CalibratedClassifierCV, ensemble=False)."""
    splits = list(GroupKFold(inner_folds).split(X, y, groups))
    return calibrated(hgb(int(len(X) * (inner_folds - 1) / inner_folds)), method, splits).fit(X, y)


def metrics(y, p) -> dict:
    out = {"prevalence": round(float(y.mean()), 5), "n_pos": int(y.sum()), "n": int(len(y))}
    if 0 < y.sum() < len(y):
        out.update(roc_auc=round(float(roc_auc_score(y, p)), 3), pr_auc=round(float(average_precision_score(y, p)), 4))
    out["brier"] = round(float(brier_score_loss(y, p)), 5)
    return out


def reliability(y, p, bins=10) -> dict:
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    pts, ece = [], 0.0
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        pts.append({"pred": round(mp, 4), "obs": round(my, 4), "n": int(m.sum())})
        ece += m.mean() * abs(mp - my)
    return {"bins": pts, "ece": round(float(ece), 4)}


def sport_prior_scores(y_tr, sport_tr, sport_te):
    rates = {s: (y_tr[sport_tr == s].mean() if (sport_tr == s).any() else y_tr.mean()) for s in np.unique(np.concatenate([sport_tr, sport_te]))}
    return np.array([rates[s] for s in sport_te])


def grouped_permutation(model, X, y, rng, repeats=2):
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for g, cols in GROUPS.items():
        idx = [FIDX[c] for c in cols]
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[g] = round(float(np.mean(drops)), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def train_simulated(n_athletes: int, folds: int = 5, log=print):
    t0 = time.perf_counter()
    c = simulate_cohort(n_athletes, seed=SEED, stride=2)
    X, Y, groups, days, sport = c["X"], c["Y"], c["groups"], c["days"], c["sport"]
    y_any = Y.max(1)
    log(f"simulated {n_athletes} athletes, {len(X):,} athlete-days, {c['n_injuries']} injuries in {time.perf_counter()-t0:.0f}s")
    targets = {"any": y_any, **{r: Y[:, i] for i, r in enumerate(REGIONS)}}

    # ---------------- grouped (athlete-level) K-fold, out-of-fold predictions
    gkf = list(GroupKFold(folds).split(X, y_any, groups))
    oof = {k: np.zeros(len(X)) for k in targets}
    base_sport = {k: np.zeros(len(X)) for k in targets}
    base_acwr = np.zeros(len(X))
    base_lr = np.zeros(len(X))
    acwr_col = FIDX["acwr_ewma"]
    for f, (tr, te) in enumerate(gkf):
        for k, y in targets.items():
            # within-fold calibration on held-out training athletes (no leakage into the test fold)
            pred = holdout_calibrated(X[tr], y[tr], groups[tr], "sigmoid", k != "any", seed=f)
            oof[k][te] = pred(X[te])
            base_sport[k][te] = sport_prior_scores(y[tr], sport[tr], sport[te])
        a_tr = np.nan_to_num(X[tr, acwr_col], nan=1.0).reshape(-1, 1)
        lr = LogisticRegression().fit(np.column_stack([a_tr, np.maximum(a_tr - 1.3, 0)]), y_any[tr])
        a_te = np.nan_to_num(X[te, acwr_col], nan=1.0).reshape(-1, 1)
        base_acwr[te] = lr.predict_proba(np.column_stack([a_te, np.maximum(a_te - 1.3, 0)]))[:, 1]
        lr_all = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        base_lr[te] = lr_all.fit(X[tr], y_any[tr]).predict_proba(X[te])[:, 1]
        log(f"  fold {f+1}/{folds} done ({time.perf_counter()-t0:.0f}s)")

    grouped = {}
    for k, y in targets.items():
        grouped[k] = {
            "model": metrics(y, oof[k]),
            "baseline_prevalence": {"roc_auc": 0.5, "pr_auc": round(float(y.mean()), 5), "brier": round(float(brier_score_loss(y, np.full(len(y), y.mean()))), 5)},
            "baseline_sport_rate": metrics(y, base_sport[k]),
            "calibration": reliability(y, oof[k]),
        }
    grouped["any"]["baseline_acwr_logistic"] = metrics(y_any, base_acwr)
    grouped["any"]["baseline_logistic_all_features"] = metrics(y_any, base_lr)
    # Ceiling: the simulator's TRUE hazard summed over the next 7 days (knows latent frailty and the future plan).
    grouped["any"]["oracle_true_hazard"] = metrics(y_any, 1 - np.exp(-c["oracle"]))

    # ---------------- top-k region accuracy on windows that contain an injury
    inj = Y.sum(1) > 0
    P = np.column_stack([oof[r] for r in REGIONS])[inj]
    S = np.column_stack([base_sport[r] for r in REGIONS])[inj]
    G = np.tile(Y.mean(0), (inj.sum(), 1))
    Yt = Y[inj]

    def topk(scores, k):
        top = np.argsort(-scores, axis=1)[:, :k]
        return round(float(np.mean([Yt[i, top[i]].any() for i in range(len(Yt))])), 3)

    top_k = {f"top{k}": {"model": topk(P, k), "sport_prior": topk(S, k), "global_prior": topk(G, k)} for k in (1, 3)}

    # ---------------- temporal split on unseen athletes
    cut = BURN_IN + int(0.67 * c["season_days"])
    rng = np.random.default_rng(SEED)
    test_ath = set(rng.choice(n_athletes, size=n_athletes // 5, replace=False).tolist())
    is_test_ath = np.array([g in test_ath for g in groups])
    tr = (~is_test_ath) & (days < cut)
    te = is_test_ath & (days >= cut + 7)
    temporal, perm_imp = {}, {}
    for k, y in targets.items():
        m = _PredWrap(holdout_calibrated(X[tr], y[tr], groups[tr], "sigmoid", k != "any"))
        p = m.predict_proba(X[te])[:, 1]
        temporal[k] = {"model": metrics(y[te], p), "baseline_sport_rate": metrics(y[te], sport_prior_scores(y[tr], sport[tr], sport[te]))}
        if k in ("any", "hamstring", "knee", "ankle", "head_neck", "shoulder", "groin", "calf_achilles"):
            perm_imp[k] = grouped_permutation(m, X[te], y[te], rng)
    log(f"temporal split done ({time.perf_counter()-t0:.0f}s)")

    # ---------------- final models on all simulated data
    final = {k: _fit_cal(X, y, groups, "sigmoid", k != "any") for k, y in targets.items()}
    log(f"final models done ({time.perf_counter()-t0:.0f}s)")

    # ---------------- injury type model (on injury events)
    E = np.array([e[0] for e in c["events"]])
    er = np.array([e[1] for e in c["events"]])
    et = np.array([e[2] for e in c["events"]])
    eg = np.array([e[3] for e in c["events"]])
    XE = np.column_stack([E, np.eye(len(REGIONS))[er]])
    type_oof = np.zeros((len(XE), len(INJURY_TYPES)))
    maj_oof = np.zeros(len(XE), dtype=int)
    freq_oof = np.zeros((len(XE), len(INJURY_TYPES)))
    for tr_i, te_i in GroupKFold(folds).split(XE, et, eg):
        tm = type_hgb().fit(XE[tr_i], et[tr_i])
        pr = tm.predict_proba(XE[te_i])
        full = np.zeros((len(te_i), len(INJURY_TYPES)))
        full[:, tm.classes_] = pr
        type_oof[te_i] = full
        for i in te_i:
            same = et[tr_i][er[tr_i] == er[i]]
            cnt = np.bincount(same, minlength=len(INJURY_TYPES)).astype(float)
            maj_oof[i] = cnt.argmax() if len(same) else 0
            freq_oof[i] = (cnt + 0.01) / (cnt + 0.01).sum()
    eps = np.clip(type_oof, 1e-6, 1)
    type_metrics = {
        "n_events": int(len(XE)),
        "model_accuracy": round(float(accuracy_score(et, type_oof.argmax(1))), 3),
        "model_log_loss": round(float(log_loss(et, eps / eps.sum(1, keepdims=True), labels=list(range(len(INJURY_TYPES))))), 3),
        "baseline_region_majority_accuracy": round(float(accuracy_score(et, maj_oof)), 3),
        "baseline_region_frequency_log_loss": round(float(log_loss(et, freq_oof, labels=list(range(len(INJURY_TYPES))))), 3),
    }
    # Region -> type frequency table (the baseline). Ship whichever generalises better (grouped-CV log loss).
    type_freq = np.zeros((len(REGIONS), len(INJURY_TYPES)))
    for ri in range(len(REGIONS)):
        cnt = np.bincount(et[er == ri], minlength=len(INJURY_TYPES)).astype(float)
        type_freq[ri] = (cnt + 0.01) / (cnt + 0.01).sum()
    type_source = "model" if type_metrics["model_log_loss"] < type_metrics["baseline_region_frequency_log_loss"] else "frequency"
    type_metrics["shipped"] = type_source
    type_model = type_hgb().fit(XE, et)

    # ---------------- reference rows (explanation baseline) and per-sport average risk
    reference = {}
    sport_mean = {}
    for si, s in enumerate(SPORTS):
        m = sport == si
        reference[s] = np.nanmedian(X[m], axis=0)
        sport_mean[s] = {k: round(float(oof[k][m].mean()), 5) for k in targets}
    q = np.quantile(oof["any"], [0.6, 0.85])
    bands = {"moderate": round(float(q[0]), 4), "high": round(float(q[1]), 4)}

    summary = {
        "n_athletes": n_athletes, "season_days": c["season_days"], "rows": int(len(X)), "injuries": c["n_injuries"],
        "injuries_per_athlete_season": round(c["n_injuries"] / n_athletes, 2), "horizon_days": 7,
        "grouped_cv": grouped, "top_k_region": top_k, "temporal_unseen_athletes": temporal,
        "temporal_split": {"train": f"80% of athletes, days < {cut - BURN_IN}", "test": f"other 20% of athletes, days >= {cut - BURN_IN + 7}"},
        "permutation_importance_auc_drop": perm_imp, "injury_type": type_metrics, "risk_bands_any": bands,
        "label_prevalence": {k: round(float(y.mean()), 5) for k, y in targets.items()},
    }
    models = {"regions": {r: final[r] for r in REGIONS}, "any": final["any"], "type": type_model,
              "type_freq": type_freq, "type_source": type_source,
              "reference": reference, "sport_mean": sport_mean, "bands": bands}
    return models, summary


def train_real(folds: int = 5, log=print):
    X, Xf, y, g, date = realdata.load()
    out = {"source": {"doi": "10.34894/UWU9PV", "licence": "CC0 1.0", "rows": int(len(y)), "athletes": int(len(np.unique(g))),
                      "positives": int(y.sum()), "prevalence": round(float(y.mean()), 5),
                      "label": "injury on the day following a 3-week window (binary, no body region)"}}

    def model():
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=3.0, random_state=0)

    for name, XX in (("app_features", X), ("all_features_benchmark", Xf)):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(folds).split(XX, y, g):
            oof[te] = model().fit(XX[tr], y[tr]).predict_proba(XX[te])[:, 1]
        res = {"grouped_cv": metrics(y, oof), "calibration": reliability(y, oof)}
        cut = np.quantile(date, 0.75)
        tr, te = date < cut, date >= cut + 21
        p = model().fit(XX[tr], y[tr]).predict_proba(XX[te])[:, 1]
        res["temporal"] = metrics(y[te], p)
        res["temporal_note"] = f"train Date < {int(cut)}, test Date >= {int(cut) + 21} (3-week gap; athletes overlap)"
        out[name] = res
        log(f"real {name}: grouped AUC {res['grouped_cv'].get('roc_auc')} temporal AUC {res['temporal'].get('roc_auc')}")
    out["baseline_prevalence"] = {"roc_auc": 0.5, "pr_auc": out["source"]["prevalence"],
                                  "brier": round(float(brier_score_loss(y, np.full(len(y), y.mean()))), 5)}
    # univariate validation of load features on real data (direction + AUC)
    uni = {}
    for i, f in enumerate(realdata.APP_FEATURES):
        v = X[:, i]
        ok = ~np.isnan(v)
        if ok.sum() > 1000 and np.unique(v[ok]).size > 2:
            auc = roc_auc_score(y[ok], v[ok])
            uni[f] = round(float(auc), 3)
    out["univariate_auc"] = dict(sorted(uni.items(), key=lambda kv: -abs(kv[1] - 0.5)))
    # injury rate by recent-week km spike bucket
    r = X[:, realdata.APP_FEATURES.index("km_ratio_w1_chronic")]
    buckets = [(0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 99)]
    out["injury_rate_by_km_ratio"] = [
        {"range": f"{a}-{b if b < 99 else '+'}", "n": int(((r >= a) & (r < b)).sum()),
         "rate": round(float(y[(r >= a) & (r < b)].mean()), 4) if ((r >= a) & (r < b)).sum() else None}
        for a, b in buckets]
    final = model().fit(X, y)
    iso_oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        iso_oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(iso_oof, y)   # calibrate on grouped OOF scores
    return {"model": final, "calibrator": iso, "features": realdata.APP_FEATURES}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athletes", type=int, default=3000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ART))
    a = ap.parse_args()
    n = 300 if a.quick else a.athletes
    t0 = time.perf_counter()
    models, sim_summary = train_simulated(n, folds=3 if a.quick else 5)
    real_model, real_summary = train_real(folds=3 if a.quick else 5)
    models["real"] = real_model
    models["features"] = FEATURES
    models["sklearn_version"] = sklearn.__version__
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sklearn_version": sklearn.__version__,
        "train_seconds": round(time.perf_counter() - t0, 1), "seed": SEED, "features": FEATURES,
        "simulated": sim_summary, "real": real_summary,
    }
    models["meta"] = {k: meta[k] for k in ("trained_at", "sklearn_version", "seed")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out / "injury_models.joblib", compress=3)
    (out / "injury_metrics.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"any": sim_summary["grouped_cv"]["any"]["model"], "top_k": sim_summary["top_k_region"],
                      "type": sim_summary["injury_type"], "real_app": real_summary["app_features"]["grouped_cv"],
                      "seconds": meta["train_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
