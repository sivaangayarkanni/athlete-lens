"""Serving-side injury engine: per-region risk, explanations, counterfactuals, real-data cross-check."""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .catalog import INJURY_TYPES, PREVENTION, REGION_LABELS, REGION_TYPES, REGIONS, TYPE_LABELS
from .features import DAILY_KEYS, FEATURES, FIDX, GROUP_LABELS, GROUPS, daily_features
from . import realdata

log = logging.getLogger("athlete_lens.injury")
ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_FILE = ART / "injury_models.joblib"
METRICS_FILE = ART / "injury_metrics.json"
EXPLAIN_GROUPS = [g for g in GROUPS if g not in ("sport", "sex_body")]   # sport & sex stay at the athlete's own value

SCENARIOS = {
    "load_minus_15": "Cut this week's training load by 15%",
    "load_minus_30": "Cut this week's training load by 30%",
    "extra_rest_day": "Replace the hardest session this week with a rest day",
    "sleep_8h": "Sleep at least 8 h every night this week",
    "skip_match": "Skip competition this week (no match exposure)",
    "nordic": "Add a Nordic hamstring programme",
    "adductor": "Add the Copenhagen adductor programme",
}


def level(rr: float) -> str:
    return "high" if rr >= 2.0 else "elevated" if rr >= 1.3 else "typical" if rr >= 0.7 else "low"


class InjuryEngine:
    def __init__(self, path: Path = MODEL_FILE):
        self.path = path
        self.m: dict | None = None
        self.metrics: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        if self.m is not None:
            return
        with self._lock:
            if self.m is not None:
                return
            m = joblib.load(self.path) if self.path.exists() else None
            if m is None or m.get("sklearn_version") != sklearn.__version__ or m.get("features") != FEATURES:
                raise RuntimeError(f"injury artifacts missing or stale at {self.path}; run python -m backend.app.injury.train")
            self.m = m
            if METRICS_FILE.exists():
                self.metrics = json.loads(METRICS_FILE.read_text())

    @property
    def ready(self) -> bool:
        return self.m is not None

    # ------------------------------------------------------------------ core prediction
    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        self.load()
        X = np.atleast_2d(X)
        out = {r: self.m["regions"][r].predict_proba(X)[:, 1] for r in REGIONS}
        out["any"] = self.m["any"].predict_proba(X)[:, 1]
        return out

    def band(self, p_any: float) -> str:
        b = self.m["bands"]
        return "high" if p_any >= b["high"] else "moderate" if p_any >= b["moderate"] else "low"

    def type_probs(self, x: np.ndarray, region: str) -> dict[str, float]:
        if self.m.get("type_source") == "frequency":   # the model did not beat the region frequency table in CV
            raw = self.m["type_freq"][REGIONS.index(region)]
        else:
            xe = np.concatenate([x, np.eye(len(REGIONS))[REGIONS.index(region)]]).reshape(1, -1)
            tm = self.m["type"]
            raw = np.zeros(len(INJURY_TYPES))
            raw[tm.classes_] = tm.predict_proba(xe)[0]
        allowed = REGION_TYPES[region]
        probs = {t: float(raw[INJURY_TYPES.index(t)]) for t in allowed}
        s = sum(probs.values()) or 1.0
        return dict(sorted(((t, round(v / s, 3)) for t, v in probs.items()), key=lambda kv: -kv[1]))

    def summarize(self, x: np.ndarray, sport: str, with_types: bool = True) -> dict:
        """Full per-region summary for one feature row."""
        p = {k: float(v[0]) for k, v in self.predict(x).items()}
        means = self.m["sport_mean"].get(sport, self.m["sport_mean"]["Athletics"])
        regions = []
        for r in REGIONS:
            rr = p[r] / max(means[r], 1e-6)
            item = {"region": r, "label": REGION_LABELS[r], "probability": round(p[r], 4),
                    "relative_risk": round(rr, 2), "level": level(rr), "sport_average": means[r]}
            if with_types:
                tp = self.type_probs(x, r)
                item["likely_type"] = next(iter(tp))
                item["likely_type_label"] = TYPE_LABELS[item["likely_type"]]
                item["type_probs"] = tp
            regions.append(item)
        likely = sorted(regions, key=lambda i: -i["probability"])        # most likely (absolute 7-day probability)
        elevated = sorted(regions, key=lambda i: -i["relative_risk"])   # most elevated vs the sport average
        # expected injury-type mix: sum_r p_r * P(type | region)
        mix = {t: 0.0 for t in INJURY_TYPES}
        if with_types:
            for i in regions:
                for t, v in i["type_probs"].items():
                    mix[t] += i["probability"] * v
        tot = sum(mix.values()) or 1.0
        return {
            "overall": {"probability": round(p["any"], 4), "band": self.band(p["any"]),
                        "relative_risk": round(p["any"] / max(means["any"], 1e-6), 2), "sport_average": means["any"],
                        "horizon_days": 7},
            "regions": regions,
            "top_regions": [i["region"] for i in likely[:3]],
            "elevated_regions": [i["region"] for i in elevated[:3] if i["relative_risk"] >= 1.3],
            "type_mix": [{"type": t, "label": TYPE_LABELS[t], "share": round(v / tot, 3)} for t, v in sorted(mix.items(), key=lambda kv: -kv[1]) if v > 0],
        }

    # ------------------------------------------------------------------ explanations (sampled Shapley over feature groups)
    def explain(self, x: np.ndarray, sport: str, targets: list[str], n_perm: int = 24, seed: int = 0) -> dict:
        """Group-level Shapley values with a single reference athlete (same sport, cohort medians).

        Each value is the average change in calibrated probability when the group switches from
        the reference value to this athlete's value, over random group orderings. Values sum to
        p(athlete) - p(reference).
        """
        self.load()
        ref = self.m["reference"].get(sport, self.m["reference"]["Athletics"]).copy()
        for g in ("sport", "sex_body"):
            for f in GROUPS[g]:
                ref[FIDX[f]] = x[FIDX[f]]
        idx = {g: [FIDX[f] for f in GROUPS[g]] for g in EXPLAIN_GROUPS}
        rng = np.random.default_rng(seed)
        G = len(EXPLAIN_GROUPS)
        orders = [rng.permutation(G) for _ in range(n_perm // 2)]
        orders += [o[::-1] for o in orders]           # antithetic orderings reduce variance
        rows, meta = [], []
        for o in orders:
            cur = ref.copy()
            rows.append(cur.copy())
            for gi in o:
                cur[idx[EXPLAIN_GROUPS[gi]]] = x[idx[EXPLAIN_GROUPS[gi]]]
                rows.append(cur.copy())
            meta.append(o)
        R = np.array(rows)
        preds = {}
        for t in targets:
            model = self.m["any"] if t == "any" else self.m["regions"][t]
            preds[t] = model.predict_proba(R)[:, 1]
        out = {}
        for t in targets:
            phi = np.zeros(G)
            pr = preds[t]
            k = 0
            for o in meta:
                seg = pr[k:k + G + 1]
                for step, gi in enumerate(o):
                    phi[gi] += seg[step + 1] - seg[step]
                k += G + 1
            phi /= len(meta)
            p_ref = float(pr[0])
            items = [{"group": g, "label": GROUP_LABELS[g], "contribution": round(float(phi[i]), 5)} for i, g in enumerate(EXPLAIN_GROUPS)]
            items.sort(key=lambda d: -abs(d["contribution"]))
            out[t] = {"reference_probability": round(p_ref, 4), "probability": round(p_ref + float(phi.sum()), 4), "drivers": items}
        return out

    # ------------------------------------------------------------------ counterfactuals
    def counterfactuals(self, daily: dict, profile: dict, injuries, prior, targets: list[str]) -> list[dict]:
        base_X = daily_features(daily, profile, injuries, prior)[-1]
        base = self.predict(base_X)
        results = []
        for key, label in SCENARIOS.items():
            if key == "nordic" and profile.get("nordic"):
                continue
            if key == "adductor" and profile.get("adductor"):
                continue
            d2 = {k: v.copy() for k, v in daily.items()}
            p2 = dict(profile)
            last = slice(max(0, len(d2["load"]) - 7), len(d2["load"]))
            sess = d2["session"][last] > 0
            if key.startswith("load_minus"):
                f = 0.85 if key.endswith("15") else 0.70
                for k in ("load", "duration", "distance"):
                    d2[k][last] = d2[k][last] * f
            elif key == "extra_rest_day":
                loads = np.where(sess & (d2["match"][last] == 0), d2["load"][last], -1)
                if loads.max() <= 0:
                    continue
                i = last.start + int(loads.argmax())
                for k in ("load", "duration", "distance", "session"):
                    d2[k][i] = 0.0
                for k in ("rpe", "sprint", "jump", "rhr", "asymmetry"):
                    d2[k][i] = np.nan
            elif key == "sleep_8h":
                s = np.nan_to_num(d2["sleep"][last], nan=7.0)
                if s.min() >= 8:
                    continue
                d2["sleep"][last] = np.maximum(s, 8.0)
            elif key == "skip_match":
                if d2["match"][last].sum() == 0:
                    continue
                d2["match"][last] = 0.0
            elif key in ("nordic", "adductor"):
                p2[key] = 1
            X2 = daily_features(d2, p2, injuries, prior)[-1]
            new = self.predict(X2)
            changes = []
            for t in targets:
                b, n = float(base[t][0]), float(new[t][0])
                changes.append({"target": t, "before": round(b, 4), "after": round(n, 4),
                                "relative_change": round((n - b) / b, 3) if b > 0 else 0.0})
            results.append({"scenario": key, "label": label, "changes": changes})
        return results

    # ------------------------------------------------------------------ real-data cross-check
    def real_check(self, daily: dict) -> dict | None:
        self.load()
        rm = self.m.get("real")
        if not rm:
            return None
        x = realdata.app_features_from_daily(daily["distance"], daily["session"], daily["rpe"]).reshape(1, -1)
        raw = float(rm["model"].predict_proba(x)[0, 1])
        p = float(rm["calibrator"].predict([raw])[0])
        base = realdata.PREVALENCE
        return {"daily_probability": round(p, 4), "relative_risk": round(p / base, 2), "cohort_daily_rate": round(base, 4),
                "note": "Model trained on REAL logs of 74 competitive runners (Lövdal et al. 2021, CC0). It uses running "
                        "volume, sessions, rest days and hard sessions only, and has no body-region labels. "
                        "Most relevant for running-based sports."}


engine = InjuryEngine()


# ---------------------------------------------------------------------- calendar builders
def empty_daily(n: int) -> dict[str, np.ndarray]:
    d = {k: np.full(n, np.nan) for k in DAILY_KEYS}
    for k in ("load", "duration", "distance", "match", "session"):
        d[k][:] = 0.0
    return d


def daily_from_sessions(sessions: list, start: date, end: date) -> dict[str, np.ndarray]:
    """Aggregate stored sessions into a contiguous calendar [start, end]."""
    n = (end - start).days + 1
    d = empty_daily(n)
    for s in sessions:
        i = (s.session_date - start).days
        if i < 0 or i >= n:
            continue
        load = s.duration_min * s.rpe
        prev_load = d["load"][i]
        d["load"][i] += load
        d["duration"][i] += s.duration_min
        d["distance"][i] += s.distance_km or 0.0
        d["rpe"][i] = s.rpe if np.isnan(d["rpe"][i]) else (d["rpe"][i] * prev_load + s.rpe * load) / max(prev_load + load, 1e-9)
        d["session"][i] += 1
        d["match"][i] = max(d["match"][i], 1.0 if getattr(s, "session_type", "training") == "match" else 0.0)
        d["sleep"][i] = s.sleep_hours
        d["wellness"][i] = s.wellness
        for key, attr in (("sprint", "sprint_100m_s"), ("jump", "vertical_jump_cm"), ("rhr", "resting_hr"), ("asymmetry", "asymmetry_pct")):
            v = getattr(s, attr, None)
            if v is not None:
                d[key][i] = v
    return d


def injury_events(records: list, start: date, history_text: str = "") -> tuple[list[tuple[int, str, int]], dict[str, int]]:
    prior: dict[str, int] = {}
    for r in [x.strip() for x in (history_text or "").split(",") if x.strip()]:
        if r in REGIONS:
            prior[r] = prior.get(r, 0) + 1
    events = []
    for rec in records:
        if rec.onset_date < start:
            prior[rec.region] = prior.get(rec.region, 0) + 1
        else:
            ret = rec.return_date or (rec.onset_date + timedelta(days=14))
            events.append(((rec.onset_date - start).days, rec.region, (ret - start).days))
    return events, prior


def prevention_for(region: str) -> dict:
    return {"region": region, "label": REGION_LABELS[region], **PREVENTION[region],
            "disclaimer": "General guidance only. Not a diagnosis or a substitute for a physiotherapist or doctor."}
