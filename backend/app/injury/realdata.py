"""REAL data: Lövdal, den Hartigh & Azzopardi (2021) competitive-runner training logs.

Dataset : "Replication Data for: Injury Prediction In Competitive Runners With Machine Learning",
          DataverseNL, doi:10.34894/UWU9PV, licence CC0 1.0 (verified via DataCite metadata).
          74 Dutch middle/long-distance runners, 2012-2019. The "week approach" file has one row
          per athlete-day with three preceding weeks of aggregated training and a binary `injury`
          label (42,798 rows, 575 positives, 1.34%). No body-region labels.
File    : week_approach_maskedID_timeseries.csv (sha256 373b01b1...b75), fetched from a
          GitHub mirror because the DataverseNL API was unreachable from our build box.

Column suffixes: "" = oldest week, ".1" = middle week, ".2" = most recent week (verified:
a row's "" week reappears as ".1" seven days later).

We train only on load features the app can also compute from its own session log:
sessions, rest days, total km, max km in a day and number of hard sessions per week, plus
week-over-week km ratios. Perceived-exertion/recovery columns are excluded from the served
model because their scaling is athlete-normalised and not reproducible from app data; they
are used only in a benchmark model.
"""
from __future__ import annotations

import csv
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "data" / "external" / "lovdal_week_approach.csv"
MIRROR = "https://raw.githubusercontent.com/sonicjoy/Injury-Prediction-for-Competitive-Runners/main/week_approach_maskedID_timeseries.csv"
SHA256 = "373b01b1a83e3da6e55dbc73f8e8d3a1febcf76e6d0c8e0d51a00904e358b75d"
PREVALENCE = 575 / 42798

BASE = ["nr. sessions", "nr. rest days", "total kms", "max km one day", "nr. tough sessions (effort in Z5, T1 or T2)"]
SHORT = ["sessions", "rest_days", "km", "max_km_day", "hard_sessions"]
WEEKS = [("", "w3"), (".1", "w2"), (".2", "w1")]          # w1 = most recent week
APP_FEATURES = [f"{s}_{w}" for _, w in WEEKS for s in SHORT] + ["km_ratio_w1_w2", "km_ratio_w1_chronic", "km_3wk"]


def fetch(path: Path = LOCAL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"checksum mismatch for {path}: {digest}")
    return path


def _ratios(w1, w2, w3):
    with np.errstate(divide="ignore", invalid="ignore"):
        r12 = np.where(w2 > 1, np.clip(w1 / np.maximum(w2, 1e-9), 0, 5), np.nan)
        chronic = (w1 + w2 + w3) / 3
        rc = np.where(chronic > 1, np.clip(w1 / np.maximum(chronic, 1e-9), 0, 5), np.nan)
    return r12, rc, w1 + w2 + w3


def load(path: Path | None = None):
    """Return X_app, X_full, y, groups (athlete), date."""
    path = fetch(path or LOCAL)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader]
    A = np.asarray(rows)
    col = {h: i for i, h in enumerate(header)}
    feats = []
    for suf, _w in WEEKS:
        for b in BASE:
            feats.append(A[:, col[b + suf]])
    Xw = np.column_stack(feats)
    km = {w: A[:, col["total kms" + suf]] for suf, w in WEEKS}
    r12, rc, k3 = _ratios(km["w1"], km["w2"], km["w3"])
    X_app = np.column_stack([Xw, r12, rc, k3])
    full_cols = [i for i, h in enumerate(header) if h not in ("Athlete ID", "injury", "Date") and not h.startswith("rel total")]
    X_full = np.column_stack([A[:, full_cols], r12, rc, k3])
    return X_app, X_full, A[:, col["injury"]].astype(int), A[:, col["Athlete ID"]].astype(int), A[:, col["Date"]]


def app_features_from_daily(distance: np.ndarray, session: np.ndarray, rpe: np.ndarray) -> np.ndarray:
    """Map an app calendar (last day = as-of day) onto the real-data feature layout.

    Approximation: 'hard session' = session with RPE >= 8 (the dataset counts Z5/T1/T2 efforts).
    """
    n = len(distance)
    pad = max(0, 21 - n)
    dist = np.concatenate([np.zeros(pad), np.nan_to_num(distance)])[-21:]
    sess = np.concatenate([np.zeros(pad), np.nan_to_num(session)])[-21:]
    hard = np.concatenate([np.zeros(pad), ((np.nan_to_num(rpe) >= 8) & (np.nan_to_num(session) > 0)).astype(float)])[-21:]
    out, kms = [], {}
    for k, w in enumerate(["w3", "w2", "w1"]):
        sl = slice(k * 7, (k + 1) * 7)
        kms[w] = dist[sl].sum()
        out += [sess[sl].sum(), 7 - (sess[sl] > 0).sum(), dist[sl].sum(), dist[sl].max(), hard[sl].sum()]
    r12, rc, k3 = _ratios(np.array([kms["w1"]]), np.array([kms["w2"]]), np.array([kms["w3"]]))
    return np.array(out + [r12[0], rc[0], k3[0]], dtype=float)


if __name__ == "__main__":   # python -m backend.app.injury.realdata  -> download + verify the CSV
    p = fetch()
    X, Xf, y, g, _ = load()
    print(f"{p}: {len(y):,} athlete-days, {len(set(g.tolist()))} athletes, {int(y.sum())} injury days")
