# Athlete Lens

**Machine-learning framework for athlete performance analysis** — built for school, college and grassroots coaches in India who do **not** have GPS vests, force plates or a sports-science lab.

> A stopwatch, a tape, and how the session felt is enough.

**Live demo:** _LIVE_URL_ · API docs at `/docs`

![stack](https://img.shields.io/badge/FastAPI-0.115-009688) ![ml](https://img.shields.io/badge/scikit--learn-1.6-F7931E) ![deploy](https://img.shields.io/badge/deploy-Vercel%20%7C%20Docker-000) ![ci](https://github.com/sivaangayarkanni/athlete-lens/actions/workflows/ci.yml/badge.svg)

---

## Why this exists

District and college athletes in Tamil Nadu (and across India) already train hard. What they lack is a second pair of eyes that can turn a week of sessions into:

- a **performance index**
- a **readiness score** (green / amber / red)
- an **injury-risk band** (low / moderate / high) with a probability
- a coach-facing **plan for the next 72 hours**

Athlete Lens is that layer. It is intentionally *not* a computer-vision lab demo. It is a product a PE teacher can use after evening practice.

## What the product does

| Screen | What you get |
| --- | --- |
| **Dashboard** | Squad KPIs, injury-risk band split, a watchlist sorted by risk, recent sessions, sport mix |
| **Roster** | Search / filter by sport and risk band, readiness ring per athlete, add athletes |
| **Athlete page** | Latest analysis, 72-hour plan, readiness / performance / injury-probability trend, sRPE training load with acute (7-day) vs chronic (28-day) load, acute:chronic workload ratio with the 0.8–1.3 band, field-test trends, session log (log / delete sessions) |
| **Analysis lab** | What-if sliders (sleep, RPE, volume, rest days…) starting from any athlete's latest session — nothing is saved |
| **CSV import** | Bulk import from a hostel register / Excel export with per-row validation errors |
| **Model card** | Held-out metrics vs. naive baselines, feature importance, bands, limitations |

Sports: Athletics, Kabaddi, Kho-Kho, Football, Hockey, Volleyball, Badminton, Wrestling. Seed data includes eight Tamil Nadu athletes with four weeks of sessions so the dashboard is never empty.

## Architecture

```
Static UI (frontend/dist, vanilla JS + SVG charts)  —JSON—▶  FastAPI (backend/app/main.py)
Optional React + Recharts shell (frontend/src)                 ├─ SQLAlchemy + SQLite
                                                               ├─ RandomForest      → injury risk
                                                               └─ GradientBoosting  → performance index
```

- Models are trained on a physiologically-plausible **synthetic** cohort of 2,400 athletes (80/20 stratified split, seed 42) and shipped **pre-trained** in `backend/app/artifacts` (~0.7 MB). If the artifacts are missing or were built with a different scikit-learn version, the API retrains on first use (~3 s) into `MODEL_DIR`.
- Runtime needs only numpy + scikit-learn (no pandas) to keep the serverless bundle small.
- Retrain: `python -m backend.app.train`

### Model metrics (held-out 20 %, from `python -m backend.app.train`)

| Model | Metric | Value | Naive baseline |
| --- | --- | --- | --- |
| Injury risk (RandomForest) | ROC AUC | 0.661 | 0.5 |
| | Accuracy | 0.669 | 0.688 (always "no injury") |
| | Balanced accuracy / recall | 0.608 / 0.447 | — |
| Performance index (GradientBoosting) | MAE | 3.40 | 9.24 (predict the mean) |
| | R² | 0.859 | 0 |

Read these honestly: the labels come from a hand-written simulator, so the numbers measure how well the models recover that simulator, not real-world accuracy. The injury classifier is only modestly better than chance (and the class-balanced model trades raw accuracy for recall) — treat the band as a prompt to talk, not a prediction. The model card screen (`/api/model`) always shows the numbers of the artifacts actually loaded.

**This is not a medical device.** High risk means “talk to the physio / rest the block”, not a diagnosis.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Open `http://localhost:8000` (UI) or `http://localhost:8000/docs` (OpenAPI). The SQLite DB defaults to `/tmp/athlete_lens.db`; set `DATABASE_URL` to change it (see `.env.example`).

Optional React shell (EN / தமிழ் toggle):

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173, proxies /api
```

## Deploy

### Vercel (serverless)

The repo deploys as-is with Vercel's zero-config FastAPI support: `index.py` exports the ASGI `app`, `vercel.json` trims the bundle, `.python-version` pins Python 3.12.

```bash
npm i -g vercel
vercel deploy --prod
```

On Vercel only `/tmp` is writable, so the SQLite DB lives in `/tmp` and the pre-trained models are loaded read-only from the bundle. **Data written on Vercel is ephemeral**: each function instance starts from the seeded demo roster on a cold start, and separate instances don't share writes. For persistent data point `DATABASE_URL` at a hosted Postgres (add a driver such as `psycopg[binary]` to `requirements.txt`).

### Docker (persistent SQLite volume)

```bash
docker compose up --build    # http://localhost:8000
```

## Tests

```bash
pytest -q
```

24 tests cover the API (CRUD, sessions, validation, what-if, CSV ingest, stats, model card, static UI) and the ML engine (feature building, training, artifact staleness). CI runs them on Python 3.12 and 3.13, boots the server for a smoke test, and builds the React shell.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/api/health` | liveness + model meta |
| GET | `/api/stats` | squad KPIs, risk bands, watchlist, recent sessions |
| GET / POST | `/api/athletes` | list (`?sport=&risk=&q=`) / create |
| GET / PATCH / DELETE | `/api/athletes/{id}` | detail with timeline + latest analysis / update (rescores history) / delete |
| GET | `/api/athletes/{id}/sessions` | session list |
| POST | `/api/sessions` | log + score a session |
| DELETE | `/api/sessions/{id}` | delete a session |
| POST | `/api/analyze` | guest or what-if analysis (`athlete_id` uses that athlete's latest session as the baseline) |
| POST | `/api/upload-csv` | bulk import (multipart `file`) |
| GET | `/api/model` | model card |
| GET | `/api/sports` | supported sports |

Validation errors return `422` with a readable `detail` (`"age: Input should be greater than or equal to 10"`) plus a per-field `errors` list.

Sample file: [`data/sample_athletes.csv`](data/sample_athletes.csv)

## Product principles

1. **Works on a dust track.** No wearable lock-in.
2. **Explainable.** Every score ships with “why” and a next action.
3. **Bilingual-ready.** EN / TA toggle on the React shell.
4. **Honest ML.** Model card is a first-class screen, not a footnote.

## Author

Sivaangayarkanni Sathyan · B.Tech AI & Data Science
[github.com/sivaangayarkanni](https://github.com/sivaangayarkanni)

MIT licensed.
