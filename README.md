# Athlete Lens

**Machine-learning framework for athlete performance analysis** — built for school, college and grassroots coaches in India who do **not** have GPS vests, force plates or a sports-science lab.

> A stopwatch, a tape, and how the session felt is enough.

Live app after boot: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

![stack](https://img.shields.io/badge/FastAPI-0.115-009688) ![ml](https://img.shields.io/badge/scikit--learn-1.6-F7931E) ![ui](https://img.shields.io/badge/UI-production-C8F542)

---

## Why this exists

District and college athletes in Tamil Nadu (and across India) already train hard. What they lack is a second pair of eyes that can turn a week of sessions into:

- a **performance index**
- a **readiness score**
- an **injury-risk band** (low / moderate / high)
- a coach-facing plan for the next 72 hours

Athlete Lens is that layer. It is intentionally *not* a computer-vision lab demo. It is a product a PE teacher can use after evening practice.

## What the product does

| Module | What you get |
| --- | --- |
| Roster | Athletes across Athletics, Kabaddi, Kho-Kho, Football, Hockey, Volleyball, Badminton, Wrestling |
| Session log | Duration, distance, 100m, vertical jump, HR, RPE, sleep, wellness, 7-day load |
| Analysis lab | What-if sliders without touching the roster |
| CSV ingest | Bulk import from a hostel register / Excel sheet |
| Model card | Honest limits, features, train metrics |

Seed data includes eight Tamil Nadu athletes so the dashboard is never empty on first run.

## Architecture

```
Static production UI  —JSON—▶  FastAPI
  (or React+Vite)                 |
                                  ├─ SQLAlchemy + SQLite
                                  ├─ RandomForest  → injury risk
                                  └─ GradientBoosting → performance index
```

Models train on a physiologically-plausible synthetic cohort (2,400 athletes) the first time the API starts, then persist as `joblib` artifacts. Swap the trainer for club data when you have 8–12 weeks of real sessions.

**This is not a medical device.** High risk means “talk to the physio / rest the block”, not a diagnosis.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python -m uvicorn backend.app.main:app --reload --port 8000
```

The production UI is bundled at `frontend/dist` and is served by FastAPI on the same port.

Open `http://localhost:8000`.

Optional React + Recharts source (hot reload):

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite proxy forwards `/api` to the backend.

## Production (one container)

```bash
docker compose up --build
```

Serves the built UI and API on port `8000`.

## Tests

```bash
export PYTHONPATH=.
pytest -q
```

## API surface

- `GET /api/health` — liveness + model meta
- `GET /api/stats` — squad KPIs
- `GET/POST /api/athletes`
- `GET /api/athletes/{id}` — timeline
- `POST /api/sessions` — log + score
- `POST /api/analyze` — guest / what-if
- `POST /api/upload-csv` — bulk
- `GET /api/model` — model card

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
