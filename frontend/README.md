The production UI is the static app in `frontend/dist` (`index.html`, `app.js`, `styles.css` — no build step), served by FastAPI.

`frontend/src` is an optional React + Recharts shell with an EN / தமிழ் toggle:

```bash
npm install
npm run dev     # http://localhost:5173, proxies /api to http://127.0.0.1:8000
npm run build   # outputs to frontend/dist-react (never overwrites dist/)
```
