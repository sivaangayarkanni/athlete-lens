const API = import.meta.env.VITE_API_URL || "";

async function req(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => req("/api/health"),
  stats: () => req("/api/stats"),
  athletes: () => req("/api/athletes"),
  athlete: (id) => req(`/api/athletes/${id}`),
  createAthlete: (body) => req("/api/athletes", { method: "POST", body: JSON.stringify(body) }),
  deleteAthlete: (id) => req(`/api/athletes/${id}`, { method: "DELETE" }),
  createSession: (body) => req("/api/sessions", { method: "POST", body: JSON.stringify(body) }),
  analyze: (body) => req("/api/analyze", { method: "POST", body: JSON.stringify(body) }),
  model: () => req("/api/model"),
  upload: async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API}/api/upload-csv`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};
