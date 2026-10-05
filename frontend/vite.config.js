import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production UI shipped by FastAPI lives in ./dist (hand-written, no build step).
// The React shell builds to ./dist-react so it never overwrites it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist-react", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
