export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Sora", "ui-sans-serif", "system-ui"],
        display: ["Syne", "Sora", "sans-serif"],
      },
      colors: {
        ink: "#061018",
        panel: "#0c1a24",
        line: "#1c3344",
        lime: "#c8f542",
        mint: "#7dffc3",
        ember: "#ffb25a",
      },
      boxShadow: {
        glow: "0 0 80px rgba(200,245,66,0.12)",
      },
    },
  },
  plugins: [],
};
