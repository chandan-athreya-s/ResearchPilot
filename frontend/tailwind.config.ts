import type { Config } from "tailwindcss";

export default {
  darkMode: 'class',
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          950: "#06070a",
          900: "#0f172a",
          800: "#111827",
          700: "#131d2e",
          600: "#1f2937",
        },
        accent: {
          500: "#2563eb",
          600: "#1d4ed8",
          400: "#3b82f6",
        },
      },
      boxShadow: {
        soft: "0 20px 70px rgba(15, 23, 42, 0.35)",
      },
    },
  },
  plugins: [],
} satisfies Config;
