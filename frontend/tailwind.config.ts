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
          500: "#7c3aed",
          600: "#6d28d9",
          400: "#8b5cf6",
        },
      },
      boxShadow: {
        soft: "0 20px 70px rgba(15, 23, 42, 0.35)",
      },
    },
  },
  plugins: [],
} satisfies Config;
