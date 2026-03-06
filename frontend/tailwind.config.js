/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Udemy-inspired palette
        udemy: {
          purple: "#a435f0",
          "purple-dark": "#8710d8",
          "purple-light": "#c77dff",
          dark: "#1c1d1f",
          "dark-hover": "#2d2f31",
          text: "#2d2f31",
          "text-muted": "#6a6f73",
          bg: "#f7f9fa",
          card: "#ffffff",
          border: "#d1d7dc",
          success: "#1e6055",
          "success-bg": "#d1f7c4",
          warning: "#b4690e",
          "warning-bg": "#fbe5c8",
          danger: "#b32d0f",
          star: "#e59819",
        },
      },
      fontFamily: {
        sans: [
          "udemy sans",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 2px 4px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.08)",
        "card-hover": "0 4px 8px rgba(0,0,0,0.12), 0 8px 24px rgba(0,0,0,0.12)",
      },
    },
  },
  plugins: [],
};
