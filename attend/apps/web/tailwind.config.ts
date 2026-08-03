import type { Config } from "tailwindcss";

// Visual system per the UI contract (Section 5 of the roadmap): a neutral
// gray scale for ~95% of the interface, exactly three semantic colours used
// only for meaning (green/amber/red), no brand colour, no gradients.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        confirmed: "#16a34a", // green -- confirmed present
        review: "#d97706", // amber -- needs teacher review
        absent: "#dc2626", // red -- marked absent
      },
      fontSize: {
        body: "14px",
        emphasis: "16px",
        title: "20px",
      },
      borderRadius: {
        DEFAULT: "8px",
      },
    },
  },
  plugins: [],
};

export default config;
