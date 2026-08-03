import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Attend",
  description: "Classroom attendance, taken from a video pan.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-900">{children}</body>
    </html>
  );
}
