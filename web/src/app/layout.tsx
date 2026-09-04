import type { Metadata } from "next";
import { Inter, EB_Garamond, Geist_Mono, Archivo_Black } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { AiChatPanel } from "@/components/AiChatPanel";
import { AiContentShift } from "@/components/AiContentShift";
import { AiAssistantProvider } from "@/lib/aiAssistantContext";

// Inter carries body/nav/buttons/captions; EB Garamond at weight 300 is the
// open-source substitute for ElevenLabs' licensed Waldenburg Light display
// serif (see DESIGN.md "Note on Font Substitutes") — used for headlines only.
const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const displaySerif = EB_Garamond({
  variable: "--font-display",
  weight: "variable",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Heavy condensed sans for the FAUXGPU wordmark only — matches the logo's
// blocky lockup, distinct from the EB Garamond used for editorial headings.
const wordmark = Archivo_Black({
  variable: "--font-wordmark",
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FauxGPU",
  description: "Learn GPU VRAM, KV cache, and cluster topology without needing real hardware.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${displaySerif.variable} ${geistMono.variable} ${wordmark.variable} h-full antialiased`}
    >
      <head>
        {/* Runs before paint so an explicit saved theme (or system preference,
            already handled by the CSS media query) never flashes light-then-dark. */}
        <Script
          id="theme-init"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `try {
              var t = localStorage.getItem("theme");
              if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
            } catch (e) {}`,
          }}
        />
      </head>
      <body className="min-h-full flex flex-col">
        <AiAssistantProvider>
          <AiContentShift>
            {children}
            <footer className="border-t border-hairline">
              <div className="max-w-7xl mx-auto px-6 md:px-10 py-6 flex flex-wrap items-center justify-between gap-3 text-xs text-muted">
                <span>
                  © {new Date().getFullYear()}{" "}
                  <a
                    href="https://github.com/devops-dojo7"
                    target="_blank"
                    rel="noreferrer"
                    className="text-body-strong hover:text-ink transition-colors"
                  >
                    Devops-Dojo
                  </a>{" "}
                  · Open source under the{" "}
                  <a
                    href="https://github.com/devops-dojo7/FauxGPU/blob/main/LICENSE"
                    target="_blank"
                    rel="noreferrer"
                    className="text-body-strong hover:text-ink transition-colors"
                  >
                    MIT License
                  </a>
                </span>
                <a
                  href="https://github.com/devops-dojo7/FauxGPU"
                  target="_blank"
                  rel="noreferrer"
                  className="text-body-strong hover:text-ink transition-colors"
                >
                  GitHub
                </a>
              </div>
            </footer>
          </AiContentShift>
          <AiChatPanel />
        </AiAssistantProvider>
      </body>
    </html>
  );
}
