import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { DashboardPage } from "@/features/dashboard/DashboardPage";
import { RoadmapPage } from "@/features/roadmap/RoadmapPage";

/**
 * Two views, because there are two questions.
 *
 * "What should I do next" is answered on the dashboard, in one sentence with its
 * arithmetic underneath. "Why in that order" is answered on the canvas, by the graph.
 * Collapsing them into one screen would bury the first inside the second.
 */
export function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen flex-col bg-ground">
        <Header />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/path" element={<RoadmapPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

function Header() {
  return (
    <header className="flex shrink-0 items-center justify-between border-b border-line bg-surface px-5 py-3">
      <div className="flex items-baseline gap-3">
        <span className="font-display text-base font-semibold tracking-tight text-ink">
          Pathwise
        </span>
        <span className="hidden font-mono text-micro text-faint sm:inline">
          adaptive learning engine
        </span>
      </div>

      <nav className="flex items-center gap-1" aria-label="Main">
        <Tab to="/" label="Today" />
        <Tab to="/path" label="Path" />
      </nav>
    </header>
  );
}

function Tab({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "rounded-node px-3 py-1.5 font-mono text-micro uppercase tracking-wider transition-colors duration-150",
          isActive ? "bg-raised text-ink" : "text-muted hover:text-ink",
        ].join(" ")
      }
    >
      {label}
    </NavLink>
  );
}
