import React, { useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "./api/client";
import { SpectrogramPanel } from "./components/CrossSpectralPanel/SpectrogramPanel";
import { WelchCrossPanel } from "./components/CrossSpectralPanel/WelchCrossPanel";
import { WtcPanel } from "./components/CrossSpectralPanel/WtcPanel";
import { JobMonitorPanel } from "./components/JobMonitor/JobMonitorPanel";
import { AuthGate } from "./components/Layout/AuthGate";
import { ControlSidebar } from "./components/Layout/ControlSidebar";
import { TimeScrubberBar } from "./components/Layout/TimeScrubberBar";
import { TopBar } from "./components/Layout/TopBar";
import { StationMapPanel } from "./components/StationMap/StationMapPanel";
import { MagnetogramPanel } from "./components/SyncedCharts/MagnetogramPanel";
import { SpaceWeatherPanel } from "./components/SyncedCharts/SpaceWeatherPanel";
import { StokesPanel } from "./components/SyncedCharts/StokesPanel";
import { AnalysisProvider, useAnalysis } from "./store/AnalysisContext";
import type { DeckTab } from "./store/analysisReducer";

const DEFAULT_MAP_HEIGHT = 350;
const MIN_MAP_HEIGHT = 180;
const MIN_DECK_HEIGHT = 240;

interface DeckTabDef {
  id: DeckTab;
  label: string;
  sub: string;
  group: "station" | "cross";
}

const DECK_TABS: DeckTabDef[] = [
  { id: "space_weather", label: "Space Weather", sub: "SOLAR-1 · OMNI · OULU", group: "station" },
  { id: "magnetogram", label: "Magnetogram", sub: "Δ(X,Y,Z) & Bandpass", group: "station" },
  { id: "stokes", label: "Stokes 2D/3D", sub: "p, ψ, χ, ε, γ², 3D", group: "station" },
  { id: "spectrogram", label: "Spectrogram", sub: "Welch 2D Power", group: "station" },
  { id: "welch", label: "Cross-Welch", sub: "γ², Δφ, m & FLR", group: "cross" },
  { id: "wtc", label: "XWT / WTC", sub: "Wavelet & Monte Carlo", group: "cross" },
];

const Workbench: React.FC = () => {
  const { state, dispatch, geomag, globalError, baselineWarning } = useAnalysis();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mapCollapsed, setMapCollapsed] = useState(false);
  const [mapDockHeight, setMapDockHeight] = useState(DEFAULT_MAP_HEIGHT);
  const [draggingSplit, setDraggingSplit] = useState(false);
  const splitStageRef = useRef<HTMLDivElement | null>(null);

  const runningJobsCount = state.jobs.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status)).length;

  const handleSplitterPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (mapCollapsed) return;
    e.preventDefault();
    const startY = e.clientY;
    const startH = mapDockHeight;
    setDraggingSplit(true);

    const onMove = (ev: PointerEvent) => {
      const stageH = splitStageRef.current?.clientHeight ?? 900;
      const maxH = Math.max(MIN_MAP_HEIGHT + 80, stageH - MIN_DECK_HEIGHT - 10);
      const nextH = Math.min(maxH, Math.max(MIN_MAP_HEIGHT, startH + (ev.clientY - startY)));
      setMapDockHeight(Math.round(nextH));
    };

    const onUp = () => {
      setDraggingSplit(false);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  if (state.serverStatus === "stopped" || state.serverStatus === "shutting_down") {
    return (
      <div className="modal-backdrop">
        <div className="modal-card">
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 15 }}>
            <CheckCircle2 size={18} color="var(--ok)" />
            <span>GeoMag Analyzer Server Stopped</span>
          </div>
          <div className="hint-text">
            <span className="mono">ProcessPoolExecutor</span> workers and FastAPI exited gracefully.
            You may now close this window.
          </div>
        </div>
      </div>
    );
  }

  const hasAlerts =
    Boolean(globalError) ||
    Boolean(baselineWarning) ||
    Boolean(geomag && geomag.warnings.length > 0 && !baselineWarning);

  return (
    <div className="app-shell">
      <TopBar sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen((v) => !v)} />
      <div className="main-layout">
        {sidebarOpen && <ControlSidebar />}
        <main className="workspace">
          <TimeScrubberBar />

          {hasAlerts && (
            <div className="workspace-alerts">
              {globalError && (
                <div className="banner banner-danger">
                  <AlertTriangle size={15} />
                  <span>{globalError}</span>
                </div>
              )}

              {baselineWarning && (
                <div className="banner banner-warn">
                  <AlertTriangle size={15} />
                  <span>{baselineWarning}</span>
                </div>
              )}

              {geomag && geomag.warnings.length > 0 && !baselineWarning && (
                <div className="banner banner-info">
                  <AlertTriangle size={14} />
                  <span>{geomag.warnings.join(" · ")}</span>
                </div>
              )}
            </div>
          )}

          <div ref={splitStageRef} className="split-stage">
            <div
              className="map-dock"
              style={mapCollapsed ? undefined : { height: mapDockHeight }}
            >
              <StationMapPanel
                fillHeight
                collapsed={mapCollapsed}
                onToggleCollapse={() => setMapCollapsed((v) => !v)}
              />
            </div>

            {!mapCollapsed && (
              <div
                className={`split-resizer ${draggingSplit ? "dragging" : ""}`}
                onPointerDown={handleSplitterPointerDown}
                onDoubleClick={() => setMapDockHeight(DEFAULT_MAP_HEIGHT)}
                role="separator"
                aria-orientation="horizontal"
                aria-label="Resize map and analytics panels"
                title="Drag to resize map vs. charts (double-click to reset)"
              >
                <div className="split-resizer-grip" />
              </div>
            )}

            <div className="analytics-deck" style={mapCollapsed ? { marginTop: 8 } : undefined}>
              <div className="deck-content">
                {state.activeDeckTab === "space_weather" && <SpaceWeatherPanel />}
                {state.activeDeckTab === "magnetogram" && <MagnetogramPanel />}
                {state.activeDeckTab === "stokes" && <StokesPanel />}
                {state.activeDeckTab === "spectrogram" && <SpectrogramPanel />}
                {state.activeDeckTab === "welch" && <WelchCrossPanel />}
                {state.activeDeckTab === "wtc" && <WtcPanel />}
              </div>

              <nav className="deck-tab-rail" aria-label="Analysis views">
                <div className="deck-rail-heading">Station ({state.stationA})</div>
                {DECK_TABS.filter((t) => t.group === "station").map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={`deck-tab-btn ${state.activeDeckTab === tab.id ? "active" : ""}`}
                    onClick={() => dispatch({ type: "SET_ACTIVE_DECK_TAB", tab: tab.id })}
                  >
                    <div className="deck-tab-title-row">
                      <span>{tab.label}</span>
                    </div>
                    <span className="deck-tab-sub">{tab.sub}</span>
                  </button>
                ))}

                <div className="deck-rail-divider" />
                <div className="deck-rail-heading">
                  Pair ({state.stationA}↔{state.stationB})
                </div>
                {DECK_TABS.filter((t) => t.group === "cross").map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={`deck-tab-btn ${state.activeDeckTab === tab.id ? "active" : ""}`}
                    onClick={() => dispatch({ type: "SET_ACTIVE_DECK_TAB", tab: tab.id })}
                  >
                    <div className="deck-tab-title-row">
                      <span>{tab.label}</span>
                      {tab.id === "wtc" && runningJobsCount > 0 && (
                        <Loader2 size={12} className="spin" color="var(--accent-primary)" />
                      )}
                    </div>
                    <span className="deck-tab-sub">{tab.sub}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </main>
      </div>
      <JobMonitorPanel />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <AnalysisProvider>
      <AuthGate>
        <Workbench />
      </AuthGate>
    </AnalysisProvider>
  );
};

export default App;
