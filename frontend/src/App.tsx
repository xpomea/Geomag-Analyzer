import React from "react";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { CrossSpectralContainer } from "./components/CrossSpectralPanel/CrossSpectralContainer";
import { JobMonitorPanel } from "./components/JobMonitor/JobMonitorPanel";
import { AuthGate } from "./components/Layout/AuthGate";
import { ControlSidebar } from "./components/Layout/ControlSidebar";
import { TimeScrubberBar } from "./components/Layout/TimeScrubberBar";
import { TopBar } from "./components/Layout/TopBar";
import { MagnetogramPanel } from "./components/SyncedCharts/MagnetogramPanel";
import { SpaceWeatherPanel } from "./components/SyncedCharts/SpaceWeatherPanel";
import { StokesPanel } from "./components/SyncedCharts/StokesPanel";
import { AnalysisProvider, useAnalysis } from "./store/AnalysisContext";

const Workbench: React.FC = () => {
  const { state, geomag, globalError, baselineWarning, shiftedSolar1 } = useAnalysis();

  if (state.serverStatus === "stopped" || state.serverStatus === "shutting_down") {
    return (
      <div className="modal-backdrop">
        <div className="modal-card">
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 15 }}>
            <CheckCircle2 size={18} color="var(--ok)" />
            <span>Сервер GeoMag Analyzer остановлен</span>
          </div>
          <div className="hint-text">
            Воркеры <span className="mono">ProcessPoolExecutor</span> и FastAPI корректно завершили
            работу. Эту вкладку браузера можно закрыть.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <TopBar />
      <div className="main-layout">
        <ControlSidebar />
        <main className="workspace">
          <TimeScrubberBar />
          <div className="panels-scroll">
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

            {shiftedSolar1?.solar1.warning && (
              <div className="hint-text" style={{ padding: "0 4px" }}>
                Примечание SOLAR-1: {shiftedSolar1.solar1.warning}
              </div>
            )}

            <SpaceWeatherPanel />
            <MagnetogramPanel />
            <StokesPanel />
            <CrossSpectralContainer />
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
