import React, { useState } from "react";
import {
  Cpu,
  Loader2,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Power,
  Radio,
  Sun,
} from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "../../api/client";
import { useAnalysis } from "../../store/AnalysisContext";

export interface TopBarProps {
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  sidebarOpen = true,
  onToggleSidebar,
}) => {
  const { state, dispatch, health, triggerShutdown } = useAnalysis();
  const [confirmShutdown, setConfirmShutdown] = useState(false);

  const runningJobs = state.jobs.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status));
  const isLightExport = state.exportTheme === "light";

  return (
    <>
      <header className="topbar">
        <div className="topbar-brand">
          {onToggleSidebar && (
            <button
              type="button"
              className="btn btn-sm btn-icon"
              onClick={onToggleSidebar}
              title={sidebarOpen ? "Collapse control sidebar" : "Expand control sidebar"}
              aria-label={sidebarOpen ? "Collapse control sidebar" : "Expand control sidebar"}
            >
              {sidebarOpen ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />}
            </button>
          )}
          <Radio size={17} color="var(--accent-primary)" />
          <div className="topbar-title-row">
            <span className="topbar-title">GeoMag Analyzer</span>
            <span className="topbar-sub">
              Pc5/Pc6 Polarization · INTERMAGNET · SOLAR-1 · OMNI · OULU
            </span>
          </div>
        </div>

        <div className="topbar-actions">
          {!health && <span className="pill pill-warn">Connecting…</span>}

          <button
            type="button"
            className="btn btn-sm"
            onClick={() =>
              dispatch({
                type: "SET_EXPORT_THEME",
                theme: isLightExport ? "dark" : "light",
              })
            }
            title="Global figure export theme for SVG / PNG 300 DPI"
          >
            {isLightExport ? <Sun size={13} color="#f59e0b" /> : <Moon size={13} />}
            <span>Export: {isLightExport ? "Print Light" : "Screen Dark"}</span>
          </button>

          <button
            type="button"
            className="btn btn-sm"
            onClick={() => dispatch({ type: "SET_JOB_DRAWER_OPEN", open: !state.jobDrawerOpen })}
          >
            {runningJobs.length > 0 ? (
              <Loader2 size={13} className="spin" color="var(--accent-primary)" />
            ) : (
              <Cpu size={13} />
            )}
            <span>Jobs &amp; Cache</span>
            <span className="pill mono" style={{ padding: "1px 6px" }}>
              {runningJobs.length > 0 ? `${runningJobs.length} active` : state.cacheEntries.length}
            </span>
          </button>

          <button
            type="button"
            className="btn btn-sm btn-danger"
            onClick={() => setConfirmShutdown(true)}
            disabled={state.serverStatus !== "running" || health?.can_shutdown === false}
            title={
              health?.can_shutdown === false
                ? "Available in desktop mode"
                : "Close GeoMag Analyzer"
            }
          >
            <Power size={13} />
            <span>Exit</span>
          </button>
        </div>
      </header>

      {confirmShutdown && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <div style={{ fontWeight: 600, fontSize: 15 }}>Exit GeoMag Analyzer?</div>
            <div className="hint-text">
              Active computations will be stopped and the application will close.
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button type="button" className="btn" onClick={() => setConfirmShutdown(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => {
                  setConfirmShutdown(false);
                  void triggerShutdown();
                }}
              >
                Exit
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
