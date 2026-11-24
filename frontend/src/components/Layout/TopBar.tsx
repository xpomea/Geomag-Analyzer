import React, { useState } from "react";
import { Activity, Cpu, Loader2, Power, Radio } from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "../../api/client";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatUtcFull } from "../../utils/formatters";

export const TopBar: React.FC = () => {
  const { state, dispatch, health, activeEpisode, triggerShutdown } = useAnalysis();
  const [confirmShutdown, setConfirmShutdown] = useState(false);

  const runningJobs = state.jobs.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status));

  return (
    <>
      <header className="topbar">
        <div className="topbar-brand">
          <Radio size={18} color="var(--accent-primary)" />
          <div>
            <div className="topbar-title">GeoMag Analyzer</div>
            <div className="topbar-sub">Поляризация Pc5/Pc6 · INTERMAGNET · SOLAR-1 · OMNI · OULU</div>
          </div>
        </div>

        <div className="topbar-pills">
          {activeEpisode && (
            <span className="pill pill-accent mono">{activeEpisode.label}</span>
          )}
          <span className="pill mono">
            {state.stationA} ↔ {state.stationB} · {state.band} · {state.coordFrame}
          </span>
          <span className="pill mono">α = {state.alpha}</span>
          {state.cursorUtcMs !== null && (
            <span className="pill mono">UTC: {formatUtcFull(state.cursorUtcMs)}</span>
          )}
          {health ? (
            <span className="pill pill-ok">
              <Activity size={12} />
              <span>
                API {health.core_version} ({health.catalog_stations} ст.)
              </span>
            </span>
          ) : (
            <span className="pill pill-warn">Подключение…</span>
          )}
        </div>

        <div className="topbar-actions">
          <button
            type="button"
            className="btn"
            onClick={() => dispatch({ type: "SET_JOB_DRAWER_OPEN", open: !state.jobDrawerOpen })}
          >
            {runningJobs.length > 0 ? (
              <Loader2 size={14} className="spin" color="var(--accent-primary)" />
            ) : (
              <Cpu size={14} />
            )}
            <span>Задачи и кэш</span>
            <span className="pill mono" style={{ padding: "1px 6px" }}>
              {runningJobs.length > 0 ? `${runningJobs.length} акт.` : state.cacheEntries.length}
            </span>
          </button>

          <button
            type="button"
            className="btn btn-danger"
            onClick={() => setConfirmShutdown(true)}
            disabled={state.serverStatus !== "running"}
            title="Корректно остановить сервер и пул воркеров (для режима --browser)"
          >
            <Power size={14} />
            <span>Выход</span>
          </button>
        </div>
      </header>

      {confirmShutdown && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <div style={{ fontWeight: 600, fontSize: 15 }}>Остановить сервер GeoMag Analyzer?</div>
            <div className="hint-text">
              Отправит команду <span className="mono">POST /api/shutdown</span> для корректного
              завершения воркеров <span className="mono">ProcessPoolExecutor</span> и закрытия локального
              сервера.
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button type="button" className="btn" onClick={() => setConfirmShutdown(false)}>
                Отмена
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => {
                  setConfirmShutdown(false);
                  void triggerShutdown();
                }}
              >
                Остановить сервер
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
