import React from "react";
import { Database, FolderOpen, Power, RefreshCw, X, XCircle } from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "../../api/client";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatBytes, formatEta } from "../../utils/formatters";

export const JobMonitorPanel: React.FC = () => {
  const {
    state,
    dispatch,
    refreshJobsAndCache,
    cancelWtcJob,
    openWtcJobResult,
    triggerShutdown,
  } = useAnalysis();

  if (!state.jobDrawerOpen) return null;

  return (
    <div
      className="drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          dispatch({ type: "SET_JOB_DRAWER_OPEN", open: false });
        }
      }}
    >
      <div className="drawer-panel">
        <div className="drawer-header">
          <div style={{ fontWeight: 600, fontSize: 14 }}>
            Монитор задач Tier 3 и дисковый кэш
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => void refreshJobsAndCache()}
              title="Обновить список задач и кэша"
            >
              <RefreshCw size={13} />
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => dispatch({ type: "SET_JOB_DRAWER_OPEN", open: false })}
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <div className="drawer-body">
          <section>
            <div className="sidebar-heading" style={{ marginBottom: 8 }}>
              <span>Очередь фоновых расчётов ({state.jobs.length})</span>
            </div>

            {state.jobs.length === 0 ? (
              <div className="hint-text">Фоновых задач WTC пока не запускалось.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {state.jobs.map((job) => {
                  const isActive = !TERMINAL_JOB_STATUSES.has(job.status);
                  const transport = state.jobTransport[job.job_id];
                  const pair = `${String(job.params.station_i ?? "?")}↔${String(job.params.station_j ?? "?")}`;
                  return (
                    <div key={job.job_id} className="job-card">
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        <span className="mono" style={{ fontWeight: 600, fontSize: 12 }}>
                          {pair} ({String(job.params.component ?? "X")}) ·{" "}
                          {String(job.params.episode_id ?? "")}
                        </span>
                        <span
                          className={`pill mono ${
                            job.status === "completed" || job.status === "cached"
                              ? "pill-ok"
                              : job.status === "failed" || job.status === "cancelled"
                                ? "pill-danger"
                                : "pill-accent"
                          }`}
                        >
                          {job.status}
                          {job.cache_hit ? " (hit)" : ""}
                        </span>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{ width: `${job.progress_pct}%` }}
                        />
                      </div>

                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          fontSize: 11,
                          color: "var(--text-secondary)",
                        }}
                      >
                        <span>{job.stage}</span>
                        <span className="mono">
                          {job.progress_pct.toFixed(0)}% · ETA: {formatEta(job.eta_sec)}
                          {transport ? ` · ${transport.toUpperCase()}` : ""}
                        </span>
                      </div>

                      {job.error_message && (
                        <div className="hint-text" style={{ color: "var(--danger)" }}>
                          {job.error_message}
                        </div>
                      )}

                      <div style={{ display: "flex", justifyContent: "flex-end", gap: 6 }}>
                        {isActive && (
                          <button
                            type="button"
                            className="btn btn-sm btn-danger"
                            onClick={() => void cancelWtcJob(job.job_id)}
                          >
                            <XCircle size={12} />
                            <span>Отменить</span>
                          </button>
                        )}
                        {(job.status === "completed" || job.status === "cached") && (
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            onClick={() => {
                              void openWtcJobResult(job.job_id);
                              dispatch({ type: "SET_JOB_DRAWER_OPEN", open: false });
                            }}
                          >
                            <FolderOpen size={12} />
                            <span>Открыть карту WTC</span>
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          <section>
            <div className="sidebar-heading" style={{ marginBottom: 8 }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                <Database size={12} />
                Индекс кэша runtime.sqlite ({state.cacheEntries.length})
              </span>
            </div>

            {state.cacheEntries.length === 0 ? (
              <div className="hint-text">Записей в индексе кэша пока нет.</div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Тип</th>
                      <th>Станция/Пара</th>
                      <th>Размер</th>
                      <th>Ключ SHA-256</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.cacheEntries.slice(0, 30).map((entry) => {
                      const label = entry.params.pair
                        ? (entry.params.pair as string[]).join("↔")
                        : String(entry.params.station_iaga ?? "—");
                      return (
                        <tr key={entry.cache_key}>
                          <td className="mono">{entry.category}</td>
                          <td className="mono">{label}</td>
                          <td className="mono">{formatBytes(entry.size_bytes)}</td>
                          <td className="mono">{entry.cache_key.slice(0, 10)}…</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="sidebar-section">
            <div className="sidebar-heading">
              <span>Управление сервером (режим --browser)</span>
            </div>
            <div className="hint-text">
              При работе через внешний браузер кнопка ниже корректно завершает сервер FastAPI и пул
              воркеров <span className="mono">ProcessPoolExecutor</span>.
            </div>
            <button
              type="button"
              className="btn btn-danger"
              disabled={state.serverStatus !== "running"}
              onClick={() => void triggerShutdown()}
            >
              <Power size={13} />
              <span>Остановить сервер</span>
            </button>
          </section>
        </div>
      </div>
    </div>
  );
};
