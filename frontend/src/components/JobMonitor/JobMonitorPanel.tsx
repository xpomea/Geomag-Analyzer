import React from "react";
import { Database, FolderOpen, RefreshCw, X, XCircle } from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "../../api/client";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatBytes, formatEta, formatJobStatus } from "../../utils/formatters";

export const JobMonitorPanel: React.FC = () => {
  const {
    state,
    dispatch,
    refreshJobsAndCache,
    cancelWtcJob,
    openWtcJobResult,
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
            Wavelet Jobs &amp; Cache
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => void refreshJobsAndCache()}
              title="Refresh jobs and cache"
            >
              <RefreshCw size={13} />
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => dispatch({ type: "SET_JOB_DRAWER_OPEN", open: false })}
              aria-label="Close job monitor"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        <div className="drawer-body">
          <section>
            <div className="sidebar-heading" style={{ marginBottom: 8 }}>
              <span>Background Job Queue ({state.jobs.length})</span>
            </div>

            {state.jobs.length === 0 ? (
              <div className="hint-text">No WTC jobs submitted yet.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {state.jobs.map((job) => {
                  const isActive = !TERMINAL_JOB_STATUSES.has(job.status);
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
                          {formatJobStatus(job.status)}
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
                            <span>Cancel</span>
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
                            <span>Open WTC Map</span>
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
                Cached Results ({state.cacheEntries.length})
              </span>
            </div>

            {state.cacheEntries.length === 0 ? (
              <div className="hint-text">No cached results yet.</div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Type</th>
                      <th>Station / Pair</th>
                      <th>Size</th>
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
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
