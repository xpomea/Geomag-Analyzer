import React, { useCallback, useMemo, useState } from "react";
import { Play, RadioReceiver, Sparkles, XCircle } from "lucide-react";
import { TERMINAL_JOB_STATUSES } from "../../api/client";
import type { GapPolicy } from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import {
  buildExportFilename,
  buildHeatmapPublicationSvg,
  formatCsvWithMetadata,
} from "../../utils/exportUtils";
import { formatEta, formatJobStatus } from "../../utils/formatters";
import { ExportToolbar } from "../Common/ExportToolbar";
import { DraftNumberInput } from "../Layout/DraftNumberInput";
import { ScientificHeatmap2D } from "../SyncedCharts/ScientificHeatmap2D";

export const WtcPanel: React.FC = () => {
  const {
    state,
    dispatch,
    wtcResult,
    episodeBoundsMs,
    loadingWtcResult,
    wtcError,
    launchWtcJob,
    cancelWtcJob,
    openWtcJobResult,
  } = useAnalysis();
  const { setCursorUtcMs } = useCursor();

  const [periodMinSec, setPeriodMinSec] = useState(120);
  const [periodMaxSec, setPeriodMaxSec] = useState(3600);
  const [dj, setDj] = useState(1 / 12);
  const [omega0, setOmega0] = useState(6);
  const [gapPolicy, setGapPolicy] = useState<GapPolicy>("subsegments");
  const [nSurrogates, setNSurrogates] = useState(300);
  const [seed, setSeed] = useState(42);

  const [mapView, setMapView] = useState<"wtc" | "xwt">("wtc");
  const [showContours, setShowContours] = useState(true);
  const [showArrows, setShowArrows] = useState(true);

  const activeJob = useMemo(
    () => state.jobs.find((j) => j.job_id === state.activeWtcJobId) ?? null,
    [state.jobs, state.activeWtcJobId],
  );
  const isJobActive = activeJob !== null && !TERMINAL_JOB_STATUSES.has(activeJob.status);

  const handleLaunch = () => {
    void launchWtcJob({
      period_min_sec: periodMinSec,
      period_max_sec: Math.max(periodMinSec + 60, periodMaxSec),
      dj,
      omega0,
      gap_policy: gapPolicy,
      n_surrogates: nSurrogates,
      alpha: state.alpha,
      seed,
    });
  };

  const hasMc = Boolean(wtcResult && wtcResult.n_surrogates > 0);
  const mcConfPct = Math.round((1 - (wtcResult?.alpha ?? state.alpha)) * 100);

  const phaseVectorsProp = useMemo(() => {
    if (!showArrows || !wtcResult) return null;
    return {
      phaseDeg: mapView === "wtc" ? wtcResult.wtc_phase_deg : wtcResult.xwt_phase_deg,
      sigMask: wtcResult.n_surrogates > 0 ? wtcResult.wtc_is_significant : wtcResult.inside_coi,
    };
  }, [showArrows, wtcResult, mapView]);

  const contourGridProp = useMemo(() => {
    if (!showContours || !wtcResult || wtcResult.n_surrogates === 0) return null;
    return {
      sigGrid: wtcResult.wtc_is_significant,
      insideCoi: wtcResult.inside_coi,
    };
  }, [showContours, wtcResult]);

  const wtcExportBase = wtcResult
    ? buildExportFilename(
        [mapView, wtcResult.episode_id, wtcResult.station_i, wtcResult.station_j, wtcResult.component],
        "svg",
      ).replace(/\.svg$/, "")
    : "wtc_export";

  const getWtcExportSvg = useCallback(() => {
    if (!wtcResult) return null;
    return buildHeatmapPublicationSvg({
      title: `${mapView === "wtc" ? "Wavelet Coherence (WTC R^2)" : "Cross-Wavelet Power (XWT |W_ij|)"} — ${wtcResult.station_i}↔${wtcResult.station_j}`,
      subtitle: `${wtcResult.episode_id} | ${wtcResult.component} (${wtcResult.coord_frame}) | AR(1) a1=${wtcResult.alpha1_i.toFixed(2)}/${wtcResult.alpha1_j.toFixed(2)} | N_MC=${wtcResult.n_surrogates} | alpha=${wtcResult.alpha}`,
      theme: state.exportTheme,
      width: 900,
      height: 340,
      timestampsUtc: wtcResult.timestamps_utc,
      yValues: wtcResult.periods_sec,
      yLabel: "Period [s]",
      yScaleMode: "log",
      invertY: true,
      zMatrix: mapView === "wtc" ? wtcResult.wtc_r2 : wtcResult.xwt_power,
      matrixLayout: "y-by-time",
      zTransform: mapView === "wtc" ? "linear" : "log10",
      zMin: mapView === "wtc" ? 0 : undefined,
      zMax: mapView === "wtc" ? 1 : undefined,
      zUnit: mapView === "wtc" ? "R^2" : "log|W|",
      colormap: mapView === "wtc" ? "turbo" : "plasma",
      viewRangeMs: state.viewRangeMs,
      coiPeriods: wtcResult.coi_periods_sec,
      contourGrid: contourGridProp,
      phaseVectors: phaseVectorsProp,
    });
  }, [wtcResult, mapView, state.exportTheme, state.viewRangeMs, contourGridProp, phaseVectorsProp]);

  const getWtcExportCsv = useCallback(() => {
    if (!wtcResult) return null;
    const mat = mapView === "wtc" ? wtcResult.wtc_r2 : wtcResult.xwt_power;
    const cols = ["period_sec", "scale", "wtc_sig95", ...wtcResult.timestamps_utc];
    const rows = wtcResult.periods_sec.map((pSec, sIdx) => [
      pSec,
      wtcResult.scales[sIdx] ?? null,
      wtcResult.wtc_sig95?.[sIdx] ?? null,
      ...(mat[sIdx] ?? []),
    ]);
    return formatCsvWithMetadata(
      {
        job_id: wtcResult.job_id,
        cache_key: wtcResult.cache_key,
        episode_id: wtcResult.episode_id,
        station_i: wtcResult.station_i,
        station_j: wtcResult.station_j,
        component: wtcResult.component,
        coord_frame: wtcResult.coord_frame,
        matrix_mode: mapView,
        gap_policy: wtcResult.gap_policy,
        alpha1_i: wtcResult.alpha1_i,
        alpha1_j: wtcResult.alpha1_j,
        n_surrogates: wtcResult.n_surrogates,
        alpha: wtcResult.alpha,
      },
      cols,
      rows,
    );
  }, [wtcResult, mapView]);

  return (
    <section className="panel-card deck-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <RadioReceiver size={15} color="#2dd4bf" />
          <span>Wavelet Cross-Spectrum (XWT) &amp; Coherence (WTC)</span>
          <span className="pill pill-accent mono">
            {state.stationA} ↔ {state.stationB} ({state.crossComponent})
          </span>
        </div>

        <div className="panel-controls">
          {activeJob && (
            <>
              <span
                className={`pill mono ${
                  activeJob.status === "completed" || activeJob.status === "cached"
                    ? "pill-ok"
                    : activeJob.status === "failed" || activeJob.status === "cancelled"
                      ? "pill-danger"
                      : "pill-accent"
                }`}
              >
                {formatJobStatus(activeJob.status)} · {activeJob.progress_pct.toFixed(0)}%
              </span>
              {activeJob.eta_sec !== null && isJobActive && (
                <span className="pill mono">ETA: {formatEta(activeJob.eta_sec)}</span>
              )}
            </>
          )}
          <ExportToolbar
            baseFilename={wtcExportBase}
            getSvg={getWtcExportSvg}
            svgWidth={900}
            svgHeight={340}
            getCsv={getWtcExportCsv}
            disabled={!wtcResult}
          />
        </div>
      </div>

      <div className="panel-body">
        <div className="wtc-params-bar">
          <label className="field-label">
            <span>T_min [s]</span>
            <DraftNumberInput
              min={60}
              max={1200}
              step={30}
              value={periodMinSec}
              onCommit={setPeriodMinSec}
            />
          </label>
          <label className="field-label">
            <span>T_max [s]</span>
            <DraftNumberInput
              min={300}
              max={14400}
              step={300}
              value={periodMaxSec}
              onCommit={setPeriodMaxSec}
            />
          </label>
          <label className="field-label">
            <span>AR(1) Surrogates</span>
            <DraftNumberInput
              min={0}
              max={1000}
              step={50}
              integer
              value={nSurrogates}
              onCommit={setNSurrogates}
            />
          </label>
          <label className="field-label">
            <span>Gap Policy</span>
            <select
              value={gapPolicy}
              onChange={(e) => setGapPolicy(e.target.value as GapPolicy)}
            >
              <option value="subsegments">Subsegments</option>
              <option value="gap_coi">Gap-COI</option>
            </select>
          </label>
          <label className="field-label">
            <span>δj / ω0</span>
            <div className="field-row">
              <select value={dj} onChange={(e) => setDj(Number(e.target.value))}>
                <option value={1 / 12}>1/12</option>
                <option value={1 / 8}>1/8</option>
                <option value={1 / 6}>1/6</option>
              </select>
              <DraftNumberInput
                min={4}
                max={10}
                step={1}
                value={omega0}
                onCommit={setOmega0}
              />
            </div>
          </label>
          <label className="field-label">
            <span>Seed</span>
            <DraftNumberInput
              integer
              value={seed}
              onCommit={setSeed}
            />
          </label>
          <label className="field-label">
            <span>Stride</span>
            <select
              value={state.wtcStride}
              onChange={(e) => {
                const nextStride = Number(e.target.value);
                dispatch({ type: "SET_WTC_STRIDE", stride: nextStride });
                if (wtcResult) {
                  void openWtcJobResult(wtcResult.job_id, nextStride);
                }
              }}
            >
              <option value={1}>×1 (1m)</option>
              <option value={2}>×2 (2m)</option>
              <option value={4}>×4 (4m)</option>
            </select>
          </label>

          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <button
              type="button"
              className="btn btn-primary"
              disabled={isJobActive}
              onClick={handleLaunch}
            >
              <Play size={13} />
              <span>Run XWT / WTC</span>
            </button>
            {isJobActive && activeJob && (
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => void cancelWtcJob(activeJob.job_id)}
              >
                <XCircle size={13} />
                <span>Cancel MC</span>
              </button>
            )}
          </div>
        </div>

        {isJobActive && activeJob && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, flexShrink: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
              <span>{activeJob.stage}</span>
              <span className="mono">
                {activeJob.completed_steps} / {activeJob.total_steps}
              </span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${activeJob.progress_pct}%` }} />
            </div>
          </div>
        )}

        {wtcError && <div className="banner banner-danger">{wtcError}</div>}
        {loadingWtcResult && <div className="empty-note">Loading wavelet results…</div>}

        {wtcResult && (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 8,
                flexWrap: "wrap",
                flexShrink: 0,
              }}
            >
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                <span className="pill pill-accent mono">
                  <Sparkles size={12} />
                  <span>
                    {wtcResult.station_i} ↔ {wtcResult.station_j} · AR(1): α₁=
                    {wtcResult.alpha1_i.toFixed(2)}/{wtcResult.alpha1_j.toFixed(2)} · N_MC=
                    {wtcResult.n_surrogates}
                  </span>
                </span>
                <span className="pill mono">Subsegments: {wtcResult.subsegments.length}</span>
              </div>

              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <div className="segmented" style={{ width: 196 }}>
                  <button
                    type="button"
                    className={mapView === "wtc" ? "active" : ""}
                    onClick={() => setMapView("wtc")}
                  >
                    WTC R²
                  </button>
                  <button
                    type="button"
                    className={mapView === "xwt" ? "active" : ""}
                    onClick={() => setMapView("xwt")}
                  >
                    XWT |W_ij|
                  </button>
                </div>
                <button
                  type="button"
                  className={`btn btn-sm ${showContours && hasMc ? "btn-primary" : ""}`}
                  disabled={!hasMc}
                  title={
                    hasMc && mapView === "xwt"
                      ? `Monte Carlo ${mcConfPct}% significance contour belongs to Wavelet Coherence (WTC R²)`
                      : undefined
                  }
                  onClick={() => setShowContours((v) => !v)}
                >
                  {hasMc
                    ? mapView === "xwt"
                      ? `WTC ${mcConfPct}% Contour`
                      : `${mcConfPct}% MC Contour`
                    : "MC disabled (N=0)"}
                </button>
                <button
                  type="button"
                  className={`btn btn-sm ${showArrows ? "btn-primary" : ""}`}
                  onClick={() => setShowArrows((v) => !v)}
                >
                  Phase Arrows
                </button>
              </div>
            </div>

            <ScientificHeatmap2D
              timestampsUtc={wtcResult.timestamps_utc}
              yValues={wtcResult.periods_sec}
              yLabel="Period [s]"
              yScaleMode="log"
              invertY={true}
              zMatrix={mapView === "wtc" ? wtcResult.wtc_r2 : wtcResult.xwt_power}
              matrixLayout="y-by-time"
              zTransform={mapView === "wtc" ? "linear" : "log10"}
              zMin={mapView === "wtc" ? 0 : undefined}
              zMax={mapView === "wtc" ? 1 : undefined}
              zUnit={mapView === "wtc" ? "R²" : "log|W|"}
              colormap={mapView === "wtc" ? "turbo" : "plasma"}
              viewRangeMs={state.viewRangeMs}
              boundsMs={episodeBoundsMs}
              onViewRangeChange={(rangeMs) =>
                dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs })
              }
              onCursorChange={setCursorUtcMs}
              coiPeriods={wtcResult.coi_periods_sec}
              contourGrid={contourGridProp}
              phaseVectors={phaseVectorsProp}
              height="auto"
            />
          </>
        )}

        {!wtcResult && !isJobActive && !loadingWtcResult && (
          <div className="empty-note" style={{ margin: "auto" }}>
            Click &ldquo;Run XWT / WTC&rdquo; to compute cross-wavelet power and wavelet coherence
            with Monte Carlo significance.
          </div>
        )}
      </div>
    </section>
  );
};
