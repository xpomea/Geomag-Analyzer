import React, { useCallback, useMemo, useState } from "react";
import type * as echarts from "echarts";
import { Eye, EyeOff, Waves } from "lucide-react";
import type { Polarization3DOut, Stokes2DOut } from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import type { MaskedPolarizationView } from "../../utils/masking";
import {
  buildExportFilename,
  formatCsvWithMetadata,
  renderEChartsToSvg,
} from "../../utils/exportUtils";
import { formatNum, formatUtcFull, formatUtcShort } from "../../utils/formatters";
import { ExportToolbar } from "../Common/ExportToolbar";
import { CursorOverlayLine } from "./CursorOverlayLine";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  useSyncedEChart,
} from "./useSyncedEChart";

interface StokesSummaryBarProps {
  s2: Stokes2DOut;
  p3: Polarization3DOut | null;
  maskedPolarization: MaskedPolarizationView;
  alpha: number;
  nEffFallback: number | null | undefined;
}

const StokesSummaryBar: React.FC<StokesSummaryBarProps> = ({
  s2,
  p3,
  maskedPolarization,
  alpha,
  nEffFallback,
}) => {
  const { cursorUtcMs } = useCursor();
  const [scope, setScope] = useState<"episode" | "cursor">("episode");

  const cursorMetrics = useMemo(() => {
    const dyn = maskedPolarization.dynamic;
    if (!dyn || dyn.timestamps_utc.length === 0) return null;
    const targetMs = cursorUtcMs ?? Date.parse(dyn.timestamps_utc[0]);
    let bestIdx = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < dyn.timestamps_utc.length; i++) {
      const diff = Math.abs(Date.parse(dyn.timestamps_utc[i]) - targetMs);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIdx = i;
      }
    }
    const pVal = dyn.p[bestIdx] ?? null;
    const pCritVal = dyn.p_crit[bestIdx] ?? null;
    const g2Val = dyn.gamma2[bestIdx] ?? null;
    const g2CritVal = dyn.gamma2_crit[bestIdx] ?? null;
    const isSigP =
      Boolean(dyn.is_significant_p[bestIdx]) ||
      (pVal !== null && pCritVal !== null && pVal > pCritVal);
    const isSigG2 = g2Val !== null && g2CritVal !== null && g2Val > g2CritVal;

    return {
      tsMs: Date.parse(dyn.timestamps_utc[bestIdx]),
      I: dyn.I[bestIdx] ?? null,
      p: pVal,
      pCrit: pCritVal,
      isSigP,
      psi: maskedPolarization.psiSig[bestIdx] ?? null,
      chi: maskedPolarization.chiSig[bestIdx] ?? null,
      eps: maskedPolarization.epsSig[bestIdx] ?? null,
      gamma2: g2Val,
      gamma2Crit: g2CritVal,
      isSigG2,
      nEff: dyn.n_eff[bestIdx] ?? nEffFallback,
    };
  }, [maskedPolarization, cursorUtcMs, nEffFallback]);

  const showCursor = scope === "cursor" && cursorMetrics !== null;

  return (
    <div className="stokes-summary-bar">
      <div className="segmented" style={{ width: 148, flexShrink: 0 }}>
        <button
          type="button"
          className={scope === "episode" ? "active" : ""}
          onClick={() => setScope("episode")}
          title="Show overall polarization summary across the episode"
        >
          Episode
        </button>
        <button
          type="button"
          className={scope === "cursor" ? "active" : ""}
          onClick={() => setScope("cursor")}
          title="Show dynamic polarization values at the nearest UTC cursor window"
        >
          At Cursor
        </button>
      </div>

      {showCursor && cursorMetrics ? (
        <>
          <div className="summary-metric">
            <span className="kpi-label">Window UTC</span>
            <span className="kpi-val">{formatUtcShort(cursorMetrics.tsMs)}</span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Power I</span>
            <span className="kpi-val">{formatNum(cursorMetrics.I, 2)} nT²</span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Degree p / p_crit</span>
            <span className="kpi-val">
              {formatNum(cursorMetrics.p)} / {formatNum(cursorMetrics.pCrit)}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Azimuth ψ</span>
            <span className="kpi-val">
              {cursorMetrics.isSigP && cursorMetrics.psi !== null
                ? `${formatNum(cursorMetrics.psi, 1)}°`
                : "masked"}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Ellipticity ε (χ)</span>
            <span className="kpi-val">
              {cursorMetrics.isSigP && cursorMetrics.eps !== null
                ? `${formatNum(cursorMetrics.eps)} (${formatNum(cursorMetrics.chi, 1)}°)`
                : "masked"}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Coherence γ²</span>
            <span className="kpi-val">
              {formatNum(cursorMetrics.gamma2)} ({cursorMetrics.isSigG2 ? ">" : "≤"}{" "}
              {formatNum(cursorMetrics.gamma2Crit)})
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">n_eff (α={alpha})</span>
            <span className="kpi-val">{formatNum(cursorMetrics.nEff, 1)}</span>
          </div>
        </>
      ) : (
        <>
          <div className="summary-metric">
            <span className="kpi-label">Power I</span>
            <span className="kpi-val">{formatNum(s2.I, 2)} nT²</span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Degree p / p_crit</span>
            <span className="kpi-val">
              {formatNum(s2.p)} / {formatNum(s2.p_crit)}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Azimuth ψ</span>
            <span className="kpi-val">
              {s2.is_significant_p ? `${formatNum(s2.psi_deg, 1)}°` : "masked"}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Ellipticity ε (χ)</span>
            <span className="kpi-val">
              {s2.is_significant_p
                ? `${formatNum(s2.epsilon)} (${formatNum(s2.chi_deg, 1)}°)`
                : "masked"}
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">Coherence γ²</span>
            <span className="kpi-val">
              {formatNum(s2.gamma2)} ({s2.is_significant_gamma2 ? ">" : "≤"}{" "}
              {formatNum(s2.gamma2_crit)})
            </span>
          </div>
          <div className="summary-metric">
            <span className="kpi-label">n_eff (α={alpha})</span>
            <span className="kpi-val">{formatNum(s2.n_eff ?? nEffFallback, 1)}</span>
          </div>
          {p3 && (
            <>
              <div className="summary-metric">
                <span className="kpi-label">3D Samson P</span>
                <span className="kpi-val">{formatNum(p3.P)}</span>
              </div>
              <div className="summary-metric">
                <span className="kpi-label">Means θ / b/a</span>
                <span className="kpi-val">
                  {formatNum(p3.theta_deg, 1)}° / {formatNum(p3.b_over_a)}
                </span>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};

export const StokesPanel: React.FC = () => {
  const {
    state,
    dispatch,
    polarizationRaw,
    maskedPolarization,
    episodeBoundsMs,
    polarizationError,
  } = useAnalysis();
  const { setCursorUtcMs } = useCursor();

  const [logI, setLogI] = useState(false);

  const option = useMemo<echarts.EChartsCoreOption | null>(() => {
    if (!maskedPolarization?.dynamic || maskedPolarization.dynamic.timestamps_utc.length === 0) {
      return null;
    }

    const dyn = maskedPolarization.dynamic;
    const tsMs = dyn.timestamps_utc.map((t) => Date.parse(t));
    const toPairs = (arr: (number | null)[]) =>
      tsMs.map((t, i): [number, number | null] => [t, arr[i] ?? null]);

    const iPlot = dyn.I.map((v) =>
      v !== null && Number.isFinite(v)
        ? logI
          ? v > 0
            ? Math.log10(v)
            : null
          : v
        : null,
    );

    const noiseMarkAreas = maskedPolarization.noiseIntervals.map(([a, b]) => [
      { xAxis: a, itemStyle: { color: "rgba(100, 116, 139, 0.24)" } },
      { xAxis: b },
    ]);

    const xMin = episodeBoundsMs?.[0] ?? tsMs[0];
    const xMax = episodeBoundsMs?.[1] ?? tsMs[tsMs.length - 1];

    const series: Record<string, unknown>[] = [
      {
        name: logI ? "log10(I)" : "I [nT²]",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { width: 1.3, color: "#38bdf8" },
        itemStyle: { color: "#38bdf8" },
        data: toPairs(iPlot),
      },
      {
        name: "p (deg. of pol.)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 1.6, color: "#fbbf24" },
        itemStyle: { color: "#fbbf24" },
        data: toPairs(dyn.p),
      },
      {
        name: `p_crit (α=${state.alpha})`,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 1.3, color: "#ef4444", type: "dashed" },
        itemStyle: { color: "#ef4444" },
        data: toPairs(dyn.p_crit),
      },
      {
        name: "ψ [°] (sig.)",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 2,
        showSymbol: true,
        symbolSize: 4,
        connectNulls: false,
        lineStyle: { width: 1.5, color: "#a78bfa" },
        itemStyle: { color: "#a78bfa" },
        data: toPairs(maskedPolarization.psiSig),
        markArea:
          noiseMarkAreas.length > 0
            ? { silent: true, data: noiseMarkAreas }
            : undefined,
      },
      {
        name: "χ [°] (sig.)",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 3,
        showSymbol: true,
        symbolSize: 4,
        connectNulls: false,
        lineStyle: { width: 1.4, color: "#34d399" },
        itemStyle: { color: "#34d399" },
        data: toPairs(maskedPolarization.chiSig),
      },
      {
        name: "ε (CW >0 / CCW <0)",
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 4,
        showSymbol: true,
        symbolSize: 4,
        connectNulls: false,
        lineStyle: { width: 1.5, color: "#2dd4bf" },
        itemStyle: { color: "#2dd4bf" },
        data: toPairs(maskedPolarization.epsSig),
        markArea:
          noiseMarkAreas.length > 0
            ? { silent: true, data: noiseMarkAreas }
            : undefined,
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: "rgba(148, 163, 184, 0.45)", type: "dashed" },
          data: [{ yAxis: 0 }],
        },
      },
      {
        name: "γ²_xy",
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 5,
        showSymbol: false,
        lineStyle: { width: 1.35, color: "#fb7185" },
        itemStyle: { color: "#fb7185" },
        data: toPairs(dyn.gamma2),
      },
      {
        name: `γ²_crit (α=${state.alpha})`,
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 5,
        showSymbol: false,
        lineStyle: { width: 1.2, color: "#f87171", type: "dotted" },
        itemStyle: { color: "#f87171" },
        data: toPairs(dyn.gamma2_crit),
      },
    ];

    if (state.showNoiseGhost) {
      series.push(
        {
          name: "ψ (noise p<p_crit)",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 2,
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.0, color: "rgba(167, 139, 250, 0.25)", type: "dotted" },
          itemStyle: { color: "rgba(167, 139, 250, 0.35)" },
          data: toPairs(maskedPolarization.psiNoiseGhost),
        },
        {
          name: "ε (noise p<p_crit)",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 4,
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.0, color: "rgba(45, 212, 191, 0.25)", type: "dotted" },
          itemStyle: { color: "rgba(45, 212, 191, 0.35)" },
          data: toPairs(maskedPolarization.epsNoiseGhost),
        },
      );
    }

    return {
      animation: false,
      useUTC: true,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line", link: [{ xAxisIndex: "all" }] },
        backgroundColor: "rgba(15, 23, 42, 0.94)",
        borderColor: "#334155",
        textStyle: { color: "#f1f5f9", fontSize: 11 },
        formatter: (params: unknown) => {
          const items = Array.isArray(params) ? params : [params];
          const first = items[0] as { axisValue?: number } | undefined;
          const header = first?.axisValue ? formatUtcFull(first.axisValue) : "";
          const lines = items.map((it) => {
            const p = it as { seriesName: string; marker: string; value: [number, number | null] };
            const v = p.value?.[1];
            return `${p.marker} ${p.seriesName}: <b>${typeof v === "number" ? v.toFixed(3) : "masked"}</b>`;
          });
          return `<div class="mono">${header}<br/>${lines.join("<br/>")}</div>`;
        },
      },
      legend: {
        top: 2,
        left: CHART_GRID_LEFT,
        right: CHART_GRID_RIGHT,
        itemWidth: 11,
        itemHeight: 6,
        itemGap: 8,
        textStyle: { color: "#94a3b8", fontSize: 10 },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1, 2],
          filterMode: "none",
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ],
      grid: [
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: "12%", height: "22%", containLabel: false },
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: "42%", height: "22%", containLabel: false },
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: "72%", height: "21%", containLabel: false },
      ],
      xAxis: [
        {
          type: "time",
          gridIndex: 0,
          min: xMin,
          max: xMax,
          axisLabel: { show: false },
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.28)" } },
        },
        {
          type: "time",
          gridIndex: 1,
          min: xMin,
          max: xMax,
          axisLabel: { show: false },
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.28)" } },
        },
        {
          type: "time",
          gridIndex: 2,
          min: xMin,
          max: xMax,
          axisLabel: { color: "#94a3b8", fontSize: 10 },
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.28)" } },
        },
      ],
      yAxis: [
        {
          type: "value",
          gridIndex: 0,
          name: logI ? "log I" : "I [nT²]",
          nameTextStyle: { color: "#38bdf8", fontSize: 10 },
          axisLabel: { color: "#38bdf8", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 0,
          position: "right",
          min: 0,
          max: 1,
          name: "p",
          nameTextStyle: { color: "#fbbf24", fontSize: 10 },
          axisLabel: { color: "#fbbf24", fontSize: 10 },
          splitLine: { show: false },
        },
        {
          type: "value",
          gridIndex: 1,
          min: -90,
          max: 90,
          name: "ψ [°]",
          nameTextStyle: { color: "#a78bfa", fontSize: 10 },
          axisLabel: { color: "#a78bfa", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 1,
          position: "right",
          min: -45,
          max: 45,
          name: "χ [°]",
          nameTextStyle: { color: "#34d399", fontSize: 10 },
          axisLabel: { color: "#34d399", fontSize: 10 },
          splitLine: { show: false },
        },
        {
          type: "value",
          gridIndex: 2,
          min: -1,
          max: 1,
          name: "ε",
          nameTextStyle: { color: "#2dd4bf", fontSize: 10 },
          axisLabel: { color: "#2dd4bf", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 2,
          position: "right",
          min: 0,
          max: 1,
          name: "γ²",
          nameTextStyle: { color: "#fb7185", fontSize: 10 },
          axisLabel: { color: "#fb7185", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series,
    };
  }, [maskedPolarization, logI, state.alpha, state.showNoiseGhost, episodeBoundsMs]);

  const { containerRef } = useSyncedEChart({
    option,
    viewRangeMs: state.viewRangeMs,
    boundsMs: episodeBoundsMs,
    onViewRangeChange: (rangeMs) =>
      dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs }),
    onCursorChange: setCursorUtcMs,
  });

  const s2 = maskedPolarization?.stokes2d ?? null;
  const p3 = polarizationRaw?.pol_3d ?? null;

  const exportBase = buildExportFilename(
    ["stokes", state.episodeId, state.stationA, state.band, state.method, state.coordFrame],
    "svg",
  ).replace(/\.svg$/, "");

  const getExportSvg = useCallback(() => {
    if (!option) return null;
    return renderEChartsToSvg(option, {
      width: 920,
      height: 360,
      theme: state.exportTheme,
      viewRangeMs: state.viewRangeMs,
      title: `Stokes Polarization ${state.stationA} (${state.band}, ${state.method})`,
      desc: `${state.episodeId} | ${state.coordFrame} | alpha=${state.alpha}`,
    });
  }, [option, state.exportTheme, state.viewRangeMs, state.stationA, state.band, state.method, state.episodeId, state.coordFrame, state.alpha]);

  const getExportCsv = useCallback(() => {
    const dyn = maskedPolarization?.dynamic;
    if (!dyn || dyn.timestamps_utc.length === 0) return null;
    const rows = dyn.timestamps_utc.map((t, i) => [
      t,
      dyn.I[i] ?? null,
      dyn.p[i] ?? null,
      dyn.p_crit[i] ?? null,
      dyn.is_significant_p[i] ?? false,
      maskedPolarization.psiSig[i] ?? null,
      dyn.psi_deg[i] ?? null,
      maskedPolarization.chiSig[i] ?? null,
      dyn.chi_deg[i] ?? null,
      maskedPolarization.epsSig[i] ?? null,
      dyn.epsilon[i] ?? null,
      dyn.gamma2[i] ?? null,
      dyn.gamma2_crit[i] ?? null,
      dyn.n_eff[i] ?? null,
      dyn.is_valid[i] ?? false,
    ]);
    return formatCsvWithMetadata(
      {
        episode_id: state.episodeId,
        station_iaga: state.stationA,
        method: state.method,
        band: state.band,
        t_max_pc6_sec: state.band === "Pc6" ? state.tMaxPc6Sec : "",
        coord_frame: state.coordFrame,
        d0_deg: polarizationRaw?.d0_deg ?? "",
        alpha: state.alpha,
        spectrum_type: state.spectrumType,
        L: state.L,
        M: state.M,
        window_samples: state.windowSamples,
        step_samples: state.stepSamples,
        overall_I: s2?.I ?? "",
        overall_p: s2?.p ?? "",
        overall_p_crit: s2?.p_crit ?? "",
        overall_psi_deg: s2?.psi_deg ?? "",
        overall_epsilon: s2?.epsilon ?? "",
        overall_3d_P: p3?.P ?? "",
      },
      [
        "timestamp_utc",
        "I_nT2",
        "p",
        "p_crit",
        "is_significant_p",
        "psi_deg_masked",
        "psi_deg_raw",
        "chi_deg_masked",
        "chi_deg_raw",
        "epsilon_masked",
        "epsilon_raw",
        "gamma2",
        "gamma2_crit",
        "n_eff",
        "is_valid",
      ],
      rows,
    );
  }, [
    maskedPolarization,
    polarizationRaw,
    state.episodeId,
    state.stationA,
    state.method,
    state.band,
    state.tMaxPc6Sec,
    state.coordFrame,
    state.alpha,
    state.spectrumType,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    s2,
    p3,
  ]);

  return (
    <section className="panel-card deck-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <Waves size={15} color="#a78bfa" />
          <span>Stokes Polarization 2D/3D — {state.stationA}</span>
          {s2 && (
            <span className={`pill mono ${s2.is_significant_p ? "pill-ok" : "pill-warn"}`}>
              p = {formatNum(s2.p)} {s2.is_significant_p ? ">" : "≤"} p_crit ({formatNum(s2.p_crit)})
            </span>
          )}
        </div>

        <div className="panel-controls">
          <button
            type="button"
            className={`btn btn-sm ${logI ? "btn-primary" : ""}`}
            onClick={() => setLogI((v) => !v)}
          >
            {logI ? "I: log10" : "I: linear"}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() =>
              dispatch({ type: "SET_SHOW_NOISE_GHOST", show: !state.showNoiseGhost })
            }
          >
            {state.showNoiseGhost ? <Eye size={13} /> : <EyeOff size={13} />}
            <span>{state.showNoiseGhost ? "Hide noise p < p_crit" : "Show faint noise"}</span>
          </button>
          <ExportToolbar
            baseFilename={exportBase}
            getSvg={getExportSvg}
            svgWidth={920}
            svgHeight={360}
            getCsv={getExportCsv}
            disabled={!option}
          />
        </div>
      </div>

      <div className="panel-body">
        {polarizationError && <div className="banner banner-danger">{polarizationError}</div>}

        {s2 && maskedPolarization && (
          <StokesSummaryBar
            s2={s2}
            p3={p3}
            maskedPolarization={maskedPolarization}
            alpha={state.alpha}
            nEffFallback={polarizationRaw?.n_eff}
          />
        )}

        {option ? (
          <div className="chart-fill-wrap">
            <div ref={containerRef} className="chart-host" />
            <CursorOverlayLine viewRangeMs={state.viewRangeMs} top="12%" bottom="7%" />
          </div>
        ) : (
          <div className="empty-note">
            No valid Stokes windows for the current station and parameters.
          </div>
        )}
      </div>
    </section>
  );
};
