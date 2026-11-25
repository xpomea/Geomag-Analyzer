import React, { useCallback, useMemo, useState } from "react";
import type * as echarts from "echarts";
import { Eye, EyeOff, Waves } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import {
  buildExportFilename,
  formatCsvWithMetadata,
  renderEChartsToSvg,
} from "../../utils/exportUtils";
import { formatNum, formatUtcFull } from "../../utils/formatters";
import { ExportToolbar } from "../Common/ExportToolbar";
import { CursorOverlayLine } from "./CursorOverlayLine";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  useSyncedEChart,
} from "./useSyncedEChart";

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
        name: logI ? "log10(I)" : "I [нТл²]",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { width: 1.3, color: "#38bdf8" },
        data: toPairs(iPlot),
      },
      {
        name: "p (степень поляриз.)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 1.6, color: "#fbbf24" },
        data: toPairs(dyn.p),
      },
      {
        name: `p_crit (α=${state.alpha})`,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 1.3, color: "#ef4444", type: "dashed" },
        data: toPairs(dyn.p_crit),
      },
      {
        name: "ψ [°] (значим.)",
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
        name: "χ [°] (значим.)",
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
        data: toPairs(dyn.gamma2),
      },
      {
        name: `γ²_crit (α=${state.alpha})`,
        type: "line",
        xAxisIndex: 2,
        yAxisIndex: 5,
        showSymbol: false,
        lineStyle: { width: 1.2, color: "#f87171", type: "dotted" },
        data: toPairs(dyn.gamma2_crit),
      },
    ];

    if (state.showNoiseGhost) {
      series.push(
        {
          name: "ψ (шум p<p_crit)",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 2,
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.0, color: "rgba(167, 139, 250, 0.25)", type: "dotted" },
          data: toPairs(maskedPolarization.psiNoiseGhost),
        },
        {
          name: "ε (шум p<p_crit)",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 4,
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.0, color: "rgba(45, 212, 191, 0.25)", type: "dotted" },
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
            return `${p.marker} ${p.seriesName}: <b>${typeof v === "number" ? v.toFixed(3) : "маск."}</b>`;
          });
          return `<div class="mono">${header}<br/>${lines.join("<br/>")}</div>`;
        },
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
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 20, height: 84, containLabel: false },
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 120, height: 84, containLabel: false },
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 220, height: 84, containLabel: false },
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
          name: logI ? "log I" : "I [нТл²]",
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
    <section className="panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <Waves size={15} color="#a78bfa" />
          <span>3. Поляризация Стокса 2D/3D (p, p_crit, ψ, χ, ε, γ²)</span>
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
            {logI ? "I: log10" : "I: линейн."}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() =>
              dispatch({ type: "SET_SHOW_NOISE_GHOST", show: !state.showNoiseGhost })
            }
          >
            {state.showNoiseGhost ? <Eye size={13} /> : <EyeOff size={13} />}
            <span>{state.showNoiseGhost ? "Скрыть шум p<p_crit" : "Показать шум бледно"}</span>
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

        {s2 && (
          <div className="kpi-strip">
            <div className="kpi-box">
              <div className="kpi-label">Мощность I</div>
              <div className="kpi-val">{formatNum(s2.I, 2)} нТл²</div>
            </div>
            <div className="kpi-box">
              <div className="kpi-label">Степень p / p_crit</div>
              <div className="kpi-val">
                {formatNum(s2.p)} / {formatNum(s2.p_crit)}
              </div>
            </div>
            <div className="kpi-box">
              <div className="kpi-label">Азимут ψ</div>
              <div className="kpi-val">{s2.is_significant_p ? `${formatNum(s2.psi_deg, 1)}°` : "маск."}</div>
            </div>
            <div className="kpi-box">
              <div className="kpi-label">Эллипт. ε (χ)</div>
              <div className="kpi-val">
                {s2.is_significant_p
                  ? `${formatNum(s2.epsilon)} (${formatNum(s2.chi_deg, 1)}°)`
                  : "маск."}
              </div>
            </div>
            <div className="kpi-box">
              <div className="kpi-label">Когерент. γ²</div>
              <div className="kpi-val">
                {formatNum(s2.gamma2)} (
                {s2.is_significant_gamma2 ? ">" : "≤"} {formatNum(s2.gamma2_crit)})
              </div>
            </div>
            <div className="kpi-box">
              <div className="kpi-label">n_eff (α={state.alpha})</div>
              <div className="kpi-val">{formatNum(s2.n_eff ?? polarizationRaw?.n_eff, 1)}</div>
            </div>
            {p3 && (
              <>
                <div className="kpi-box">
                  <div className="kpi-label">3D Самсон P</div>
                  <div className="kpi-val">{formatNum(p3.P)}</div>
                </div>
                <div className="kpi-box">
                  <div className="kpi-label">Минс θ / b/a</div>
                  <div className="kpi-val">
                    {formatNum(p3.theta_deg, 1)}° / {formatNum(p3.b_over_a)}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {option ? (
          <div style={{ position: "relative", height: 328 }}>
            <div ref={containerRef} className="chart-host" style={{ height: 328 }} />
            <CursorOverlayLine viewRangeMs={state.viewRangeMs} top={20} bottom={24} />
          </div>
        ) : (
          <div className="empty-note">Нет валидных окон Стокса для текущей станции и параметров.</div>
        )}
      </div>
    </section>
  );
};
