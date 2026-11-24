import React, { useMemo } from "react";
import type * as echarts from "echarts";
import { Activity, AlertTriangle } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import { formatUtcFull } from "../../utils/formatters";
import { extractContiguousRanges } from "../../utils/masking";
import { CursorOverlayLine } from "./CursorOverlayLine";
import { CHART_GRID_LEFT, CHART_GRID_RIGHT, useSyncedEChart } from "./useSyncedEChart";

function centeredDelta(arr: (number | null)[]): (number | null)[] {
  let sum = 0;
  let cnt = 0;
  for (const v of arr) {
    if (v !== null && v !== undefined && Number.isFinite(v)) {
      sum += v;
      cnt++;
    }
  }
  if (cnt === 0) return arr;
  const mean = sum / cnt;
  return arr.map((v) => (v !== null && Number.isFinite(v) ? v - mean : null));
}

export const MagnetogramPanel: React.FC = () => {
  const {
    state,
    dispatch,
    geomag,
    polarizationRaw,
    episodeBoundsMs,
    loadingPolarization,
  } = useAnalysis();
  const { setCursorUtcMs } = useCursor();

  const isVariantB = state.method.startsWith("variant_b");
  const hasFiltered = Boolean(isVariantB && polarizationRaw?.filtered);
  const showFiltered = hasFiltered && state.showFilteredMagnetogram;

  const option = useMemo<echarts.EChartsCoreOption | null>(() => {
    if (!geomag || !geomag.available || geomag.timestamps_utc.length === 0) {
      return null;
    }

    const tsMs = geomag.timestamps_utc.map((t) => Date.parse(t));
    const xLabel = state.coordFrame === "XmYm" ? "Xm" : "X";
    const yLabel = state.coordFrame === "XmYm" ? "Ym" : "Y";

    const interpRanges = extractContiguousRanges(tsMs, geomag.is_interpolated, 60_000);
    const despikedPts: { name: string; coord: [number, number] }[] = [];

    let xPlot: (number | null)[];
    let yPlot: (number | null)[];
    let zPlot: (number | null)[];
    let envXTop: (number | null)[] = [];
    let envYTop: (number | null)[] = [];
    let tCutRanges: [number, number][] = [];

    if (isVariantB && polarizationRaw?.filtered) {
      const flt = polarizationRaw.filtered;
      const fltTsMs = flt.timestamps_utc.map((t) => Date.parse(t));
      const trimMask = flt.usable_mask.map((ok) => !ok);
      tCutRanges = extractContiguousRanges(fltTsMs, trimMask, 60_000);
    }

    if (showFiltered && polarizationRaw?.filtered) {
      const flt = polarizationRaw.filtered;
      xPlot = flt.x_filt;
      yPlot = flt.y_filt;
      zPlot = flt.z_filt;
      envXTop = flt.x_amp;
      envYTop = flt.y_amp;
    } else {
      xPlot = centeredDelta(geomag.x_nT);
      yPlot = centeredDelta(geomag.y_nT);
      zPlot = centeredDelta(geomag.z_nT);
      geomag.was_despiked.forEach((flag, idx) => {
        if (flag) {
          const v = zPlot[idx] ?? xPlot[idx] ?? 0;
          despikedPts.push({ name: "Despike", coord: [tsMs[idx], v] });
        }
      });
    }

    const toPairs = (arr: (number | null)[]) =>
      tsMs.map((t, i): [number, number | null] => [t, arr[i] ?? null]);

    const markAreas: Array<[{ xAxis: number; itemStyle: { color: string } }, { xAxis: number }]> = [];
    for (const [a, b] of interpRanges) {
      markAreas.push([
        { xAxis: a, itemStyle: { color: "rgba(245, 158, 11, 0.24)" } },
        { xAxis: b },
      ]);
    }
    for (const [a, b] of tCutRanges) {
      markAreas.push([
        { xAxis: a, itemStyle: { color: "rgba(100, 116, 139, 0.28)" } },
        { xAxis: b },
      ]);
    }

    const xMin = episodeBoundsMs?.[0] ?? tsMs[0];
    const xMax = episodeBoundsMs?.[1] ?? tsMs[tsMs.length - 1];

    const series: Record<string, unknown>[] = [
      {
        name: showFiltered ? `${xLabel} (${state.band})` : `Δ${xLabel}`,
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.35, color: "#38bdf8" },
        data: toPairs(xPlot),
        markArea:
          markAreas.length > 0
            ? { silent: true, data: markAreas }
            : undefined,
      },
      {
        name: showFiltered ? `${yLabel} (${state.band})` : `Δ${yLabel}`,
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.35, color: "#a78bfa" },
        data: toPairs(yPlot),
      },
      {
        name: showFiltered ? `Z (${state.band})` : "ΔZ",
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.15, color: "#34d399" },
        data: toPairs(zPlot),
        markPoint:
          despikedPts.length > 0
            ? {
                symbol: "triangle",
                symbolSize: 8,
                itemStyle: { color: "#f97316" },
                label: { show: false },
                data: despikedPts,
              }
            : undefined,
      },
    ];

    if (showFiltered && envXTop.length === tsMs.length) {
      series.push({
        name: `Огибающая |Z_${xLabel}|`,
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.1, color: "rgba(56, 189, 248, 0.55)", type: "dashed" },
        data: toPairs(envXTop),
      });
    }
    if (showFiltered && envYTop.length === tsMs.length) {
      series.push({
        name: `Огибающая |Z_${yLabel}|`,
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.1, color: "rgba(167, 139, 250, 0.55)", type: "dashed" },
        data: toPairs(envYTop),
      });
    }

    return {
      animation: false,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
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
            return `${p.marker} ${p.seriesName}: <b>${typeof v === "number" ? v.toFixed(2) : "—"} нТл</b>`;
          });
          return `<div class="mono">${header}<br/>${lines.join("<br/>")}</div>`;
        },
      },
      legend: {
        top: 2,
        right: CHART_GRID_RIGHT,
        textStyle: { color: "#94a3b8", fontSize: 10 },
      },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: 0,
          filterMode: "none",
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ],
      grid: {
        left: CHART_GRID_LEFT,
        right: CHART_GRID_RIGHT,
        top: 24,
        bottom: 24,
        containLabel: false,
      },
      xAxis: {
        type: "time",
        min: xMin,
        max: xMax,
        axisLabel: { color: "#94a3b8", fontSize: 10 },
        axisLine: { lineStyle: { color: "#334155" } },
        splitLine: { lineStyle: { color: "rgba(51,65,85,0.28)" } },
      },
      yAxis: {
        type: "value",
        name: "нТл",
        nameTextStyle: { color: "#94a3b8", fontSize: 10 },
        axisLabel: { color: "#94a3b8", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
      },
      series,
    };
  }, [geomag, polarizationRaw, isVariantB, showFiltered, state.coordFrame, state.band, episodeBoundsMs]);

  const { containerRef } = useSyncedEChart({
    option,
    viewRangeMs: state.viewRangeMs,
    boundsMs: episodeBoundsMs,
    onViewRangeChange: (rangeMs) =>
      dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs }),
    onCursorChange: setCursorUtcMs,
  });

  return (
    <section className="panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <Activity size={15} color="#38bdf8" />
          <span>2. Магнитометр {state.stationA}: вариации и полосовые ряды</span>
          {geomag && (
            <span className="pill mono">
              {geomag.coord_frame} (D0={geomag.d0_deg.toFixed(2)}°) · валидн. {geomag.valid_pct.toFixed(1)}%
            </span>
          )}
        </div>
        <div className="panel-controls">
          {hasFiltered && (
            <div className="segmented">
              <button
                type="button"
                className={!showFiltered ? "active" : ""}
                onClick={() => dispatch({ type: "SET_SHOW_FILTERED", show: false })}
              >
                Исходный Δ(X,Y,Z)
              </button>
              <button
                type="button"
                className={showFiltered ? "active" : ""}
                onClick={() => dispatch({ type: "SET_SHOW_FILTERED", show: true })}
              >
                Фильтр {state.band} + огибающая
              </button>
            </div>
          )}
          <span className="pill" style={{ background: "rgba(245,158,11,0.12)", color: "#fcd34d" }}>
            Интерп. ≤3 мин
          </span>
          {polarizationRaw?.trim_hours != null && (
            <span className="pill" style={{ background: "rgba(100,116,139,0.22)", color: "#cbd5e1" }}>
              T_cut = {polarizationRaw.trim_hours} ч
            </span>
          )}
          {loadingPolarization && <span className="pill">Расчёт…</span>}
        </div>
      </div>

      <div className="panel-body">
        {geomag && !geomag.available ? (
          <div className="banner banner-warn">
            <AlertTriangle size={15} />
            <span>{geomag.warnings[0] ?? `Станция ${state.stationA} отсутствует в данном эпизоде.`}</span>
          </div>
        ) : (
          <div style={{ position: "relative", height: 225 }}>
            <div ref={containerRef} className="chart-host" style={{ height: 225 }} />
            <CursorOverlayLine viewRangeMs={state.viewRangeMs} top={24} bottom={24} />
          </div>
        )}
      </div>
    </section>
  );
};
