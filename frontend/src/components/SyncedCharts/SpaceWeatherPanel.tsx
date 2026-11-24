import React, { useMemo } from "react";
import type * as echarts from "echarts";
import { AlertCircle, Sun } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatUtcFull } from "../../utils/formatters";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  computeCursorOverlayPercent,
  useSyncedEChart,
} from "./useSyncedEChart";

export const SpaceWeatherPanel: React.FC = () => {
  const {
    state,
    dispatch,
    rawSpaceWeather,
    shiftedSolar1,
    episodeBoundsMs,
    loadingTimeseries,
  } = useAnalysis();

  const option = useMemo<echarts.EChartsCoreOption | null>(() => {
    if (!shiftedSolar1 || !rawSpaceWeather) return null;
    const s1 = shiftedSolar1.solar1;
    const tsMs = s1.timestamps_utc.map((t) => Date.parse(t));

    const zipPair = (arr: (number | null)[] | undefined, tArr: number[] = tsMs) => {
      if (!arr) return [];
      const out: [number, number | null][] = new Array(tArr.length);
      for (let i = 0; i < tArr.length; i++) {
        out[i] = [tArr[i], arr[i] ?? null];
      }
      return out;
    };

    const bzData = zipPair(s1.series.bz_gsm_nT);
    const bMagData = zipPair(s1.series.b_mag_gsm_nT);
    const vswData = tsMs.map((t, i): [number, number | null] => {
      const vx = s1.series.vx_gse_kms?.[i];
      return [t, vx !== null && vx !== undefined ? Math.abs(vx) : null];
    });
    const pdynData = zipPair(s1.series.pdyn_nPa);

    const omniTsMs = rawSpaceWeather.omni?.timestamps_utc.map((t) => Date.parse(t)) ?? [];
    const aeData =
      rawSpaceWeather.omni?.available ? zipPair(rawSpaceWeather.omni.series.ae_nT, omniTsMs) : [];
    const alData =
      rawSpaceWeather.omni?.available ? zipPair(rawSpaceWeather.omni.series.al_nT, omniTsMs) : [];
    const auData =
      rawSpaceWeather.omni?.available ? zipPair(rawSpaceWeather.omni.series.au_nT, omniTsMs) : [];

    const ouluTsMs = rawSpaceWeather.oulu?.timestamps_utc.map((t) => Date.parse(t)) ?? [];
    const ouluDeltaData =
      rawSpaceWeather.oulu?.available
        ? zipPair(rawSpaceWeather.oulu.delta_n_pct, ouluTsMs)
        : [];

    const xMin = episodeBoundsMs?.[0] ?? tsMs[0];
    const xMax = episodeBoundsMs?.[1] ?? tsMs[tsMs.length - 1];

    return {
      animation: false,
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
            return `${p.marker} ${p.seriesName}: <b>${typeof v === "number" ? v.toFixed(2) : "—"}</b>`;
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
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 220, height: 88, containLabel: false },
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
          name: "B [нТл]",
          nameTextStyle: { color: "#94a3b8", fontSize: 10 },
          axisLabel: { color: "#94a3b8", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 1,
          name: "Vsw [км/с]",
          nameTextStyle: { color: "#0ea5e9", fontSize: 10 },
          axisLabel: { color: "#0ea5e9", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 1,
          position: "right",
          name: "Pdyn [нПа]",
          nameTextStyle: { color: "#f59e0b", fontSize: 10 },
          axisLabel: { color: "#f59e0b", fontSize: 10 },
          splitLine: { show: false },
        },
        {
          type: "value",
          gridIndex: 2,
          name: "AE [нТл]",
          nameTextStyle: { color: "#fb7185", fontSize: 10 },
          axisLabel: { color: "#fb7185", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 2,
          position: "right",
          name: "ΔN/N0 [%]",
          nameTextStyle: { color: "#2dd4bf", fontSize: 10 },
          axisLabel: { color: "#2dd4bf", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "Bz GSM",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          showSymbol: false,
          lineStyle: { width: 1.4, color: "#f43f5e" },
          data: bzData,
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "rgba(148,163,184,0.45)", type: "dashed" },
            data: [{ yAxis: 0 }],
          },
        },
        {
          name: "|B| GSM",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          showSymbol: false,
          lineStyle: { width: 1.1, color: "#94a3b8" },
          data: bMagData,
        },
        {
          name: "|Vx|",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          showSymbol: false,
          lineStyle: { width: 1.4, color: "#0ea5e9" },
          data: vswData,
        },
        {
          name: "Pdyn",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 2,
          showSymbol: false,
          lineStyle: { width: 1.3, color: "#f59e0b" },
          data: pdynData,
        },
        {
          name: "OMNI AE",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 3,
          showSymbol: false,
          lineStyle: { width: 1.3, color: "#fb7185" },
          data: aeData,
        },
        {
          name: "OMNI AU",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 3,
          showSymbol: false,
          lineStyle: { width: 1.0, color: "#fda4af", type: "dashed" },
          data: auData,
        },
        {
          name: "OMNI AL",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 3,
          showSymbol: false,
          lineStyle: { width: 1.0, color: "#e11d48", type: "dotted" },
          data: alData,
        },
        {
          name: "OULU ΔN/N0",
          type: "line",
          xAxisIndex: 2,
          yAxisIndex: 4,
          showSymbol: false,
          lineStyle: { width: 1.4, color: "#2dd4bf" },
          data: ouluDeltaData,
        },
      ],
    };
  }, [shiftedSolar1, rawSpaceWeather, episodeBoundsMs]);

  const { containerRef } = useSyncedEChart({
    option,
    viewRangeMs: state.viewRangeMs,
    boundsMs: episodeBoundsMs,
    onViewRangeChange: (rangeMs) =>
      dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs }),
    onCursorChange: (utcMs) => dispatch({ type: "SET_CURSOR_UTC", utcMs }),
  });

  const cursorFrac = computeCursorOverlayPercent(state.cursorUtcMs, state.viewRangeMs);

  return (
    <section className="panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <Sun size={15} color="#f59e0b" />
          <span>1. Космическая погода: SOLAR-1 · OMNI · OULU NM</span>
          {shiftedSolar1 && (
            <span className="pill pill-accent mono">{shiftedSolar1.solar1.time_frame}</span>
          )}
        </div>
        <div className="panel-controls">
          {rawSpaceWeather?.omni && !rawSpaceWeather.omni.available && (
            <span className="pill pill-warn">
              <AlertCircle size={12} />
              <span>OMNI отсутствует для этого эпизода</span>
            </span>
          )}
          {loadingTimeseries && <span className="pill">Загрузка…</span>}
        </div>
      </div>
      <div className="panel-body">
        <div style={{ position: "relative", height: 332 }}>
          <div ref={containerRef} className="chart-host" style={{ height: 332 }} />
          {cursorFrac !== null && (
            <div
              style={{
                position: "absolute",
                top: 20,
                bottom: 24,
                left: `calc(${CHART_GRID_LEFT}px + ${cursorFrac} * (100% - ${CHART_GRID_LEFT + CHART_GRID_RIGHT}px))`,
                width: 1,
                borderLeft: "1px dashed rgba(56, 189, 248, 0.75)",
                pointerEvents: "none",
              }}
            />
          )}
        </div>
      </div>
    </section>
  );
};
