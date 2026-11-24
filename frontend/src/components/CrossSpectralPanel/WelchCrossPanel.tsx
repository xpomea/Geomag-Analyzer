import React, { useEffect, useMemo, useState } from "react";
import * as echarts from "echarts";
import { ArrowLeftRight, ChevronDown, ChevronUp, Database, GitBranch } from "lucide-react";
import type { CrossComponent } from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatNum, formatUtcFull } from "../../utils/formatters";
import { ScientificHeatmap2D, type BandGuide } from "../SyncedCharts/ScientificHeatmap2D";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  computeCursorOverlayPercent,
  useSyncedEChart,
} from "../SyncedCharts/useSyncedEChart";

export const WelchCrossPanel: React.FC = () => {
  const {
    state,
    dispatch,
    welchPairRaw,
    maskedWelch,
    flrProfile,
    episodeBoundsMs,
    loadingWelch,
    loadingFlr,
    welchError,
    flrError,
    loadFlrAtCursor,
  } = useAnalysis();

  const [tfMode, setTfMode] = useState<"coherence" | "phase">("coherence");
  const [flrOpen, setFlrOpen] = useState(false);

  const availableComponents: CrossComponent[] =
    state.coordFrame === "XmYm" ? ["Xm", "Ym", "Z"] : ["X", "Y", "Z"];

  const option1D = useMemo<echarts.EChartsCoreOption | null>(() => {
    if (!welchPairRaw || !welchPairRaw.available || !maskedWelch) return null;
    const tsMs = welchPairRaw.window_timestamps_utc.map((t) => Date.parse(t));
    if (tsMs.length === 0) return null;

    const toPairs = (arr: (number | null)[]) =>
      tsMs.map((t, i): [number, number | null] => [t, arr[i] ?? null]);

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
            return `${p.marker} ${p.seriesName}: <b>${typeof v === "number" ? v.toFixed(3) : "маск."}</b>`;
          });
          return `<div class="mono">${header}<br/>${lines.join("<br/>")}</div>`;
        },
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0, 1],
          filterMode: "none",
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ],
      grid: [
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 18, height: 76, containLabel: false },
        { left: CHART_GRID_LEFT, right: CHART_GRID_RIGHT, top: 112, height: 76, containLabel: false },
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
          axisLabel: { color: "#94a3b8", fontSize: 10 },
          axisLine: { lineStyle: { color: "#334155" } },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.28)" } },
        },
      ],
      yAxis: [
        {
          type: "value",
          gridIndex: 0,
          min: 0,
          max: 1,
          name: "γ²_12",
          nameTextStyle: { color: "#38bdf8", fontSize: 10 },
          axisLabel: { color: "#94a3b8", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 1,
          min: -180,
          max: 180,
          name: "Δφ [°]",
          nameTextStyle: { color: "#f59e0b", fontSize: 10 },
          axisLabel: { color: "#f59e0b", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          gridIndex: 1,
          position: "right",
          name: "m",
          nameTextStyle: { color: "#2dd4bf", fontSize: 10 },
          axisLabel: { color: "#2dd4bf", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "γ²_12 (Уэлч)",
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          showSymbol: false,
          lineStyle: { width: 1.5, color: "#38bdf8" },
          data: toPairs(welchPairRaw.gamma2),
        },
        {
          name: `γ²_crit (α=${state.alpha})`,
          type: "line",
          xAxisIndex: 0,
          yAxisIndex: 0,
          showSymbol: false,
          lineStyle: { width: 1.3, color: "#ef4444", type: "dashed" },
          data: toPairs(maskedWelch.gamma2Crit),
        },
        {
          name: "Δφ_12 (значим.)",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 1,
          showSymbol: true,
          symbolSize: 4,
          connectNulls: false,
          lineStyle: { width: 1.4, color: "#f59e0b" },
          itemStyle: { color: "#f59e0b" },
          data: toPairs(maskedWelch.phaseDegMasked),
        },
        {
          name: "Азимутальн. число m",
          type: "line",
          xAxisIndex: 1,
          yAxisIndex: 2,
          showSymbol: true,
          symbolSize: 4,
          connectNulls: false,
          lineStyle: { width: 1.2, color: "#2dd4bf", type: "dotted" },
          itemStyle: { color: "#2dd4bf" },
          data: toPairs(maskedWelch.azimuthalMMasked),
        },
      ],
    };
  }, [welchPairRaw, maskedWelch, state.alpha, episodeBoundsMs]);

  const { containerRef } = useSyncedEChart({
    option: option1D,
    viewRangeMs: state.viewRangeMs,
    boundsMs: episodeBoundsMs,
    onViewRangeChange: (rangeMs) =>
      dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs }),
    onCursorChange: (utcMs) => dispatch({ type: "SET_CURSOR_UTC", utcMs }),
  });

  // Separate non-time ECharts chart for Meridional FLR profile (callback-ref safe)
  const [flrHostEl, setFlrHostEl] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!flrHostEl || !flrProfile || !flrOpen) return;
    const chart = echarts.init(flrHostEl);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(flrHostEl);

    chart.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(15, 23, 42, 0.94)",
        borderColor: "#334155",
        textStyle: { color: "#f1f5f9", fontSize: 11 },
      },
      legend: {
        top: 2,
        textStyle: { color: "#94a3b8", fontSize: 10 },
      },
      grid: { left: 60, right: 60, top: 28, bottom: 32 },
      xAxis: {
        type: "category",
        data: flrProfile.station_codes.map(
          (c, i) => `${c} (${flrProfile.latitudes_deg[i]?.toFixed(1)}°)`,
        ),
        axisLabel: { color: "#94a3b8", fontSize: 10 },
        axisLine: { lineStyle: { color: "#334155" } },
      },
      yAxis: [
        {
          type: "value",
          name: "Амплитуда",
          nameTextStyle: { color: "#38bdf8", fontSize: 10 },
          axisLabel: { color: "#38bdf8", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(51,65,85,0.3)" } },
        },
        {
          type: "value",
          position: "right",
          name: "Накопл. фаза [°]",
          nameTextStyle: { color: "#f59e0b", fontSize: 10 },
          axisLabel: { color: "#f59e0b", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "Амплитуда полосы",
          type: "line",
          yAxisIndex: 0,
          symbolSize: 7,
          lineStyle: { width: 2, color: "#38bdf8" },
          itemStyle: { color: "#38bdf8" },
          data: flrProfile.amplitude,
        },
        {
          name: "Накопленная фаза [°]",
          type: "line",
          yAxisIndex: 1,
          symbolSize: 6,
          connectNulls: false,
          lineStyle: { width: 2, color: "#f59e0b" },
          itemStyle: { color: "#f59e0b" },
          data: flrProfile.cumulative_phase_deg,
        },
      ],
    });

    return () => {
      ro.disconnect();
      chart.dispose();
    };
  }, [flrHostEl, flrProfile, flrOpen]);

  const bandGuides = useMemo<BandGuide[]>(
    () => [{ value: 1 / 600, label: "600с", color: "rgba(248, 250, 252, 0.72)" }],
    [],
  );

  const cursorFrac = computeCursorOverlayPercent(state.cursorUtcMs, state.viewRangeMs);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span className="pill pill-accent mono">
            {state.stationA} ↔ {state.stationB}
          </span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() =>
              dispatch({
                type: "SET_PAIR",
                stationA: state.stationB,
                stationB: state.stationA,
              })
            }
          >
            <ArrowLeftRight size={12} />
            <span>Поменять</span>
          </button>

          <div className="segmented">
            {availableComponents.map((comp) => (
              <button
                key={comp}
                type="button"
                className={state.crossComponent === comp ? "active" : ""}
                onClick={() => dispatch({ type: "SET_CROSS_COMPONENT", component: comp })}
              >
                {comp}
              </button>
            ))}
          </div>

          {welchPairRaw?.cache_hits &&
            Object.entries(welchPairRaw.cache_hits).map(([code, hit]) => (
              <span key={code} className={`pill mono ${hit ? "pill-ok" : ""}`}>
                <Database size={11} />
                <span>
                  {code}: {hit ? "cache hit" : "computed"}
                </span>
              </span>
            ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {welchPairRaw?.azimuthal && (
            <span
              className={`pill mono ${welchPairRaw.azimuthal.is_meridional_guarded ? "pill-warn" : ""}`}
            >
              Δλ = {welchPairRaw.azimuthal.delta_lon_deg.toFixed(2)}°
              {welchPairRaw.azimuthal.is_meridional_guarded
                ? " (мерид. защита m)"
                : ` · |m_alias| = ${formatNum(welchPairRaw.azimuthal.m_alias, 1)}`}
            </span>
          )}
          {loadingWelch && <span className="pill">Расчёт Уэлча…</span>}
        </div>
      </div>

      {welchError && <div className="banner banner-danger">{welchError}</div>}

      {welchPairRaw && !welchPairRaw.available && (
        <div className="banner banner-warn">
          {welchPairRaw.warnings.join(" ") || "Одна из станций пары отсутствует в эпизоде."}
        </div>
      )}

      {option1D && (
        <div style={{ position: "relative", height: 224 }}>
          <div ref={containerRef} className="chart-host" style={{ height: 224 }} />
          {cursorFrac !== null && (
            <div
              style={{
                position: "absolute",
                top: 18,
                bottom: 24,
                left: `calc(${CHART_GRID_LEFT}px + ${cursorFrac} * (100% - ${CHART_GRID_LEFT + CHART_GRID_RIGHT}px))`,
                width: 1,
                borderLeft: "1px dashed rgba(56, 189, 248, 0.75)",
                pointerEvents: "none",
              }}
            />
          )}
        </div>
      )}

      {welchPairRaw && welchPairRaw.available && maskedWelch && welchPairRaw.freqs_hz.length > 0 && (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <span className="hint-text">
              2D частотно-временная карта пары ({welchPairRaw.station_i}–{welchPairRaw.station_j}),
              α = {state.alpha}
            </span>
            <div className="segmented" style={{ width: 290 }}>
              <button
                type="button"
                className={tfMode === "coherence" ? "active" : ""}
                onClick={() => setTfMode("coherence")}
              >
                Когерентность γ²(t, f)
              </button>
              <button
                type="button"
                className={tfMode === "phase" ? "active" : ""}
                onClick={() => setTfMode("phase")}
              >
                Фаза Δφ(t, f) (α-маска)
              </button>
            </div>
          </div>

          <ScientificHeatmap2D
            timestampsUtc={welchPairRaw.window_timestamps_utc}
            yValues={welchPairRaw.freqs_hz}
            yLabel="Частота"
            yScaleMode="linear"
            zMatrix={tfMode === "coherence" ? welchPairRaw.gamma2_tf : maskedWelch.phaseDegTfMasked}
            matrixLayout="time-by-y"
            zMin={tfMode === "coherence" ? 0 : -180}
            zMax={tfMode === "coherence" ? 1 : 180}
            zUnit={tfMode === "coherence" ? "γ²" : "°"}
            colormap={tfMode === "coherence" ? "turbo" : "phase"}
            viewRangeMs={state.viewRangeMs}
            boundsMs={episodeBoundsMs}
            onViewRangeChange={(rangeMs) =>
              dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs })
            }
            cursorUtcMs={state.cursorUtcMs}
            onCursorChange={(utcMs) => dispatch({ type: "SET_CURSOR_UTC", utcMs })}
            bandGuidesY={bandGuides}
            height={210}
          />
        </>
      )}

      <div
        style={{
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-md)",
          padding: 8,
          background: "rgba(15, 23, 42, 0.45)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <GitBranch size={14} color="var(--accent-primary)" />
            <span style={{ fontWeight: 600, fontSize: 12 }}>
              Меридиональный профиль FLR ({state.meridionalChain.join(" → ")})
            </span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={loadingFlr}
              onClick={() => {
                setFlrOpen(true);
                void loadFlrAtCursor();
              }}
            >
              {loadingFlr ? "Расчёт FLR…" : "Рассчитать FLR у курсора"}
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setFlrOpen((v) => !v)}
            >
              {flrOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
          </div>
        </div>

        {flrError && (
          <div className="banner banner-danger" style={{ marginTop: 8 }}>
            {flrError}
          </div>
        )}

        {flrOpen && flrProfile && (
          <div style={{ marginTop: 8 }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
              <span className="pill pill-accent mono">Окно: {flrProfile.window_timestamp_utc}</span>
              <span className="pill pill-ok mono">
                Пик амплитуды: {flrProfile.peak_station} ({formatNum(flrProfile.peak_latitude_deg, 2)}°N)
              </span>
            </div>
            <div ref={setFlrHostEl} style={{ width: "100%", height: 220 }} />
          </div>
        )}
      </div>
    </div>
  );
};
