import React, { useCallback, useEffect, useMemo, useState } from "react";
import * as echarts from "echarts";
import { ArrowLeftRight, ChevronDown, ChevronUp, Database, GitBranch } from "lucide-react";
import type { CrossComponent } from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import {
  buildExportFilename,
  buildHeatmapPublicationSvg,
  formatCsvWithMetadata,
  renderEChartsToSvg,
} from "../../utils/exportUtils";
import { formatNum, formatUtcFull } from "../../utils/formatters";
import { ExportToolbar } from "../Common/ExportToolbar";
import { CursorOverlayLine } from "../SyncedCharts/CursorOverlayLine";
import { ScientificHeatmap2D, type BandGuide } from "../SyncedCharts/ScientificHeatmap2D";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
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
  const { setCursorUtcMs } = useCursor();

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
    onCursorChange: setCursorUtcMs,
  });

  // Separate non-time ECharts chart for Meridional FLR profile (callback-ref safe)
  const flrOption = useMemo<echarts.EChartsCoreOption | null>(() => {
    if (!flrProfile) return null;
    return {
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
    };
  }, [flrProfile]);

  const [flrHostEl, setFlrHostEl] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!flrHostEl || !flrOption || !flrOpen) return;
    const chart = echarts.init(flrHostEl);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(flrHostEl);
    chart.setOption(flrOption);

    return () => {
      ro.disconnect();
      chart.dispose();
    };
  }, [flrHostEl, flrOption, flrOpen]);

  const bandGuides = useMemo<BandGuide[]>(
    () => [{ value: 1 / 600, label: "600с", color: "rgba(248, 250, 252, 0.72)" }],
    [],
  );

  const pairExportBase = buildExportFilename(
    ["welch_pair", state.episodeId, state.stationA, state.stationB, state.crossComponent, state.band],
    "svg",
  ).replace(/\.svg$/, "");

  const getPairExportSvg = useCallback(() => {
    if (!option1D) return null;
    return renderEChartsToSvg(option1D, {
      width: 920,
      height: 260,
      theme: state.exportTheme,
      viewRangeMs: state.viewRangeMs,
      title: `Welch Cross-Spectrum ${state.stationA}-${state.stationB} (${state.crossComponent}, ${state.band})`,
      desc: `${state.episodeId} | alpha=${state.alpha}`,
    });
  }, [option1D, state.exportTheme, state.viewRangeMs, state.stationA, state.stationB, state.crossComponent, state.band, state.episodeId, state.alpha]);

  const getPairExportCsv = useCallback(() => {
    if (!welchPairRaw || !welchPairRaw.available || !maskedWelch) return null;
    const rows = welchPairRaw.window_timestamps_utc.map((t, i) => [
      t,
      welchPairRaw.K_ij[i] ?? null,
      welchPairRaw.n_eff[i] ?? null,
      welchPairRaw.gamma2[i] ?? null,
      maskedWelch.gamma2Crit[i] ?? null,
      maskedWelch.isSignificant[i] ?? false,
      welchPairRaw.phase_deg[i] ?? null,
      maskedWelch.phaseDegMasked[i] ?? null,
      maskedWelch.azimuthalMMasked[i] ?? null,
    ]);
    return formatCsvWithMetadata(
      {
        episode_id: welchPairRaw.episode_id,
        station_i: welchPairRaw.station_i,
        station_j: welchPairRaw.station_j,
        component: welchPairRaw.component,
        coord_frame: welchPairRaw.coord_frame,
        band: welchPairRaw.band,
        alpha: state.alpha,
        delta_lon_deg: welchPairRaw.azimuthal?.delta_lon_deg ?? "",
        m_alias: welchPairRaw.azimuthal?.m_alias ?? "",
      },
      [
        "timestamp_utc",
        "K_ij",
        "n_eff",
        "gamma2",
        "gamma2_crit",
        "is_significant",
        "phase_deg_raw",
        "phase_deg_masked",
        "azimuthal_m_masked",
      ],
      rows,
    );
  }, [welchPairRaw, maskedWelch, state.alpha]);

  const getTfHeatmapSvg = useCallback(() => {
    if (!welchPairRaw || !welchPairRaw.available || !maskedWelch || welchPairRaw.freqs_hz.length === 0) {
      return null;
    }
    return buildHeatmapPublicationSvg({
      title: `Welch 2D ${tfMode === "coherence" ? "Coherence" : "Phase"} (${welchPairRaw.station_i}-${welchPairRaw.station_j})`,
      subtitle: `${welchPairRaw.episode_id} | ${welchPairRaw.component} | alpha=${state.alpha}`,
      theme: state.exportTheme,
      width: 900,
      height: 290,
      timestampsUtc: welchPairRaw.window_timestamps_utc,
      yValues: welchPairRaw.freqs_hz,
      yLabel: "Frequency [mHz]",
      yScaleMode: "linear",
      yFormat: (y) => (y * 1000).toFixed(2),
      zMatrix: tfMode === "coherence" ? welchPairRaw.gamma2_tf : maskedWelch.phaseDegTfMasked,
      matrixLayout: "time-by-y",
      zMin: tfMode === "coherence" ? 0 : -180,
      zMax: tfMode === "coherence" ? 1 : 180,
      zUnit: tfMode === "coherence" ? "gamma2" : "deg",
      colormap: tfMode === "coherence" ? "turbo" : "phase",
      viewRangeMs: state.viewRangeMs,
      bandGuidesY: bandGuides,
    });
  }, [welchPairRaw, maskedWelch, tfMode, state.alpha, state.exportTheme, state.viewRangeMs, bandGuides]);

  const getFlrExportSvg = useCallback(() => {
    if (!flrOption || !flrProfile) return null;
    return renderEChartsToSvg(flrOption, {
      width: 880,
      height: 260,
      theme: state.exportTheme,
      title: `Meridional FLR Profile (${flrProfile.band})`,
      desc: `${flrProfile.episode_id} | ${flrProfile.window_timestamp_utc} | peak=${flrProfile.peak_station}`,
    });
  }, [flrOption, flrProfile, state.exportTheme]);

  const getFlrExportCsv = useCallback(() => {
    if (!flrProfile) return null;
    const rows = flrProfile.station_codes.map((code, i) => [
      code,
      flrProfile.latitudes_deg[i] ?? null,
      flrProfile.band_power[i] ?? null,
      flrProfile.amplitude[i] ?? null,
      flrProfile.cumulative_phase_deg[i] ?? null,
      i < flrProfile.pair_coherence.length ? (flrProfile.pair_coherence[i] ?? null) : null,
      i < flrProfile.pair_gamma2_crit.length ? (flrProfile.pair_gamma2_crit[i] ?? null) : null,
      i < flrProfile.pair_phase_deg.length ? (flrProfile.pair_phase_deg[i] ?? null) : null,
      i < flrProfile.pair_is_significant.length ? flrProfile.pair_is_significant[i] : null,
      i < flrProfile.phase_gradient_deg_per_lat.length
        ? (flrProfile.phase_gradient_deg_per_lat[i] ?? null)
        : null,
    ]);
    return formatCsvWithMetadata(
      {
        episode_id: flrProfile.episode_id,
        band: flrProfile.band,
        window_timestamp_utc: flrProfile.window_timestamp_utc,
        alpha: state.alpha,
        peak_station: flrProfile.peak_station,
        peak_latitude_deg: flrProfile.peak_latitude_deg ?? "",
      },
      [
        "station_code",
        "latitude_deg",
        "band_power",
        "amplitude",
        "cumulative_phase_deg",
        "pair_coherence_to_next",
        "pair_gamma2_crit",
        "pair_phase_deg",
        "pair_is_significant",
        "phase_gradient_deg_per_lat",
      ],
      rows,
    );
  }, [flrProfile, state.alpha]);

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
          <ExportToolbar
            baseFilename={pairExportBase}
            getSvg={getPairExportSvg}
            svgWidth={920}
            svgHeight={260}
            getCsv={getPairExportCsv}
            disabled={!option1D}
          />
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
          <CursorOverlayLine viewRangeMs={state.viewRangeMs} top={18} bottom={24} />
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
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
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
              <ExportToolbar
                baseFilename={`${pairExportBase}_${tfMode}_2d`}
                getSvg={getTfHeatmapSvg}
                svgWidth={900}
                svgHeight={290}
              />
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
            onCursorChange={setCursorUtcMs}
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
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 8,
                flexWrap: "wrap",
                marginBottom: 6,
              }}
            >
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span className="pill pill-accent mono">Окно: {flrProfile.window_timestamp_utc}</span>
                <span className="pill mono">α = {state.alpha}</span>
                <span className="pill pill-ok mono">
                  Пик амплитуды: {flrProfile.peak_station} ({formatNum(flrProfile.peak_latitude_deg, 2)}°N)
                </span>
              </div>
              <ExportToolbar
                baseFilename={buildExportFilename(
                  ["flr_profile", flrProfile.episode_id, flrProfile.band, flrProfile.window_timestamp_utc],
                  "svg",
                ).replace(/\.svg$/, "")}
                getSvg={getFlrExportSvg}
                svgWidth={880}
                svgHeight={260}
                getCsv={getFlrExportCsv}
              />
            </div>
            <div ref={setFlrHostEl} style={{ width: "100%", height: 220 }} />
          </div>
        )}
      </div>
    </div>
  );
};
