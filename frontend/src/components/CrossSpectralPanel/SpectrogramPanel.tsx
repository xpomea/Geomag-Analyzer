import React, { useCallback, useMemo, useState } from "react";
import { RadioReceiver } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import {
  buildExportFilename,
  buildHeatmapPublicationSvg,
  formatCsvWithMetadata,
} from "../../utils/exportUtils";
import { ExportToolbar } from "../Common/ExportToolbar";
import { ScientificHeatmap2D, type BandGuide } from "../SyncedCharts/ScientificHeatmap2D";

type SpecComponentMode = "total" | "x" | "y";

export const SpectrogramPanel: React.FC = () => {
  const { state, dispatch, polarizationRaw, episodeBoundsMs, loadingPolarization } =
    useAnalysis();
  const { setCursorUtcMs } = useCursor();
  const [compMode, setCompMode] = useState<SpecComponentMode>("total");

  const spec = polarizationRaw?.spectrogram ?? null;

  const matrix = useMemo(() => {
    if (!spec) return [];
    if (compMode === "x") return spec.power_x;
    if (compMode === "y") return spec.power_y;
    return spec.power_total;
  }, [spec, compMode]);

  const bandGuides = useMemo<BandGuide[]>(
    () => [
      { value: 1 / 600, label: "Pc5/Pc6 600s", color: "rgba(56, 189, 248, 0.85)" },
      { value: 1 / 150, label: "Pc5 150s", color: "rgba(248, 250, 252, 0.78)" },
      {
        value: 1 / state.tMaxPc6Sec,
        label: `Pc6 ${state.tMaxPc6Sec}s`,
        color: "rgba(167, 139, 250, 0.8)",
      },
    ],
    [state.tMaxPc6Sec],
  );

  const exportBase = buildExportFilename(
    ["spectrogram", state.episodeId, state.stationA, compMode, state.coordFrame],
    "svg",
  ).replace(/\.svg$/, "");

  const getExportSvg = useCallback(() => {
    if (!spec || spec.timestamps_utc.length === 0) return null;
    return buildHeatmapPublicationSvg({
      title: `Welch Spectrogram — ${state.stationA} (${compMode}, ${state.coordFrame})`,
      subtitle: `${state.episodeId} | L=${state.L}, M=${state.M}, win=${state.windowSamples} min | log10(nT^2/Hz)`,
      theme: state.exportTheme,
      width: 900,
      height: 330,
      timestampsUtc: spec.timestamps_utc,
      yValues: spec.freqs_hz,
      yLabel: "Frequency [mHz]",
      yScaleMode: "linear",
      yFormat: (y) => (y * 1000).toFixed(2),
      zMatrix: matrix,
      matrixLayout: "time-by-y",
      zTransform: "log10",
      zUnit: "logP",
      colormap: "viridis",
      viewRangeMs: state.viewRangeMs,
      bandGuidesY: bandGuides,
    });
  }, [spec, matrix, state.stationA, compMode, state.coordFrame, state.episodeId, state.L, state.M, state.windowSamples, state.exportTheme, state.viewRangeMs, bandGuides]);

  const getExportCsv = useCallback(() => {
    if (!spec || spec.timestamps_utc.length === 0) return null;
    const cols = [
      "timestamp_utc",
      ...spec.freqs_hz.map((f) => `f_${(f * 1000).toFixed(3)}mHz`),
    ];
    const rows = spec.timestamps_utc.map((t, i) => [t, ...(matrix[i] ?? [])]);
    return formatCsvWithMetadata(
      {
        episode_id: state.episodeId,
        station_iaga: state.stationA,
        component_mode: compMode,
        coord_frame: state.coordFrame,
        L: state.L,
        M: state.M,
        window_samples: state.windowSamples,
        step_samples: state.stepSamples,
      },
      cols,
      rows,
    );
  }, [spec, matrix, state.episodeId, state.stationA, compMode, state.coordFrame, state.L, state.M, state.windowSamples, state.stepSamples]);

  return (
    <section className="panel-card deck-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <RadioReceiver size={15} color="#2dd4bf" />
          <span>Welch Dynamic Spectrogram — {state.stationA}</span>
          <span className="pill mono">
            L={state.L}, M={state.M}, win={state.windowSamples} min · log10(nT²/Hz)
          </span>
        </div>

        <div className="panel-controls">
          <div className="segmented" style={{ width: 220 }}>
            <button
              type="button"
              className={compMode === "total" ? "active" : ""}
              onClick={() => setCompMode("total")}
            >
              Sxx + Syy
            </button>
            <button
              type="button"
              className={compMode === "x" ? "active" : ""}
              onClick={() => setCompMode("x")}
            >
              {state.coordFrame === "XmYm" ? "Xm" : "X"}
            </button>
            <button
              type="button"
              className={compMode === "y" ? "active" : ""}
              onClick={() => setCompMode("y")}
            >
              {state.coordFrame === "XmYm" ? "Ym" : "Y"}
            </button>
          </div>
          <ExportToolbar
            baseFilename={exportBase}
            getSvg={getExportSvg}
            svgWidth={900}
            svgHeight={330}
            getCsv={getExportCsv}
            disabled={!spec || spec.timestamps_utc.length === 0 || spec.freqs_hz.length === 0}
          />
        </div>
      </div>

      <div className="panel-body">
        {loadingPolarization && !spec ? (
          <div className="empty-note">Computing Welch dynamic spectrogram…</div>
        ) : !spec || spec.timestamps_utc.length === 0 || spec.freqs_hz.length === 0 ? (
          <div className="empty-note">
            Spectrogram unavailable (insufficient valid samples or station absent).
          </div>
        ) : (
          <ScientificHeatmap2D
            timestampsUtc={spec.timestamps_utc}
            yValues={spec.freqs_hz}
            yLabel="Frequency"
            yScaleMode="linear"
            zMatrix={matrix}
            matrixLayout="time-by-y"
            zTransform="log10"
            zUnit="logP"
            colormap="viridis"
            viewRangeMs={state.viewRangeMs}
            boundsMs={episodeBoundsMs}
            onViewRangeChange={(rangeMs) =>
              dispatch({ type: "SET_VIEW_RANGE", rangeMs, boundsMs: episodeBoundsMs })
            }
            onCursorChange={setCursorUtcMs}
            bandGuidesY={bandGuides}
            height="auto"
          />
        )}
      </div>
    </section>
  );
};
