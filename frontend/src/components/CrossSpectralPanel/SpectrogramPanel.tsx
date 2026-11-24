import React, { useMemo, useState } from "react";
import { useAnalysis } from "../../store/AnalysisContext";
import { ScientificHeatmap2D, type BandGuide } from "../SyncedCharts/ScientificHeatmap2D";

type SpecComponentMode = "total" | "x" | "y";

export const SpectrogramPanel: React.FC = () => {
  const { state, dispatch, polarizationRaw, episodeBoundsMs, loadingPolarization } =
    useAnalysis();
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
      { value: 1 / 600, label: "Pc5/Pc6 600с", color: "rgba(56, 189, 248, 0.85)" },
      { value: 1 / 150, label: "Pc5 150с", color: "rgba(248, 250, 252, 0.78)" },
      {
        value: 1 / state.tMaxPc6Sec,
        label: `Pc6 ${state.tMaxPc6Sec}с`,
        color: "rgba(167, 139, 250, 0.8)",
      },
    ],
    [state.tMaxPc6Sec],
  );

  if (loadingPolarization && !spec) {
    return <div className="empty-note">Расчёт динамической спектрограммы Уэлча…</div>;
  }

  if (!spec || spec.timestamps_utc.length === 0 || spec.freqs_hz.length === 0) {
    return (
      <div className="empty-note">
        Спектрограмма недоступна (недостаточно отсчётов в окне или станция отсутствует).
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div className="hint-text">
          Односторонний спектр мощности Уэлча (L={state.L}, M={state.M}, окно={state.windowSamples} мин).
          Шкала: <span className="mono">log10(нТл²/Гц)</span>.
        </div>
        <div className="segmented" style={{ width: 280 }}>
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
      </div>

      <ScientificHeatmap2D
        timestampsUtc={spec.timestamps_utc}
        yValues={spec.freqs_hz}
        yLabel="Частота"
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
        cursorUtcMs={state.cursorUtcMs}
        onCursorChange={(utcMs) => dispatch({ type: "SET_CURSOR_UTC", utcMs })}
        bandGuidesY={bandGuides}
        height={255}
      />
    </div>
  );
};
