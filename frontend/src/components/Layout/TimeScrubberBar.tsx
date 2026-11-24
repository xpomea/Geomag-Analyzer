import React from "react";
import { Clock, ZoomIn } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { formatUtcFull, formatUtcShort } from "../../utils/formatters";

export const TimeScrubberBar: React.FC = () => {
  const { state, dispatch, episodeBoundsMs } = useAnalysis();

  if (!episodeBoundsMs) {
    return (
      <div className="scrubber-bar">
        <span className="hint-text">Загрузка временной сетки эпизода…</span>
      </div>
    );
  }

  const [bStart, bEnd] = episodeBoundsMs;
  const viewRange = state.viewRangeMs ?? [bStart, bEnd];
  const cursorMs = state.cursorUtcMs ?? Math.round(0.5 * (bStart + bEnd));

  const setDurationPreset = (hours: number | null) => {
    if (hours === null) {
      dispatch({ type: "SET_VIEW_RANGE", rangeMs: [bStart, bEnd], boundsMs: episodeBoundsMs });
      return;
    }
    const halfSpan = 0.5 * hours * 3600_000;
    dispatch({
      type: "SET_VIEW_RANGE",
      rangeMs: [cursorMs - halfSpan, cursorMs + halfSpan],
      boundsMs: episodeBoundsMs,
    });
  };

  return (
    <div className="scrubber-bar">
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <ZoomIn size={14} color="var(--accent-primary)" />
        <span className="mono" style={{ fontSize: 12 }}>
          {formatUtcShort(viewRange[0])} → {formatUtcShort(viewRange[1])}
        </span>
      </div>

      <div className="quick-chips">
        <button type="button" className="chip-btn" onClick={() => setDurationPreset(null)}>
          Весь эпизод
        </button>
        <button type="button" className="chip-btn" onClick={() => setDurationPreset(24)}>
          24 ч
        </button>
        <button type="button" className="chip-btn" onClick={() => setDurationPreset(12)}>
          12 ч
        </button>
        <button type="button" className="chip-btn" onClick={() => setDurationPreset(6)}>
          6 ч
        </button>
      </div>

      <div className="scrubber-slider-wrap">
        <Clock size={14} color="var(--text-secondary)" />
        <input
          type="range"
          min={bStart}
          max={bEnd}
          step={60_000}
          value={Math.min(bEnd, Math.max(bStart, cursorMs))}
          onChange={(e) => dispatch({ type: "SET_CURSOR_UTC", utcMs: Number(e.target.value) })}
          aria-label="Курсор времени UTC"
        />
        <span className="mono" style={{ fontSize: 12, minWidth: 145, textAlign: "right" }}>
          {formatUtcFull(cursorMs)}
        </span>
      </div>
    </div>
  );
};
