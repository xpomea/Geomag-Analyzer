import React from "react";
import { ChevronLeft, ChevronRight, Clock, ZoomIn } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import { formatUtcFull, formatUtcShort } from "../../utils/formatters";

export const TimeScrubberBar: React.FC = () => {
  const { state, dispatch, episodeBoundsMs } = useAnalysis();
  const { cursorUtcMs, setCursorUtcMs } = useCursor();

  if (!episodeBoundsMs) {
    return (
      <div className="scrubber-bar">
        <span className="hint-text">Loading episode time grid…</span>
      </div>
    );
  }

  const [bStart, bEnd] = episodeBoundsMs;
  const viewRange = state.viewRangeMs ?? [bStart, bEnd];
  const cursorMs = Math.min(
    bEnd,
    Math.max(bStart, cursorUtcMs ?? Math.round(0.5 * (bStart + bEnd))),
  );
  const stepMs = Math.max(1, state.stepSamples) * 60_000;

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

  const viewSpanMs = viewRange[1] - viewRange[0];
  const fullSpanMs = bEnd - bStart;
  const isPresetActive = (hours: number | null) => {
    if (hours === null) return Math.abs(viewSpanMs - fullSpanMs) < 120_000;
    return Math.abs(viewSpanMs - hours * 3600_000) < 120_000;
  };

  return (
    <div className="scrubber-bar">
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
        <ZoomIn size={13} color="var(--accent-primary)" />
        <span className="mono" style={{ fontSize: 12 }}>
          {formatUtcShort(viewRange[0])} → {formatUtcShort(viewRange[1])}
        </span>
      </div>

      <div className="quick-chips" style={{ flexShrink: 0 }}>
        <button
          type="button"
          className={`chip-btn ${isPresetActive(null) ? "active" : ""}`}
          onClick={() => setDurationPreset(null)}
        >
          Full
        </button>
        {[24, 12, 6].map((h) => (
          <button
            key={h}
            type="button"
            className={`chip-btn ${isPresetActive(h) ? "active" : ""}`}
            onClick={() => setDurationPreset(h)}
          >
            {h}h
          </button>
        ))}
      </div>

      <div className="scrubber-slider-wrap">
        <Clock size={13} color="var(--text-secondary)" style={{ flexShrink: 0 }} />
        <button
          type="button"
          className="btn btn-sm btn-icon"
          onClick={() => setCursorUtcMs(Math.max(bStart, cursorMs - stepMs))}
          disabled={cursorMs <= bStart}
          title={`Step backward (−${state.stepSamples} min)`}
          aria-label="Step backward"
        >
          <ChevronLeft size={13} />
        </button>
        <input
          type="range"
          min={bStart}
          max={bEnd}
          step={60_000}
          value={cursorMs}
          onChange={(e) => setCursorUtcMs(Number(e.target.value))}
          aria-label="UTC time cursor"
        />
        <button
          type="button"
          className="btn btn-sm btn-icon"
          onClick={() => setCursorUtcMs(Math.min(bEnd, cursorMs + stepMs))}
          disabled={cursorMs >= bEnd}
          title={`Step forward (+${state.stepSamples} min)`}
          aria-label="Step forward"
        >
          <ChevronRight size={13} />
        </button>
        <span
          className="mono"
          style={{ fontSize: 12, minWidth: 145, textAlign: "right", flexShrink: 0 }}
        >
          {formatUtcFull(cursorMs)}
        </span>
      </div>
    </div>
  );
};
