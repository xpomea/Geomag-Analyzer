import React from "react";
import { useCursor } from "../../store/CursorContext";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  computeCursorOverlayPercent,
} from "./useSyncedEChart";

export interface CursorOverlayLineProps {
  viewRangeMs: [number, number] | null;
  top: number;
  bottom?: number;
  height?: number;
  borderColor?: string;
}

export const CursorOverlayLine: React.FC<CursorOverlayLineProps> = ({
  viewRangeMs,
  top,
  bottom,
  height,
  borderColor = "rgba(56, 189, 248, 0.75)",
}) => {
  const { cursorUtcMs } = useCursor();
  const cursorFrac = computeCursorOverlayPercent(cursorUtcMs, viewRangeMs);
  if (cursorFrac === null) {
    return null;
  }
  return (
    <div
      style={{
        position: "absolute",
        top,
        ...(height !== undefined ? { height } : { bottom: bottom ?? 24 }),
        left: `calc(${CHART_GRID_LEFT}px + ${cursorFrac} * (100% - ${CHART_GRID_LEFT + CHART_GRID_RIGHT}px))`,
        width: 1,
        borderLeft: `1px dashed ${borderColor}`,
        pointerEvents: "none",
      }}
    />
  );
};
