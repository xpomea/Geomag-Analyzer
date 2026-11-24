import React, { useEffect, useMemo, useRef, useState } from "react";
import { MIN_ZOOM_SPAN_MS } from "../../store/analysisReducer";
import { buildColorLut, extractSignificanceContours, type ColormapName } from "../../utils/contours";
import { formatUtcFull, formatUtcShort } from "../../utils/formatters";
import { CursorOverlayLine } from "./CursorOverlayLine";
import {
  CHART_GRID_LEFT,
  CHART_GRID_RIGHT,
  computeCursorOverlayPercent,
} from "./useSyncedEChart";

export function computeVisibleArrowStepX(visibleCount: number, explicitStepX?: number): number {
  if (explicitStepX !== undefined && explicitStepX >= 1) {
    return Math.round(explicitStepX);
  }
  return Math.max(1, Math.floor(Math.max(1, visibleCount) / 32));
}

export function findNearestYIndex(
  yValues: number[],
  targetY: number,
  yScaleMode: "linear" | "log",
): number {
  if (yValues.length === 0) return 0;
  let nearestY = 0;
  let minYDiff = Number.POSITIVE_INFINITY;
  const logTarget = yScaleMode === "log" ? Math.log10(Math.max(1e-12, targetY)) : 0;
  for (let j = 0; j < yValues.length; j++) {
    const val = yValues[j];
    const d =
      yScaleMode === "log"
        ? Math.abs(Math.log10(Math.max(1e-12, val)) - logTarget)
        : Math.abs(val - targetY);
    if (d < minYDiff) {
      minYDiff = d;
      nearestY = j;
    }
  }
  return nearestY;
}

export interface BandGuide {
  value: number;
  label: string;
  color?: string;
}

export interface ScientificHeatmap2DProps {
  timestampsUtc: string[];
  yValues: number[];
  yLabel: string;
  yScaleMode: "linear" | "log";
  invertY?: boolean;
  yFormat?: (y: number) => string;
  zMatrix: (number | null)[][];
  matrixLayout: "time-by-y" | "y-by-time";
  zTransform?: "linear" | "log10";
  zMin?: number;
  zMax?: number;
  zUnit?: string;
  colormap: ColormapName;
  viewRangeMs: [number, number] | null;
  boundsMs: [number, number] | null;
  onViewRangeChange: (range: [number, number]) => void;
  cursorUtcMs?: number | null;
  onCursorChange?: (utcMs: number) => void;
  bandGuidesY?: BandGuide[];
  coiPeriods?: (number | null)[];
  contourGrid?: { sigGrid: boolean[][]; insideCoi?: boolean[][] } | null;
  phaseVectors?: {
    phaseDeg: (number | null)[][];
    sigMask?: boolean[][];
    stepX?: number;
    stepY?: number;
  } | null;
  height?: number;
}

const OFFSCREEN_RASTER_H = 256;

function defaultYFormatter(y: number): string {
  if (y < 0.05) {
    return `${(y * 1000).toFixed(2)} мГц`;
  }
  if (y >= 60) {
    return `${Math.round(y)} с`;
  }
  return y.toFixed(2);
}

export const ScientificHeatmap2D: React.FC<ScientificHeatmap2DProps> = ({
  timestampsUtc,
  yValues,
  yLabel,
  yScaleMode,
  invertY = false,
  yFormat = defaultYFormatter,
  zMatrix,
  matrixLayout,
  zTransform = "linear",
  zMin,
  zMax,
  zUnit = "",
  colormap,
  viewRangeMs,
  boundsMs,
  onViewRangeChange,
  cursorUtcMs,
  onCursorChange,
  bandGuidesY,
  coiPeriods,
  contourGrid,
  phaseVectors,
  height = 250,
}) => {
  const [wrapEl, setWrapEl] = useState<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [width, setWidth] = useState(720);
  const [hoverInfo, setHoverInfo] = useState<{
    xPx: number;
    yPx: number;
    utcMs: number;
    yVal: number;
    zVal: number | null;
  } | null>(null);
  const dragRef = useRef<{ startX: number; origRange: [number, number] } | null>(null);
  const lastCursorEmitRef = useRef(0);

  useEffect(() => {
    if (!wrapEl) return;
    const ro = new ResizeObserver((entries) => {
      const nextW = Math.floor(entries[0]?.contentRect.width ?? wrapEl.clientWidth);
      if (nextW > 180) {
        setWidth(nextW);
      }
    });
    ro.observe(wrapEl);
    return () => ro.disconnect();
  }, [wrapEl]);

  const timesMs = useMemo(() => timestampsUtc.map((s) => Date.parse(s)), [timestampsUtc]);
  const lut = useMemo(() => buildColorLut(colormap), [colormap]);

  const { zLow, zHigh } = useMemo(() => {
    if (zMin !== undefined && zMax !== undefined) {
      return { zLow: zMin, zHigh: zMax };
    }
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let r = 0; r < zMatrix.length; r++) {
      const row = zMatrix[r];
      for (let c = 0; c < row.length; c++) {
        const raw = row[c];
        if (raw === null || raw === undefined || !Number.isFinite(raw)) continue;
        const val = zTransform === "log10" ? (raw > 0 ? Math.log10(raw) : Number.NaN) : raw;
        if (!Number.isFinite(val)) continue;
        if (val < lo) lo = val;
        if (val > hi) hi = val;
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      return { zLow: 0, zHigh: 1 };
    }
    if (Math.abs(hi - lo) < 1e-12) {
      return { zLow: lo - 0.5, zHigh: hi + 0.5 };
    }
    return { zLow: zMin ?? lo, zHigh: zMax ?? hi };
  }, [zMatrix, zTransform, zMin, zMax]);

  const contourSegments = useMemo(() => {
    if (!contourGrid) return [];
    return extractSignificanceContours(contourGrid.sigGrid, contourGrid.insideCoi);
  }, [contourGrid]);

  const effectiveRange = useMemo<[number, number]>(() => {
    if (viewRangeMs) return viewRangeMs;
    if (timesMs.length >= 2) return [timesMs[0], timesMs[timesMs.length - 1]];
    return [0, 1];
  }, [viewRangeMs, timesMs]);

  const plotLeft = CHART_GRID_LEFT;
  const plotRight = Math.max(plotLeft + 40, width - CHART_GRID_RIGHT);
  const plotTop = 14;
  const plotBottom = Math.max(plotTop + 40, height - 28);
  const plotW = plotRight - plotLeft;
  const plotH = plotBottom - plotTop;

  const wheelStateRef = useRef({
    effectiveRange,
    plotLeft,
    plotW,
    onViewRangeChange,
  });
  wheelStateRef.current = {
    effectiveRange,
    plotLeft,
    plotW,
    onViewRangeChange,
  };

  // Native non-passive wheel listener avoids passive-event warnings and blocks page scroll during zoom
  useEffect(() => {
    if (!wrapEl) return;
    const onNativeWheel = (e: WheelEvent) => {
      const { effectiveRange: er, plotLeft: pl, plotW: pw, onViewRangeChange: emit } =
        wheelStateRef.current;
      if (pw <= 0) return;
      e.preventDefault();
      const rect = wrapEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frac = Math.min(1, Math.max(0, (x - pl) / pw));
      const [s, en] = er;
      const span = en - s;
      const factor = e.deltaY > 0 ? 1.25 : 0.8;
      const anchor = s + frac * span;
      const nextSpan = Math.max(MIN_ZOOM_SPAN_MS, span * factor);
      const nextS = anchor - frac * nextSpan;
      emit([nextS, nextS + nextSpan]);
    };
    wrapEl.addEventListener("wheel", onNativeWheel, { passive: false });
    return () => wrapEl.removeEventListener("wheel", onNativeWheel);
  }, [wrapEl]);

  const yBounds = useMemo(() => {
    if (yValues.length === 0) return { min: 0, max: 1 };
    const first = yValues[0];
    const last = yValues[yValues.length - 1];
    return { min: Math.min(first, last), max: Math.max(first, last) };
  }, [yValues]);

  const yToNorm = useMemo(() => {
    const yMinV = yBounds.min;
    const yMaxV = yBounds.max;
    return (y: number): number => {
      if (yScaleMode === "log") {
        const l0 = Math.log10(Math.max(1e-12, yMinV));
        const l1 = Math.log10(Math.max(1e-12, yMaxV));
        const ly = Math.log10(Math.max(1e-12, y));
        return l1 === l0 ? 0.5 : (ly - l0) / (l1 - l0);
      }
      return yMaxV === yMinV ? 0.5 : (y - yMinV) / (yMaxV - yMinV);
    };
  }, [yBounds, yScaleMode]);

  const yToPixel = useMemo(
    () => (y: number): number => {
      const norm = yToNorm(y);
      const dir = invertY ? norm : 1 - norm;
      return plotTop + dir * plotH;
    },
    [yToNorm, invertY, plotTop, plotH],
  );

  const pixelToY = useMemo(() => {
    const yMinV = yBounds.min;
    const yMaxV = yBounds.max;
    return (pxY: number): number => {
      const frac = Math.min(1, Math.max(0, (pxY - plotTop) / plotH));
      const norm = invertY ? frac : 1 - frac;
      if (yScaleMode === "log") {
        const l0 = Math.log10(Math.max(1e-12, yMinV));
        const l1 = Math.log10(Math.max(1e-12, yMaxV));
        return Math.pow(10, l0 + norm * (l1 - l0));
      }
      return yMinV + norm * (yMaxV - yMinV);
    };
  }, [yBounds, yScaleMode, invertY, plotTop, plotH]);

  // Cached offscreen raster bitmap rebuilt only when heatmap data or color bounds change
  const offscreenCanvas = useMemo<HTMLCanvasElement | null>(() => {
    const nTime = timesMs.length;
    const nY = yValues.length;
    if (nTime === 0 || nY === 0 || typeof document === "undefined") return null;

    const off = document.createElement("canvas");
    off.width = nTime;
    off.height = OFFSCREEN_RASTER_H;
    const octx = off.getContext("2d");
    if (!octx) return null;

    const yNorms = yValues.map((y) => yToNorm(y));
    const rowToYIdx = new Int32Array(OFFSCREEN_RASTER_H);
    for (let py = 0; py < OFFSCREEN_RASTER_H; py++) {
      const frac = py / Math.max(1, OFFSCREEN_RASTER_H - 1);
      const targetNorm = invertY ? frac : 1 - frac;
      let bestIdx = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let j = 0; j < nY; j++) {
        const d = Math.abs(yNorms[j] - targetNorm);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = j;
        }
      }
      rowToYIdx[py] = bestIdx;
    }

    const imgData = octx.createImageData(nTime, OFFSCREEN_RASTER_H);
    const pixels = imgData.data;
    const zRange = Math.max(1e-12, zHigh - zLow);

    for (let py = 0; py < OFFSCREEN_RASTER_H; py++) {
      const yIdx = rowToYIdx[py];
      const rowBase = py * nTime * 4;
      for (let tIdx = 0; tIdx < nTime; tIdx++) {
        const raw =
          matrixLayout === "time-by-y"
            ? (zMatrix[tIdx]?.[yIdx] ?? null)
            : (zMatrix[yIdx]?.[tIdx] ?? null);
        const offIdx = rowBase + tIdx * 4;
        if (raw === null || raw === undefined || !Number.isFinite(raw)) {
          pixels[offIdx + 3] = 0;
          continue;
        }
        const val = zTransform === "log10" ? (raw > 0 ? Math.log10(raw) : Number.NaN) : raw;
        if (!Number.isFinite(val)) {
          pixels[offIdx + 3] = 0;
          continue;
        }
        const norm = Math.min(1, Math.max(0, (val - zLow) / zRange));
        const cIdx = Math.round(norm * 255) * 3;
        pixels[offIdx] = lut[cIdx];
        pixels[offIdx + 1] = lut[cIdx + 1];
        pixels[offIdx + 2] = lut[cIdx + 2];
        pixels[offIdx + 3] = 255;
      }
    }

    octx.putImageData(imgData, 0, 0);
    return off;
  }, [timesMs.length, yValues, zMatrix, matrixLayout, zTransform, zLow, zHigh, lut, yToNorm, invertY]);

  // Fast canvas draw: blits cached offscreen raster + draws crisp vector overlays (no cursor dep!)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#090f1d";
    ctx.fillRect(plotLeft, plotTop, plotW, plotH);

    const nTime = timesMs.length;
    const nY = yValues.length;
    const [vStart, vEnd] = effectiveRange;
    const spanMs = Math.max(1, vEnd - vStart);
    const dtMs =
      nTime > 1 ? Math.max(1000, (timesMs[nTime - 1] - timesMs[0]) / (nTime - 1)) : 60_000;

    ctx.save();
    ctx.beginPath();
    ctx.rect(plotLeft, plotTop, plotW, plotH);
    ctx.clip();

    if (offscreenCanvas && nTime > 0 && nY > 0) {
      const srcStartMs = timesMs[0] - 0.5 * dtMs;
      const srcEndMs = timesMs[nTime - 1] + 0.5 * dtMs;
      const visStart = Math.max(vStart, srcStartMs);
      const visEnd = Math.min(vEnd, srcEndMs);
      if (visEnd > visStart) {
        const totalSrcMs = Math.max(1, srcEndMs - srcStartMs);
        const sx = ((visStart - srcStartMs) / totalSrcMs) * nTime;
        const sw = Math.max(0.01, ((visEnd - visStart) / totalSrcMs) * nTime);
        const dx = plotLeft + ((visStart - vStart) / spanMs) * plotW;
        const dw = ((visEnd - visStart) / spanMs) * plotW;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreenCanvas, sx, 0, sw, OFFSCREEN_RASTER_H, dx, plotTop, dw, plotH);
      }
    }

    // Optional band guide lines
    if (bandGuidesY) {
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      for (const guide of bandGuidesY) {
        const gy = yToPixel(guide.value);
        if (gy >= plotTop && gy <= plotBottom) {
          ctx.strokeStyle = guide.color ?? "rgba(248, 250, 252, 0.75)";
          ctx.beginPath();
          ctx.moveTo(plotLeft, gy);
          ctx.lineTo(plotRight, gy);
          ctx.stroke();
          ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
          ctx.fillRect(plotLeft + 4, gy - 12, 82, 13);
          ctx.fillStyle = "#e2e8f0";
          ctx.font = "10px 'JetBrains Mono', monospace";
          ctx.fillText(guide.label, plotLeft + 7, gy - 2);
        }
      }
      ctx.setLineDash([]);
    }

    // Optional COI shading and boundary curve closed per contiguous finite run
    if (coiPeriods && coiPeriods.length === nTime && nTime > 1) {
      const outsideEdgeY = invertY ? plotBottom : plotTop;
      ctx.fillStyle = "rgba(15, 23, 42, 0.58)";

      let runStart = -1;
      for (let tIdx = 0; tIdx <= nTime; tIdx++) {
        const p = tIdx < nTime ? coiPeriods[tIdx] : null;
        const ok = p !== null && p !== undefined && Number.isFinite(p);
        if (ok) {
          if (runStart === -1) runStart = tIdx;
        } else if (runStart !== -1) {
          const firstX = plotLeft + ((timesMs[runStart] - vStart) / spanMs) * plotW;
          const lastX = plotLeft + ((timesMs[tIdx - 1] - vStart) / spanMs) * plotW;
          ctx.beginPath();
          ctx.moveTo(firstX, outsideEdgeY);
          for (let k = runStart; k < tIdx; k++) {
            const pk = coiPeriods[k] as number;
            const xk = plotLeft + ((timesMs[k] - vStart) / spanMs) * plotW;
            const yk = Math.min(plotBottom, Math.max(plotTop, yToPixel(pk)));
            ctx.lineTo(xk, yk);
          }
          ctx.lineTo(lastX, outsideEdgeY);
          ctx.closePath();
          ctx.fill();
          runStart = -1;
        }
      }

      ctx.strokeStyle = "rgba(248, 250, 252, 0.92)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      let strokeOpen = false;
      for (let tIdx = 0; tIdx < nTime; tIdx++) {
        const p = coiPeriods[tIdx];
        if (p === null || p === undefined || !Number.isFinite(p)) {
          strokeOpen = false;
          continue;
        }
        const x = plotLeft + ((timesMs[tIdx] - vStart) / spanMs) * plotW;
        const y = Math.min(plotBottom, Math.max(plotTop, yToPixel(p)));
        if (!strokeOpen) {
          ctx.moveTo(x, y);
          strokeOpen = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Optional 95% Monte Carlo significance contours (geometric interpolation when yScaleMode='log')
    if (contourSegments.length > 0 && nTime > 1 && nY > 1) {
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const mapT = (tF: number) => {
        const i0 = Math.min(nTime - 1, Math.max(0, Math.floor(tF)));
        const i1 = Math.min(nTime - 1, i0 + 1);
        const frac = tF - i0;
        const tMs = timesMs[i0] + frac * (timesMs[i1] - timesMs[i0]);
        return plotLeft + ((tMs - vStart) / spanMs) * plotW;
      };
      const mapY = (sF: number) => {
        const j0 = Math.min(nY - 1, Math.max(0, Math.floor(sF)));
        const j1 = Math.min(nY - 1, j0 + 1);
        const frac = sF - j0;
        const y0 = yValues[j0];
        const y1 = yValues[j1];
        const yVal =
          yScaleMode === "log" && y0 > 0 && y1 > 0
            ? y0 * Math.pow(y1 / y0, frac)
            : y0 + frac * (y1 - y0);
        return yToPixel(yVal);
      };
      for (const [[t0, s0], [t1, s1]] of contourSegments) {
        ctx.moveTo(mapT(t0), mapY(s0));
        ctx.lineTo(mapT(t1), mapY(s1));
      }
      ctx.stroke();
    }

    // Optional phase arrows (0 deg -> right/in-phase, 90 deg -> up, 180 deg -> left)
    if (phaseVectors && nTime > 0 && nY > 0) {
      let firstVisIdx = 0;
      while (firstVisIdx < nTime && timesMs[firstVisIdx] < vStart) firstVisIdx++;
      let lastVisIdx = nTime - 1;
      while (lastVisIdx >= 0 && timesMs[lastVisIdx] > vEnd) lastVisIdx--;
      const visibleCount = Math.max(0, lastVisIdx - firstVisIdx + 1);
      const stepX = computeVisibleArrowStepX(visibleCount, phaseVectors.stepX);
      const stepY = phaseVectors.stepY ?? Math.max(2, Math.floor(nY / 14));
      const arrowLen = 7;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
      ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
      ctx.lineWidth = 1.2;

      const startT = Math.floor(firstVisIdx / stepX) * stepX;
      for (let tIdx = startT; tIdx <= lastVisIdx; tIdx += stepX) {
        if (tIdx < firstVisIdx) continue;
        const tMs = timesMs[tIdx];
        if (tMs < vStart || tMs > vEnd) continue;
        const cx = plotLeft + ((tMs - vStart) / spanMs) * plotW;
        for (let yIdx = 0; yIdx < nY; yIdx += stepY) {
          if (phaseVectors.sigMask && !phaseVectors.sigMask[yIdx]?.[tIdx]) continue;
          const deg = phaseVectors.phaseDeg[yIdx]?.[tIdx];
          if (deg === null || deg === undefined || !Number.isFinite(deg)) continue;
          const cy = yToPixel(yValues[yIdx]);
          const rad = (deg * Math.PI) / 180;
          const dx = Math.cos(rad) * arrowLen;
          const dy = -Math.sin(rad) * arrowLen;
          const xStart = cx - dx * 0.5;
          const yStart = cy - dy * 0.5;
          const xEnd = cx + dx * 0.5;
          const yEnd = cy + dy * 0.5;

          ctx.beginPath();
          ctx.moveTo(xStart, yStart);
          ctx.lineTo(xEnd, yEnd);
          ctx.stroke();

          const angle = Math.atan2(yEnd - yStart, xEnd - xStart);
          ctx.beginPath();
          ctx.moveTo(xEnd, yEnd);
          ctx.lineTo(
            xEnd - 3.2 * Math.cos(angle - Math.PI / 6),
            yEnd - 3.2 * Math.sin(angle - Math.PI / 6),
          );
          ctx.lineTo(
            xEnd - 3.2 * Math.cos(angle + Math.PI / 6),
            yEnd - 3.2 * Math.sin(angle + Math.PI / 6),
          );
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    ctx.restore();

    // Axes frame
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotLeft, plotTop, plotW, plotH);

    // Y-axis ticks & label
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTickCount = 5;
    for (let i = 0; i < yTickCount; i++) {
      const frac = i / (yTickCount - 1);
      const py = plotTop + frac * plotH;
      const val = pixelToY(py);
      ctx.beginPath();
      ctx.moveTo(plotLeft - 4, py);
      ctx.lineTo(plotLeft, py);
      ctx.stroke();
      ctx.fillText(yFormat(val), plotLeft - 7, py);
    }

    ctx.save();
    ctx.translate(13, plotTop + plotH * 0.5);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px 'Inter', sans-serif";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // X-axis ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px 'JetBrains Mono', monospace";
    const xTickCount = Math.max(2, Math.min(7, Math.floor(plotW / 115)));
    for (let i = 0; i <= xTickCount; i++) {
      const frac = i / xTickCount;
      const px = plotLeft + frac * plotW;
      const tMs = vStart + frac * spanMs;
      ctx.beginPath();
      ctx.moveTo(px, plotBottom);
      ctx.lineTo(px, plotBottom + 4);
      ctx.stroke();
      ctx.fillText(formatUtcShort(tMs), px, plotBottom + 7);
    }

    // Colorbar inside right 64px margin
    const cbX = plotRight + 10;
    const cbW = 10;
    for (let row = 0; row < plotH; row++) {
      const norm = 1 - row / Math.max(1, plotH - 1);
      const cIdx = Math.round(norm * 255);
      ctx.fillStyle = `rgb(${lut[cIdx * 3]}, ${lut[cIdx * 3 + 1]}, ${lut[cIdx * 3 + 2]})`;
      ctx.fillRect(cbX, plotTop + row, cbW, 1.5);
    }
    ctx.strokeStyle = "#334155";
    ctx.strokeRect(cbX, plotTop, cbW, plotH);

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText(zHigh.toFixed(1), cbX + cbW + 4, plotTop + 4);
    ctx.fillText(((zHigh + zLow) * 0.5).toFixed(1), cbX + cbW + 4, plotTop + plotH * 0.5);
    ctx.fillText(zLow.toFixed(1), cbX + cbW + 4, plotBottom - 4);
    if (zUnit) {
      ctx.fillText(zUnit, cbX, plotBottom + 10);
    }
  }, [
    width,
    height,
    plotLeft,
    plotRight,
    plotTop,
    plotBottom,
    plotW,
    plotH,
    timesMs,
    yValues,
    offscreenCanvas,
    zMatrix,
    matrixLayout,
    zTransform,
    zLow,
    zHigh,
    zUnit,
    lut,
    effectiveRange,
    yScaleMode,
    yToPixel,
    pixelToY,
    yFormat,
    yLabel,
    invertY,
    bandGuidesY,
    coiPeriods,
    contourSegments,
    phaseVectors,
  ]);

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    dragRef.current = { startX: e.clientX, origRange: effectiveRange };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (dragRef.current) {
      const dx = e.clientX - dragRef.current.startX;
      const [oS, oE] = dragRef.current.origRange;
      const span = oE - oS;
      const shiftMs = (-dx / Math.max(1, plotW)) * span;
      let nextS = oS + shiftMs;
      let nextE = oE + shiftMs;
      if (boundsMs) {
        if (nextS < boundsMs[0]) {
          nextE += boundsMs[0] - nextS;
          nextS = boundsMs[0];
        }
        if (nextE > boundsMs[1]) {
          nextS -= nextE - boundsMs[1];
          nextE = boundsMs[1];
        }
      }
      onViewRangeChange([nextS, nextE]);
      return;
    }

    if (x < plotLeft || x > plotRight || y < plotTop || y > plotBottom || timesMs.length === 0) {
      setHoverInfo(null);
      return;
    }

    const [vStart, vEnd] = effectiveRange;
    const utcMs = vStart + ((x - plotLeft) / plotW) * (vEnd - vStart);
    if (onCursorChange) {
      const now = performance.now();
      if (now - lastCursorEmitRef.current >= 45) {
        lastCursorEmitRef.current = now;
        onCursorChange(utcMs);
      }
    }

    const yVal = pixelToY(y);
    let nearestT = 0;
    let minTDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < timesMs.length; i++) {
      const d = Math.abs(timesMs[i] - utcMs);
      if (d < minTDiff) {
        minTDiff = d;
        nearestT = i;
      }
    }
    const nearestY = findNearestYIndex(yValues, yVal, yScaleMode);

    const raw =
      matrixLayout === "time-by-y"
        ? (zMatrix[nearestT]?.[nearestY] ?? null)
        : (zMatrix[nearestY]?.[nearestT] ?? null);
    const zVal =
      raw === null || raw === undefined || !Number.isFinite(raw)
        ? null
        : zTransform === "log10"
          ? raw > 0
            ? Math.log10(raw)
            : null
          : raw;

    setHoverInfo({
      xPx: Math.min(plotRight - 140, x + 12),
      yPx: Math.max(plotTop + 4, y - 28),
      utcMs,
      yVal: yValues[nearestY] ?? yVal,
      zVal,
    });
  };

  const handleMouseUp = () => {
    dragRef.current = null;
  };

  const explicitCursorFrac =
    cursorUtcMs !== undefined ? computeCursorOverlayPercent(cursorUtcMs, effectiveRange) : null;

  return (
    <div
      ref={setWrapEl}
      className="heatmap-wrapper"
      style={{ height }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        dragRef.current = null;
        setHoverInfo(null);
      }}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
      {cursorUtcMs !== undefined ? (
        explicitCursorFrac !== null && (
          <div
            style={{
              position: "absolute",
              top: plotTop,
              height: plotH,
              left: `calc(${CHART_GRID_LEFT}px + ${explicitCursorFrac} * (100% - ${CHART_GRID_LEFT + CHART_GRID_RIGHT}px))`,
              width: 1,
              borderLeft: "1px dashed rgba(56, 189, 248, 0.9)",
              pointerEvents: "none",
            }}
          />
        )
      ) : (
        <CursorOverlayLine
          viewRangeMs={effectiveRange}
          top={plotTop}
          height={plotH}
          borderColor="rgba(56, 189, 248, 0.9)"
        />
      )}
      {hoverInfo && (
        <div className="heatmap-tooltip mono" style={{ left: hoverInfo.xPx, top: hoverInfo.yPx }}>
          <div>{formatUtcFull(hoverInfo.utcMs)}</div>
          <div>
            {yFormat(hoverInfo.yVal)} ·{" "}
            {hoverInfo.zVal !== null ? `${hoverInfo.zVal.toFixed(3)} ${zUnit}` : "нет данных"}
          </div>
        </div>
      )}
    </div>
  );
};
