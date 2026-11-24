import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export const CHART_GRID_LEFT = 68;
export const CHART_GRID_RIGHT = 64;
export const CHART_SYNC_GROUP = "geomag-sync";

export interface UseSyncedEChartArgs {
  option: echarts.EChartsCoreOption | null;
  viewRangeMs: [number, number] | null;
  boundsMs: [number, number] | null;
  onViewRangeChange: (range: [number, number]) => void;
  onCursorChange?: (utcMs: number) => void;
}

export function useSyncedEChart({
  option,
  viewRangeMs,
  boundsMs,
  onViewRangeChange,
  onCursorChange,
}: UseSyncedEChartArgs) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const isSyncingRef = useRef(false);

  const viewRangeRef = useRef(viewRangeMs);
  viewRangeRef.current = viewRangeMs;
  const boundsRef = useRef(boundsMs);
  boundsRef.current = boundsMs;
  const onRangeRef = useRef(onViewRangeChange);
  onRangeRef.current = onViewRangeChange;
  const onCursorRef = useRef(onCursorChange);
  onCursorRef.current = onCursorChange;
  const lastCursorEmitRef = useRef(0);

  // Initialize ECharts instance + ResizeObserver + group sync
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = echarts.init(el, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    chart.group = CHART_SYNC_GROUP;
    echarts.connect(CHART_SYNC_GROUP);

    const observer = new ResizeObserver(() => {
      chart.resize();
    });
    observer.observe(el);

    const handleDataZoom = () => {
      if (isSyncingRef.current) return;
      const opt = chart.getOption() as {
        dataZoom?: Array<{
          startValue?: number;
          endValue?: number;
          start?: number;
          end?: number;
        }>;
      };
      const dz = opt.dataZoom?.[0];
      if (!dz) return;

      let startVal = typeof dz.startValue === "number" ? dz.startValue : Number.NaN;
      let endVal = typeof dz.endValue === "number" ? dz.endValue : Number.NaN;

      if ((!Number.isFinite(startVal) || !Number.isFinite(endVal)) && boundsRef.current) {
        const [b0, b1] = boundsRef.current;
        const p0 = typeof dz.start === "number" ? dz.start / 100 : 0;
        const p1 = typeof dz.end === "number" ? dz.end / 100 : 1;
        startVal = b0 + p0 * (b1 - b0);
        endVal = b0 + p1 * (b1 - b0);
      }

      if (!Number.isFinite(startVal) || !Number.isFinite(endVal) || endVal <= startVal) {
        return;
      }

      const cur = viewRangeRef.current;
      if (!cur || Math.abs(startVal - cur[0]) > 1500 || Math.abs(endVal - cur[1]) > 1500) {
        onRangeRef.current([startVal, endVal]);
      }
    };

    const handleAxisPointer = (event: unknown) => {
      if (!onCursorRef.current) return;
      const ev = event as { axesInfo?: Array<{ axisDim?: string; value?: unknown }> };
      const info = ev.axesInfo?.find((a) => a.axisDim === "x" || a.axisDim === undefined);
      const raw = info?.value;
      const utcMs = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(utcMs) || utcMs < 1e11) return;
      const now = performance.now();
      if (now - lastCursorEmitRef.current < 45) return;
      lastCursorEmitRef.current = now;
      onCursorRef.current(utcMs);
    };

    chart.on("datazoom", handleDataZoom);
    chart.on("updateAxisPointer", handleAxisPointer);

    return () => {
      chart.off("datazoom", handleDataZoom);
      chart.off("updateAxisPointer", handleAxisPointer);
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  // Apply option updates
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !option) return;

    isSyncingRef.current = true;
    try {
      chart.setOption(option, { notMerge: true, lazyUpdate: false });
      const vr = viewRangeRef.current;
      if (vr) {
        chart.dispatchAction({
          type: "dataZoom",
          dataZoomIndex: 0,
          startValue: vr[0],
          endValue: vr[1],
        });
      }
    } finally {
      queueMicrotask(() => {
        isSyncingRef.current = false;
      });
    }
  }, [option]);

  // Apply external viewRangeMs updates
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !viewRangeMs || !option) return;

    const curOpt = chart.getOption() as {
      dataZoom?: Array<{ startValue?: number; endValue?: number }>;
    };
    const dz = curOpt.dataZoom?.[0];
    if (
      dz &&
      typeof dz.startValue === "number" &&
      typeof dz.endValue === "number" &&
      Math.abs(dz.startValue - viewRangeMs[0]) <= 1500 &&
      Math.abs(dz.endValue - viewRangeMs[1]) <= 1500
    ) {
      return;
    }

    isSyncingRef.current = true;
    try {
      chart.dispatchAction({
        type: "dataZoom",
        dataZoomIndex: 0,
        startValue: viewRangeMs[0],
        endValue: viewRangeMs[1],
      });
    } finally {
      queueMicrotask(() => {
        isSyncingRef.current = false;
      });
    }
  }, [viewRangeMs, option]);

  return { containerRef, chartRef };
}

export function computeCursorOverlayPercent(
  cursorUtcMs: number | null,
  viewRangeMs: [number, number] | null,
): number | null {
  if (cursorUtcMs === null || !viewRangeMs) return null;
  const span = viewRangeMs[1] - viewRangeMs[0];
  if (span <= 0) return null;
  const frac = (cursorUtcMs - viewRangeMs[0]) / span;
  if (frac < 0 || frac > 1) return null;
  return frac;
}
