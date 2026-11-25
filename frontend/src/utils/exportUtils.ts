/**
 * Publication-ready export subsystem for GeoMag Analyzer:
 * - Standalone vector SVG (Word / Inkscape / Illustrator compatible, Light & Dark print themes)
 * - True 300 DPI PNG with standard PNG pHYs physical-pixel-density chunk + IEEE CRC-32
 * - RFC 4180 UTF-8 CSV with scientific parameter metadata headers
 * - Windows-safe filename sanitization
 */

import * as echarts from "echarts";
import type { ExportTheme } from "../store/analysisReducer";
import { buildColorLut, extractSignificanceContours, type ColormapName } from "./contours";
import { formatUtcShort } from "./formatters";
import type { ArcCoherenceMetrics, StokesEllipseGeometry } from "./stokesEllipse";
import { coherenceArcColor, coherenceArcStrokeWidth } from "./stokesEllipse";
import type { PlacedLabel } from "./mapProjection";

// ---------------------------------------------------------------------------
// Filename & XML / CSV Helpers
// ---------------------------------------------------------------------------

export function buildExportFilename(
  parts: ReadonlyArray<string | number | null | undefined>,
  extension: string,
): string {
  const cleanedParts = parts
    .filter((p) => p !== null && p !== undefined && String(p).trim() !== "")
    .map((p) =>
      String(p)
        .trim()
        .replace(/[<>:"/\\|?*\s]+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, ""),
    )
    .filter((p) => p.length > 0);
  const cleanExt = extension.replace(/^\.+/, "").replace(/[^a-zA-Z0-9]/g, "");
  const stem = cleanedParts.length > 0 ? cleanedParts.join("_") : "geomag_export";
  return `${stem}.${cleanExt || "txt"}`;
}

function escapeCsvCell(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "";
  }
  const str = typeof value === "boolean" ? (value ? "true" : "false") : String(value);
  if (/[",\r\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function formatCsvWithMetadata(
  metadata: Record<string, string | number | boolean | null | undefined>,
  columns: ReadonlyArray<string>,
  rows: ReadonlyArray<ReadonlyArray<string | number | boolean | null | undefined>>,
): string {
  const lines: string[] = [];
  for (const [key, val] of Object.entries(metadata)) {
    const cleanVal =
      val === null || val === undefined
        ? ""
        : String(val).replace(/[\r\n]+/g, " ");
    lines.push(`# ${key}: ${cleanVal}`);
  }
  lines.push(columns.map((c) => escapeCsvCell(c)).join(","));
  for (const row of rows) {
    lines.push(row.map((cell) => escapeCsvCell(cell)).join(","));
  }
  return lines.join("\n") + "\n";
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

const CSS_VAR_FALLBACKS: Record<string, string> = {
  "--bg-canvas": "#0b1120",
  "--bg-surface": "#111827",
  "--bg-elevated": "#1e293b",
  "--bg-input": "#0f172a",
  "--border-subtle": "#1f2937",
  "--border-strong": "#334155",
  "--text-primary": "#f1f5f9",
  "--text-secondary": "#94a3b8",
  "--text-muted": "#64748b",
  "--accent-primary": "#38bdf8",
  "--accent-strong": "#0ea5e9",
  "--ok": "#10b981",
  "--warn": "#f59e0b",
  "--danger": "#ef4444",
  "--font-sans": "Inter, Segoe UI, Arial, sans-serif",
  "--font-mono": "JetBrains Mono, Consolas, monospace",
};

/**
 * Convert CSS3 rgba(r, g, b, a) in SVG fill/stroke attributes into SVG 1.1
 * paint="rgb(r,g,b)" + fill-opacity/stroke-opacity so Microsoft Word, LibreOffice,
 * and older Illustrator SVG 1.1 parsers never fall back to opaque black.
 */
function normalizeSvgRgbaAttributes(svg: string): string {
  return svg.replace(/<[a-zA-Z][^>]*>/g, (tag) => {
    let nextTag = tag;
    for (const attr of ["fill", "stroke"] as const) {
      const rgbaRe = new RegExp(
        `\\b${attr}="rgba\\(\\s*([\\d.]+)\\s*,\\s*([\\d.]+)\\s*,\\s*([\\d.]+)\\s*,\\s*([\\d.]+)\\s*\\)"`,
        "i",
      );
      const match = nextTag.match(rgbaRe);
      if (!match) continue;
      const r = Math.min(255, Math.max(0, Math.round(Number(match[1]))));
      const g = Math.min(255, Math.max(0, Math.round(Number(match[2]))));
      const b = Math.min(255, Math.max(0, Math.round(Number(match[3]))));
      const a = Math.min(1, Math.max(0, Number(match[4])));
      if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b) || !Number.isFinite(a)) {
        continue;
      }
      const opAttr = `${attr}-opacity`;
      const opRe = new RegExp(`\\b${opAttr}="([\\d.]+)"`, "i");
      const existingOp = nextTag.match(opRe);
      if (existingOp) {
        const combined = Math.min(1, Math.max(0, Number(existingOp[1]) * a));
        const opStr = Number.isFinite(combined) ? Number(combined.toFixed(4)).toString() : "1";
        nextTag = nextTag.replace(rgbaRe, `${attr}="rgb(${r},${g},${b})"`);
        nextTag = nextTag.replace(opRe, `${opAttr}="${opStr}"`);
      } else {
        const opStr = Number(a.toFixed(4)).toString();
        nextTag = nextTag.replace(rgbaRe, `${attr}="rgb(${r},${g},${b})" ${opAttr}="${opStr}"`);
      }
    }
    return nextTag;
  });
}

/**
 * Normalize an SVG string so it opens cleanly in MS Word, LibreOffice, Inkscape,
 * Illustrator, and browsers without external CSS variables or missing namespaces.
 */
export function sanitizeStandaloneSvg(
  svgString: string,
  opts: {
    title?: string;
    desc?: string;
    backgroundColor?: string;
    width?: number;
    height?: number;
  } = {},
): string {
  let out = svgString.trim();

  // Resolve any CSS var(--...) tokens to literal hex/font fallbacks
  out = out.replace(/var\((--[a-zA-Z0-9_-]+)(?:,\s*([^)]+))?\)/g, (_m, varName, fb) => {
    return CSS_VAR_FALLBACKS[String(varName)] ?? (fb ? String(fb).trim() : "#334155");
  });

  // Ensure SVG 1.1 compatibility by splitting rgba(...) fill/stroke attributes
  out = normalizeSvgRgbaAttributes(out);

  // Strip any existing xml declaration before rebuilding uniformly
  out = out.replace(/^<\?xml[^>]*\?>\s*/i, "");

  const svgOpenMatch = out.match(/^<svg\b([^>]*)>/i);
  if (svgOpenMatch) {
    let attrs = svgOpenMatch[1];
    if (!/\bxmlns=/.test(attrs)) {
      attrs += ' xmlns="http://www.w3.org/2000/svg"';
    }
    if (!/\bxmlns:xlink=/.test(attrs)) {
      attrs += ' xmlns:xlink="http://www.w3.org/1999/xlink"';
    }
    if (opts.width && !/\bwidth=/.test(attrs)) {
      attrs += ` width="${opts.width}"`;
    }
    if (opts.height && !/\bheight=/.test(attrs)) {
      attrs += ` height="${opts.height}"`;
    }
    if (opts.width && opts.height && !/\bviewBox=/.test(attrs)) {
      attrs += ` viewBox="0 0 ${opts.width} ${opts.height}"`;
    }
    const injectedParts: string[] = [];
    if (opts.title && !/<title\b/i.test(out)) {
      injectedParts.push(`<title>${escapeXml(opts.title)}</title>`);
    }
    if (opts.desc && !/<desc\b/i.test(out)) {
      injectedParts.push(`<desc>${escapeXml(opts.desc)}</desc>`);
    }
    if (
      opts.backgroundColor &&
      !/data-export-bg/.test(out) &&
      !/<rect[^>]*width="(?:100%|\d+)"[^>]*height="(?:100%|\d+)"/.test(out.slice(0, 420))
    ) {
      injectedParts.push(
        `<rect data-export-bg="1" x="0" y="0" width="100%" height="100%" fill="${escapeXml(opts.backgroundColor)}"/>`,
      );
    }
    const replacement = `<svg${attrs}>${injectedParts.join("")}`;
    out = replacement + out.slice(svgOpenMatch[0].length);
  }

  return `<?xml version="1.0" encoding="UTF-8"?>\n${out}\n`;
}

// ---------------------------------------------------------------------------
// 300 DPI PNG Chunk Injector (IEEE CRC-32 + pHYs) & SVG Rasterizer
// ---------------------------------------------------------------------------

const CRC32_TABLE = (() => {
  const tbl = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    tbl[i] = c >>> 0;
  }
  return tbl;
})();

export function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

const PNG_SIG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

/**
 * Insert or replace the standard PNG `pHYs` chunk immediately after `IHDR`
 * with `round(dpi / 0.0254)` pixels/meter (11811 px/m for 300 DPI).
 */
export function injectPngDpiChunk(pngBytes: Uint8Array, dpi = 300): Uint8Array {
  if (pngBytes.length < 33) {
    throw new Error("Invalid PNG byte buffer (too short).");
  }
  for (let i = 0; i < PNG_SIG.length; i++) {
    if (pngBytes[i] !== PNG_SIG[i]) {
      throw new Error("Invalid PNG signature.");
    }
  }

  const ppm = Math.round(dpi / 0.0254);
  const physChunk = new Uint8Array(21);
  const pView = new DataView(physChunk.buffer);
  pView.setUint32(0, 9, false); // payload length = 9 bytes
  physChunk[4] = 0x70; // 'p'
  physChunk[5] = 0x48; // 'H'
  physChunk[6] = 0x59; // 'Y'
  physChunk[7] = 0x73; // 's'
  pView.setUint32(8, ppm, false);
  pView.setUint32(12, ppm, false);
  physChunk[16] = 1; // unit = meter
  const physCrc = crc32(physChunk.subarray(4, 17));
  pView.setUint32(17, physCrc, false);

  const view = new DataView(pngBytes.buffer, pngBytes.byteOffset, pngBytes.byteLength);
  const pieces: Uint8Array[] = [pngBytes.subarray(0, 8)];
  let offset = 8;
  let inserted = false;

  while (offset + 12 <= pngBytes.length) {
    const chunkLen = view.getUint32(offset, false);
    const totalLen = 12 + chunkLen;
    if (offset + totalLen > pngBytes.length) break;
    const type = String.fromCharCode(
      pngBytes[offset + 4],
      pngBytes[offset + 5],
      pngBytes[offset + 6],
      pngBytes[offset + 7],
    );
    const slice = pngBytes.subarray(offset, offset + totalLen);
    if (type !== "pHYs") {
      pieces.push(slice);
      if (type === "IHDR" && !inserted) {
        pieces.push(physChunk);
        inserted = true;
      }
    }
    offset += totalLen;
    if (type === "IEND") break;
  }

  if (!inserted) {
    pieces.splice(1, 0, physChunk);
  }

  const totalOut = pieces.reduce((acc, p) => acc + p.length, 0);
  const out = new Uint8Array(totalOut);
  let pos = 0;
  for (const p of pieces) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

export async function rasterizeSvgToPng300Dpi(
  svgString: string,
  widthCssPx: number,
  heightCssPx: number,
  dpi = 300,
): Promise<Blob> {
  const scale = dpi / 96;
  const outW = Math.max(1, Math.round(widthCssPx * scale));
  const outH = Math.max(1, Math.round(heightCssPx * scale));

  const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("Failed to rasterize SVG to PNG."));
      el.src = url;
    });

    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2D Canvas context unavailable for PNG export.");
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.drawImage(img, 0, 0, widthCssPx, heightCssPx);

    const rawBlob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Canvas toBlob returned null."))),
        "image/png",
      );
    });
    const rawBuf = new Uint8Array(await rawBlob.arrayBuffer());
    const withDpi = injectPngDpiChunk(rawBuf, dpi);
    return new Blob([withDpi.buffer as ArrayBuffer], { type: "image/png" });
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function downloadTextFile(
  filename: string,
  content: string,
  mimeType = "text/plain;charset=utf-8",
): void {
  const blob = new Blob([content], { type: mimeType });
  downloadBlob(filename, blob);
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

// ---------------------------------------------------------------------------
// ECharts True Vector SVG Exporter (SSR Mode, Light/Dark Print Themes)
// ---------------------------------------------------------------------------

const LIGHT_COLOR_MAP: Record<string, string> = {
  "#38bdf8": "#0284c7",
  "#0ea5e9": "#0369a1",
  "#a78bfa": "#6d28d9",
  "#34d399": "#059669",
  "#2dd4bf": "#0f766e",
  "#f43f5e": "#dc2626",
  "#fb7185": "#be123c",
  "#f59e0b": "#b45309",
  "#fbbf24": "#b45309",
  "#ef4444": "#dc2626",
  "#f87171": "#b91c1c",
  "#94a3b8": "#475569",
  "#cbd5e1": "#334155",
  "#f1f5f9": "#0f172a",
  "#fda4af": "#9f1239",
  "#e11d48": "#9f1239",
};

function mapPrintColor(color: unknown, theme: ExportTheme): unknown {
  if (theme !== "light" || typeof color !== "string") return color;
  const lower = color.trim().toLowerCase();
  if (LIGHT_COLOR_MAP[lower]) return LIGHT_COLOR_MAP[lower];
  if (lower.startsWith("rgba(148,163,184") || lower.startsWith("rgba(148, 163, 184")) {
    return "rgba(100, 116, 139, 0.55)";
  }
  if (lower.startsWith("rgba(56, 189, 248") || lower.startsWith("rgba(56,189,248")) {
    return "rgba(2, 132, 199, 0.55)";
  }
  if (lower.startsWith("rgba(167, 139, 250") || lower.startsWith("rgba(167,139,250")) {
    return "rgba(109, 40, 217, 0.55)";
  }
  return color;
}

function clonePlain(val: unknown): unknown {
  if (val === null || val === undefined || typeof val !== "object") {
    return val;
  }
  if (Array.isArray(val)) {
    return val.map((item) => clonePlain(item));
  }
  const src = val as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(src)) {
    if (typeof v === "function") continue;
    out[k] = clonePlain(v);
  }
  return out;
}

export function renderEChartsToSvg(
  baseOption: echarts.EChartsCoreOption,
  opts: {
    width: number;
    height: number;
    theme: ExportTheme;
    viewRangeMs?: [number, number] | null;
    title?: string;
    desc?: string;
  },
): string {
  const { width, height, theme, viewRangeMs, title, desc } = opts;
  const opt = clonePlain(baseOption) as Record<string, unknown>;

  opt.animation = false;
  opt.useUTC = true;
  opt.backgroundColor = theme === "light" ? "#ffffff" : "#0b1120";
  delete opt.tooltip;
  delete opt.dataZoom;
  delete opt.axisPointer;

  const isLight = theme === "light";
  const axisLineColor = isLight ? "#475569" : "#334155";
  const splitLineColor = isLight ? "#e2e8f0" : "rgba(51,65,85,0.32)";
  const textPrimary = isLight ? "#0f172a" : "#e2e8f0";

  const restyleAxis = (axisRaw: unknown, isX: boolean) => {
    const arr = Array.isArray(axisRaw) ? axisRaw : axisRaw ? [axisRaw] : [];
    for (const ax of arr) {
      if (!ax || typeof ax !== "object") continue;
      const a = ax as Record<string, unknown>;
      if (isX && a.type === "time" && viewRangeMs) {
        a.min = viewRangeMs[0];
        a.max = viewRangeMs[1];
      }
      if (a.axisLine && typeof a.axisLine === "object") {
        const al = a.axisLine as Record<string, unknown>;
        al.lineStyle = { ...(al.lineStyle as object), color: axisLineColor };
      } else {
        a.axisLine = { lineStyle: { color: axisLineColor } };
      }
      if (a.axisLabel && typeof a.axisLabel === "object") {
        const lbl = a.axisLabel as Record<string, unknown>;
        lbl.color = isX ? textPrimary : mapPrintColor(lbl.color ?? textPrimary, theme);
      }
      if (a.nameTextStyle && typeof a.nameTextStyle === "object") {
        const nts = a.nameTextStyle as Record<string, unknown>;
        nts.color = isX ? textPrimary : mapPrintColor(nts.color ?? textPrimary, theme);
      }
      if (a.splitLine && typeof a.splitLine === "object") {
        const sl = a.splitLine as Record<string, unknown>;
        if (sl.show !== false) {
          sl.lineStyle = { ...(sl.lineStyle as object), color: splitLineColor };
        }
      }
    }
  };

  restyleAxis(opt.xAxis, true);
  restyleAxis(opt.yAxis, false);

  if (opt.legend && typeof opt.legend === "object") {
    const leg = opt.legend as Record<string, unknown>;
    leg.textStyle = { ...(leg.textStyle as object), color: textPrimary };
  }

  if (Array.isArray(opt.series)) {
    for (const sRaw of opt.series) {
      if (!sRaw || typeof sRaw !== "object") continue;
      const s = sRaw as Record<string, unknown>;
      if (s.lineStyle && typeof s.lineStyle === "object") {
        const ls = s.lineStyle as Record<string, unknown>;
        ls.color = mapPrintColor(ls.color, theme);
      }
      if (s.itemStyle && typeof s.itemStyle === "object") {
        const it = s.itemStyle as Record<string, unknown>;
        it.color = mapPrintColor(it.color, theme);
      }
    }
  }

  const chart = echarts.init(null, undefined, {
    renderer: "svg",
    ssr: true,
    width,
    height,
  });
  try {
    chart.setOption(opt as echarts.EChartsCoreOption);
    const rawSvg = chart.renderToSVGString();
    return sanitizeStandaloneSvg(rawSvg, {
      title,
      desc,
      backgroundColor: isLight ? "#ffffff" : "#0b1120",
      width,
      height,
    });
  } finally {
    chart.dispose();
  }
}

// ---------------------------------------------------------------------------
// 2D Heatmap Hybrid Vector SVG Exporter (Spectrogram / Welch 2D / XWT / WTC)
// ---------------------------------------------------------------------------

export interface HeatmapExportSpec {
  title: string;
  subtitle?: string;
  theme: ExportTheme;
  width?: number;
  height?: number;
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
  bandGuidesY?: Array<{ value: number; label: string; color?: string }>;
  coiPeriods?: (number | null)[];
  contourGrid?: { sigGrid: boolean[][]; insideCoi?: boolean[][] } | null;
  phaseVectors?: {
    phaseDeg: (number | null)[][];
    sigMask?: boolean[][];
    stepX?: number;
    stepY?: number;
  } | null;
}

function defaultHeatmapYFormat(y: number): string {
  if (y < 0.05) return `${(y * 1000).toFixed(2)} mHz`;
  if (y >= 60) return `${Math.round(y)} s`;
  return y.toFixed(2);
}

function rasterMatrixToPngDataUrl(
  timesMs: number[],
  yValues: number[],
  zMatrix: (number | null)[][],
  matrixLayout: "time-by-y" | "y-by-time",
  zTransform: "linear" | "log10",
  zLow: number,
  zHigh: number,
  lut: Uint8Array,
  yToNorm: (y: number) => number,
  invertY: boolean,
  vStart: number,
  vEnd: number,
): { dataUrl: string | null; xStartFrac: number; xWidthFrac: number } {
  const nTime = timesMs.length;
  const nY = yValues.length;
  if (nTime === 0 || nY === 0) {
    return { dataUrl: null, xStartFrac: 0, xWidthFrac: 1 };
  }

  const dtMs =
    nTime > 1 ? Math.max(1000, (timesMs[nTime - 1] - timesMs[0]) / (nTime - 1)) : 60_000;
  const srcStartMs = timesMs[0] - 0.5 * dtMs;
  const srcEndMs = timesMs[nTime - 1] + 0.5 * dtMs;
  const visStart = Math.max(vStart, srcStartMs);
  const visEnd = Math.min(vEnd, srcEndMs);
  const spanMs = Math.max(1, vEnd - vStart);

  if (typeof document === "undefined" || visEnd <= visStart) {
    return { dataUrl: null, xStartFrac: 0, xWidthFrac: 0 };
  }

  let firstIdx = 0;
  while (firstIdx < nTime && timesMs[firstIdx] + 0.5 * dtMs <= visStart) firstIdx++;
  let lastIdx = nTime - 1;
  while (lastIdx >= firstIdx && timesMs[lastIdx] - 0.5 * dtMs >= visEnd) lastIdx--;
  if (firstIdx > lastIdx) {
    return { dataUrl: null, xStartFrac: 0, xWidthFrac: 0 };
  }

  const subStartMs = timesMs[firstIdx] - 0.5 * dtMs;
  const subEndMs = timesMs[lastIdx] + 0.5 * dtMs;
  const xStartFrac = (subStartMs - vStart) / spanMs;
  const xWidthFrac = Math.max(0, (subEndMs - subStartMs) / spanMs);
  const subW = lastIdx - firstIdx + 1;
  const subH = 220;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = subW;
    canvas.height = subH;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return { dataUrl: null, xStartFrac, xWidthFrac };
    }

    const yNorms = yValues.map((y) => yToNorm(y));
    const rowToY = new Int32Array(subH);
    for (let py = 0; py < subH; py++) {
      const frac = py / Math.max(1, subH - 1);
      const targetNorm = invertY ? frac : 1 - frac;
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let j = 0; j < nY; j++) {
        const d = Math.abs(yNorms[j] - targetNorm);
        if (d < bestDist) {
          bestDist = d;
          best = j;
        }
      }
      rowToY[py] = best;
    }

    const img = ctx.createImageData(subW, subH);
    const px = img.data;
    const zRange = Math.max(1e-12, zHigh - zLow);

    for (let py = 0; py < subH; py++) {
      const yIdx = rowToY[py];
      const base = py * subW * 4;
      for (let dx = 0; dx < subW; dx++) {
        const tIdx = firstIdx + dx;
        const raw =
          matrixLayout === "time-by-y"
            ? (zMatrix[tIdx]?.[yIdx] ?? null)
            : (zMatrix[yIdx]?.[tIdx] ?? null);
        const off = base + dx * 4;
        if (raw === null || raw === undefined || !Number.isFinite(raw)) {
          px[off + 3] = 0;
          continue;
        }
        const val = zTransform === "log10" ? (raw > 0 ? Math.log10(raw) : Number.NaN) : raw;
        if (!Number.isFinite(val)) {
          px[off + 3] = 0;
          continue;
        }
        const norm = Math.min(1, Math.max(0, (val - zLow) / zRange));
        const cIdx = Math.round(norm * 255) * 3;
        px[off] = lut[cIdx];
        px[off + 1] = lut[cIdx + 1];
        px[off + 2] = lut[cIdx + 2];
        px[off + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return {
      dataUrl: canvas.toDataURL("image/png"),
      xStartFrac,
      xWidthFrac,
    };
  } catch {
    return { dataUrl: null, xStartFrac, xWidthFrac };
  }
}

export function buildHeatmapPublicationSvg(spec: HeatmapExportSpec): string {
  const width = spec.width ?? 880;
  const height = spec.height ?? 340;
  const isLight = spec.theme === "light";
  const bgColor = isLight ? "#ffffff" : "#0b1120";
  const plotBg = isLight ? "#f8fafc" : "#090f1d";
  const fgColor = isLight ? "#0f172a" : "#e2e8f0";
  const mutedColor = isLight ? "#475569" : "#94a3b8";
  const frameColor = isLight ? "#475569" : "#334155";
  const contourColor = isLight ? "#0f172a" : "#f8fafc";
  const coiFillColor = "#0f172a";
  const coiFillOpacity = isLight ? "0.25" : "0.58";
  const coiStroke = isLight ? "#334155" : "#f8fafc";
  const arrowColor = isLight ? "#0f172a" : "#ffffff";

  const plotLeft = 78;
  const plotRight = width - 74;
  const plotTop = spec.subtitle ? 44 : 30;
  const plotBottom = height - 38;
  const plotW = plotRight - plotLeft;
  const plotH = plotBottom - plotTop;

  const timesMs = spec.timestampsUtc.map((s) => Date.parse(s));
  const nTime = timesMs.length;
  const nY = spec.yValues.length;
  const invertY = Boolean(spec.invertY);
  const zTransform = spec.zTransform ?? "linear";
  const yFormat = spec.yFormat ?? defaultHeatmapYFormat;

  let zLow = spec.zMin;
  let zHigh = spec.zMax;
  if (zLow === undefined || zHigh === undefined) {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let r = 0; r < spec.zMatrix.length; r++) {
      for (let c = 0; c < spec.zMatrix[r].length; c++) {
        const raw = spec.zMatrix[r][c];
        if (raw === null || raw === undefined || !Number.isFinite(raw)) continue;
        const val = zTransform === "log10" ? (raw > 0 ? Math.log10(raw) : Number.NaN) : raw;
        if (!Number.isFinite(val)) continue;
        if (val < lo) lo = val;
        if (val > hi) hi = val;
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      lo = 0;
      hi = 1;
    } else if (Math.abs(hi - lo) < 1e-12) {
      lo -= 0.5;
      hi += 0.5;
    }
    zLow = zLow ?? lo;
    zHigh = zHigh ?? hi;
  }

  const yMinV = nY > 0 ? Math.min(spec.yValues[0], spec.yValues[nY - 1]) : 0;
  const yMaxV = nY > 0 ? Math.max(spec.yValues[0], spec.yValues[nY - 1]) : 1;

  const yToNorm = (y: number): number => {
    if (spec.yScaleMode === "log") {
      const l0 = Math.log10(Math.max(1e-12, yMinV));
      const l1 = Math.log10(Math.max(1e-12, yMaxV));
      const ly = Math.log10(Math.max(1e-12, y));
      return l1 === l0 ? 0.5 : (ly - l0) / (l1 - l0);
    }
    return yMaxV === yMinV ? 0.5 : (y - yMinV) / (yMaxV - yMinV);
  };

  const yToPixel = (y: number): number => {
    const norm = yToNorm(y);
    const dir = invertY ? norm : 1 - norm;
    return plotTop + dir * plotH;
  };

  const pixelToY = (pxY: number): number => {
    const frac = Math.min(1, Math.max(0, (pxY - plotTop) / plotH));
    const norm = invertY ? frac : 1 - frac;
    if (spec.yScaleMode === "log") {
      const l0 = Math.log10(Math.max(1e-12, yMinV));
      const l1 = Math.log10(Math.max(1e-12, yMaxV));
      return Math.pow(10, l0 + norm * (l1 - l0));
    }
    return yMinV + norm * (yMaxV - yMinV);
  };

  const [vStart, vEnd] =
    spec.viewRangeMs ?? (nTime >= 2 ? [timesMs[0], timesMs[nTime - 1]] : [0, 1]);
  const spanMs = Math.max(1, vEnd - vStart);
  const lut = buildColorLut(spec.colormap);

  const raster = rasterMatrixToPngDataUrl(
    timesMs,
    spec.yValues,
    spec.zMatrix,
    spec.matrixLayout,
    zTransform,
    zLow,
    zHigh,
    lut,
    yToNorm,
    invertY,
    vStart,
    vEnd,
  );

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  );
  parts.push(`<title>${escapeXml(spec.title)}</title>`);
  if (spec.subtitle) {
    parts.push(`<desc>${escapeXml(spec.subtitle)}</desc>`);
  }
  parts.push(
    `<defs><clipPath id="heatmap-plot-clip"><rect x="${plotLeft}" y="${plotTop}" width="${plotW}" height="${plotH}"/></clipPath></defs>`,
  );
  parts.push(`<rect data-export-bg="1" x="0" y="0" width="${width}" height="${height}" fill="${bgColor}"/>`);

  // Header
  parts.push(
    `<text x="${plotLeft}" y="18" fill="${fgColor}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="13" font-weight="700">${escapeXml(spec.title)}</text>`,
  );
  if (spec.subtitle) {
    parts.push(
      `<text x="${plotLeft}" y="34" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="10">${escapeXml(spec.subtitle)}</text>`,
    );
  }

  parts.push(`<rect x="${plotLeft}" y="${plotTop}" width="${plotW}" height="${plotH}" fill="${plotBg}"/>`);

  // Clipped plot content
  parts.push(`<g clip-path="url(#heatmap-plot-clip)">`);
  if (raster.dataUrl && raster.xWidthFrac > 0) {
    const imgX = plotLeft + raster.xStartFrac * plotW;
    const imgW = raster.xWidthFrac * plotW;
    parts.push(
      `<image x="${imgX.toFixed(2)}" y="${plotTop}" width="${imgW.toFixed(2)}" height="${plotH}" preserveAspectRatio="none" image-rendering="pixelated" href="${raster.dataUrl}" xlink:href="${raster.dataUrl}"/>`,
    );
  }

  // Band guide lines
  if (spec.bandGuidesY) {
    for (const g of spec.bandGuidesY) {
      const gy = yToPixel(g.value);
      if (gy >= plotTop && gy <= plotBottom) {
        const gCol = isLight ? "#0f172a" : (g.color ?? "#f8fafc");
        parts.push(
          `<line x1="${plotLeft}" y1="${gy.toFixed(2)}" x2="${plotRight}" y2="${gy.toFixed(2)}" stroke="${gCol}" stroke-width="1" stroke-dasharray="5,4"/>`,
        );
        parts.push(
          `<rect x="${plotLeft + 4}" y="${(gy - 12).toFixed(2)}" width="84" height="12" fill="${isLight ? "#ffffff" : "#0f172a"}" fill-opacity="${isLight ? "0.85" : "0.82"}" rx="2"/>`,
        );
        parts.push(
          `<text x="${plotLeft + 7}" y="${(gy - 3).toFixed(2)}" fill="${fgColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5">${escapeXml(g.label)}</text>`,
        );
      }
    }
  }

  // COI polygon + dashed line
  if (spec.coiPeriods && spec.coiPeriods.length === nTime && nTime > 1) {
    const outsideY = invertY ? plotBottom : plotTop;
    let runStart = -1;
    const strokeCmds: string[] = [];
    for (let i = 0; i <= nTime; i++) {
      const p = i < nTime ? spec.coiPeriods[i] : null;
      const ok = p !== null && p !== undefined && Number.isFinite(p);
      if (ok) {
        if (runStart === -1) runStart = i;
        const x = plotLeft + ((timesMs[i] - vStart) / spanMs) * plotW;
        const y = Math.min(plotBottom, Math.max(plotTop, yToPixel(p as number)));
        strokeCmds.push(`${i === runStart ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
      } else if (runStart !== -1) {
        const polyPts: string[] = [];
        const firstX = plotLeft + ((timesMs[runStart] - vStart) / spanMs) * plotW;
        const lastX = plotLeft + ((timesMs[i - 1] - vStart) / spanMs) * plotW;
        polyPts.push(`${firstX.toFixed(2)},${outsideY.toFixed(2)}`);
        for (let k = runStart; k < i; k++) {
          const xk = plotLeft + ((timesMs[k] - vStart) / spanMs) * plotW;
          const yk = Math.min(plotBottom, Math.max(plotTop, yToPixel(spec.coiPeriods[k] as number)));
          polyPts.push(`${xk.toFixed(2)},${yk.toFixed(2)}`);
        }
        polyPts.push(`${lastX.toFixed(2)},${outsideY.toFixed(2)}`);
        parts.push(
          `<polygon points="${polyPts.join(" ")}" fill="${coiFillColor}" fill-opacity="${coiFillOpacity}"/>`,
        );
        runStart = -1;
      }
    }
    if (strokeCmds.length > 0) {
      parts.push(
        `<path d="${strokeCmds.join(" ")}" fill="none" stroke="${coiStroke}" stroke-width="1.5" stroke-dasharray="6,4"/>`,
      );
    }
  }

  // 95% MC contours
  if (spec.contourGrid && nTime > 1 && nY > 1) {
    const segs = extractSignificanceContours(spec.contourGrid.sigGrid, spec.contourGrid.insideCoi);
    if (segs.length > 0) {
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
        const y0 = spec.yValues[j0];
        const y1 = spec.yValues[j1];
        const yVal =
          spec.yScaleMode === "log" && y0 > 0 && y1 > 0
            ? y0 * Math.pow(y1 / y0, frac)
            : y0 + frac * (y1 - y0);
        return yToPixel(yVal);
      };
      const d = segs
        .map(
          ([[t0, s0], [t1, s1]]) =>
            `M${mapT(t0).toFixed(2)},${mapY(s0).toFixed(2)} L${mapT(t1).toFixed(2)},${mapY(s1).toFixed(2)}`,
        )
        .join(" ");
      parts.push(`<path d="${d}" fill="none" stroke="${contourColor}" stroke-width="1.5"/>`);
    }
  }

  // Phase arrows (explicit line + polygon arrowheads, no SVG markers)
  if (spec.phaseVectors && nTime > 0 && nY > 0) {
    let firstVis = 0;
    while (firstVis < nTime && timesMs[firstVis] < vStart) firstVis++;
    let lastVis = nTime - 1;
    while (lastVis >= 0 && timesMs[lastVis] > vEnd) lastVis--;
    const visCount = Math.max(0, lastVis - firstVis + 1);
    const stepX = spec.phaseVectors.stepX ?? Math.max(1, Math.floor(Math.max(1, visCount) / 30));
    const stepY = spec.phaseVectors.stepY ?? Math.max(2, Math.floor(nY / 14));
    const arrowLen = 7;
    const startT = Math.floor(firstVis / stepX) * stepX;

    for (let tIdx = startT; tIdx <= lastVis; tIdx += stepX) {
      if (tIdx < firstVis) continue;
      const tMs = timesMs[tIdx];
      if (tMs < vStart || tMs > vEnd) continue;
      const cx = plotLeft + ((tMs - vStart) / spanMs) * plotW;
      for (let yIdx = 0; yIdx < nY; yIdx += stepY) {
        if (spec.phaseVectors.sigMask && !spec.phaseVectors.sigMask[yIdx]?.[tIdx]) continue;
        const deg = spec.phaseVectors.phaseDeg[yIdx]?.[tIdx];
        if (deg === null || deg === undefined || !Number.isFinite(deg)) continue;
        const cy = yToPixel(spec.yValues[yIdx]);
        const rad = (deg * Math.PI) / 180;
        const dx = Math.cos(rad) * arrowLen;
        const dy = -Math.sin(rad) * arrowLen;
        const x0 = cx - dx * 0.5;
        const y0 = cy - dy * 0.5;
        const x1 = cx + dx * 0.5;
        const y1 = cy + dy * 0.5;
        const ang = Math.atan2(y1 - y0, x1 - x0);
        const ax1 = x1 - 3.2 * Math.cos(ang - Math.PI / 6);
        const ay1 = y1 - 3.2 * Math.sin(ang - Math.PI / 6);
        const ax2 = x1 - 3.2 * Math.cos(ang + Math.PI / 6);
        const ay2 = y1 - 3.2 * Math.sin(ang + Math.PI / 6);
        parts.push(
          `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="${arrowColor}" stroke-width="1.2"/>`,
        );
        parts.push(
          `<polygon points="${x1.toFixed(1)},${y1.toFixed(1)} ${ax1.toFixed(1)},${ay1.toFixed(1)} ${ax2.toFixed(1)},${ay2.toFixed(1)}" fill="${arrowColor}"/>`,
        );
      }
    }
  }
  parts.push(`</g>`);

  // Frame
  parts.push(
    `<rect x="${plotLeft}" y="${plotTop}" width="${plotW}" height="${plotH}" fill="none" stroke="${frameColor}" stroke-width="1"/>`,
  );

  // Y ticks & label
  const yTickCount = 5;
  for (let i = 0; i < yTickCount; i++) {
    const frac = i / (yTickCount - 1);
    const py = plotTop + frac * plotH;
    const val = pixelToY(py);
    parts.push(
      `<line x1="${plotLeft - 4}" y1="${py.toFixed(1)}" x2="${plotLeft}" y2="${py.toFixed(1)}" stroke="${frameColor}" stroke-width="1"/>`,
    );
    parts.push(
      `<text x="${plotLeft - 7}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5">${escapeXml(yFormat(val))}</text>`,
    );
  }
  parts.push(
    `<text transform="translate(15, ${(plotTop + plotH * 0.5).toFixed(1)}) rotate(-90)" text-anchor="middle" fill="${fgColor}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11">${escapeXml(spec.yLabel)}</text>`,
  );

  // X ticks
  const xTickCount = 6;
  for (let i = 0; i <= xTickCount; i++) {
    const frac = i / xTickCount;
    const px = plotLeft + frac * plotW;
    const tMs = vStart + frac * spanMs;
    parts.push(
      `<line x1="${px.toFixed(1)}" y1="${plotBottom}" x2="${px.toFixed(1)}" y2="${plotBottom + 4}" stroke="${frameColor}" stroke-width="1"/>`,
    );
    parts.push(
      `<text x="${px.toFixed(1)}" y="${plotBottom + 15}" text-anchor="middle" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5">${escapeXml(formatUtcShort(tMs))}</text>`,
    );
  }

  // Vector colorbar
  const cbX = plotRight + 10;
  const cbW = 12;
  const cbSteps = 64;
  const stepH = plotH / cbSteps;
  for (let s = 0; s < cbSteps; s++) {
    const norm = 1 - s / Math.max(1, cbSteps - 1);
    const cIdx = Math.round(norm * 255) * 3;
    const ry = plotTop + s * stepH;
    parts.push(
      `<rect x="${cbX}" y="${ry.toFixed(2)}" width="${cbW}" height="${(stepH + 0.6).toFixed(2)}" fill="rgb(${lut[cIdx]},${lut[cIdx + 1]},${lut[cIdx + 2]})"/>`,
    );
  }
  parts.push(
    `<rect x="${cbX}" y="${plotTop}" width="${cbW}" height="${plotH}" fill="none" stroke="${frameColor}" stroke-width="1"/>`,
  );
  parts.push(
    `<text x="${cbX + cbW + 4}" y="${plotTop + 7}" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9">${zHigh.toFixed(2)}</text>`,
  );
  parts.push(
    `<text x="${cbX + cbW + 4}" y="${(plotTop + plotH * 0.5 + 3).toFixed(1)}" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9">${((zHigh + zLow) * 0.5).toFixed(2)}</text>`,
  );
  parts.push(
    `<text x="${cbX + cbW + 4}" y="${plotBottom}" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9">${zLow.toFixed(2)}</text>`,
  );
  if (spec.zUnit) {
    parts.push(
      `<text x="${cbX}" y="${plotBottom + 14}" fill="${mutedColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9">${escapeXml(spec.zUnit)}</text>`,
    );
  }

  parts.push("</svg>");
  return sanitizeStandaloneSvg(parts.join("\n"));
}

// ---------------------------------------------------------------------------
// Station Map Standalone Vector SVG Builder (100% Vector, Light/Dark Print)
// ---------------------------------------------------------------------------

export interface MapSvgStationItem {
  iagaCode: string;
  x: number;
  y: number;
  geometry: StokesEllipseGeometry;
  isStationA: boolean;
  isStationB: boolean;
  inFlrChain: boolean;
}

export interface MapSvgArcItem {
  pathD: string;
  metrics: ArcCoherenceMetrics;
  isActivePair: boolean;
  labelX: number | null;
  labelY: number | null;
  labelText: string | null;
}

export interface StationMapSvgScene {
  width: number;
  height: number;
  theme: ExportTheme;
  title: string;
  caption: string;
  graticulePathD: string;
  landPathsD: string[];
  bordersPathD: string;
  nightPolyPathD: string | null;
  terminatorPathD: string | null;
  mlt12PathD: string | null;
  mlt12LabelPos: [number, number] | null;
  mlt00PathD: string | null;
  subsolarPos: [number, number] | null;
  nmpPos: [number, number] | null;
  arcs: MapSvgArcItem[];
  stations: MapSvgStationItem[];
  ouluPos: [number, number] | null;
  labels: PlacedLabel[];
  showTerminator: boolean;
  showEllipses: boolean;
  showArcs: boolean;
}

export function buildStationMapSvg(scene: StationMapSvgScene): string {
  const { width, height, theme } = scene;
  const isLight = theme === "light";

  const oceanFill = isLight ? "#f8fafc" : "#070d19";
  const landFill = isLight ? "#e2e8f0" : "#162238";
  const coastStroke = isLight ? "#94a3b8" : "#334155";
  const borderStroke = isLight ? "#cbd5e1" : "#1e293b";
  const gratStroke = isLight ? "#94a3b8" : "#334155";
  const gratOpacity = isLight ? "0.35" : "0.4";
  const nightFill = isLight ? "#0f172a" : "#020617";
  const nightOpacity = isLight ? "0.14" : "0.42";
  const termStroke = isLight ? "#b45309" : "#f59e0b";
  const mlt12Stroke = isLight ? "#d97706" : "#fbbf24";
  const cwColor = isLight ? "#dc2626" : "#f43f5e";
  const ccwColor = isLight ? "#0284c7" : "#38bdf8";
  const linColor = isLight ? "#6d28d9" : "#a78bfa";
  const insigColor = isLight ? "#64748b" : "#64748b";
  const textColor = isLight ? "#0f172a" : "#f1f5f9";
  const mutedText = isLight ? "#475569" : "#94a3b8";
  const frameStroke = isLight ? "#475569" : "#334155";

  const legendH = 44;
  const totalH = height + legendH;
  const parts: string[] = [];

  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${totalH}" viewBox="0 0 ${width} ${totalH}">`,
  );
  parts.push(`<title>${escapeXml(scene.title)}</title>`);
  parts.push(`<desc>${escapeXml(scene.caption)}</desc>`);
  parts.push(
    `<defs><clipPath id="map-viewport-clip"><rect x="0" y="0" width="${width}" height="${height}"/></clipPath></defs>`,
  );
  parts.push(
    `<rect data-export-bg="1" x="0" y="0" width="${width}" height="${totalH}" fill="${oceanFill}"/>`,
  );

  parts.push(`<g clip-path="url(#map-viewport-clip)">`);
  if (scene.graticulePathD) {
    parts.push(
      `<path d="${scene.graticulePathD}" fill="none" stroke="${gratStroke}" stroke-opacity="${gratOpacity}" stroke-width="0.7"/>`,
    );
  }
  for (const d of scene.landPathsD) {
    if (d) {
      parts.push(
        `<path d="${d}" fill="${landFill}" stroke="${coastStroke}" stroke-width="0.8"/>`,
      );
    }
  }
  if (scene.bordersPathD) {
    parts.push(
      `<path d="${scene.bordersPathD}" fill="none" stroke="${borderStroke}" stroke-width="0.5"/>`,
    );
  }

  if (scene.showTerminator) {
    if (scene.nightPolyPathD) {
      parts.push(
        `<path d="${scene.nightPolyPathD}" fill="${nightFill}" fill-opacity="${nightOpacity}" stroke="none"/>`,
      );
    }
    if (scene.terminatorPathD) {
      parts.push(
        `<path d="${scene.terminatorPathD}" fill="none" stroke="${termStroke}" stroke-width="1.5" stroke-dasharray="6,4"/>`,
      );
    }
    if (scene.mlt00PathD) {
      parts.push(
        `<path d="${scene.mlt00PathD}" fill="none" stroke="${mutedText}" stroke-width="1" stroke-dasharray="3,4" opacity="0.75"/>`,
      );
    }
    if (scene.mlt12PathD) {
      parts.push(
        `<path d="${scene.mlt12PathD}" fill="none" stroke="${mlt12Stroke}" stroke-width="2"/>`,
      );
    }
    if (scene.mlt12LabelPos) {
      parts.push(
        `<rect x="${(scene.mlt12LabelPos[0] - 28).toFixed(1)}" y="${(scene.mlt12LabelPos[1] - 9).toFixed(1)}" width="56" height="16" rx="3" fill="${isLight ? "#ffffff" : "#0f172a"}" fill-opacity="${isLight ? "0.9" : "0.88"}" stroke="${mlt12Stroke}" stroke-width="0.8"/>`,
      );
      parts.push(
        `<text x="${scene.mlt12LabelPos[0].toFixed(1)}" y="${(scene.mlt12LabelPos[1] + 2.5).toFixed(1)}" text-anchor="middle" fill="${mlt12Stroke}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5" font-weight="700">MLT = 12</text>`,
      );
    }
    if (scene.subsolarPos) {
      parts.push(
        `<circle cx="${scene.subsolarPos[0].toFixed(1)}" cy="${scene.subsolarPos[1].toFixed(1)}" r="4.5" fill="${mlt12Stroke}" stroke="${textColor}" stroke-width="1"/>`,
      );
    }
  }

  if (scene.nmpPos) {
    parts.push(
      `<circle cx="${scene.nmpPos[0].toFixed(1)}" cy="${scene.nmpPos[1].toFixed(1)}" r="3.5" fill="none" stroke="${mlt12Stroke}" stroke-width="1.2"/>`,
    );
    parts.push(
      `<text x="${(scene.nmpPos[0] + 6).toFixed(1)}" y="${(scene.nmpPos[1] + 3).toFixed(1)}" fill="${mlt12Stroke}" font-family="JetBrains Mono, Consolas, monospace" font-size="8.5">NMP</text>`,
    );
  }

  // Coherence arcs
  if (scene.showArcs) {
    for (const arc of scene.arcs) {
      if (!arc.pathD) continue;
      const col = coherenceArcColor(arc.metrics.gamma2, arc.metrics.isSignificant, theme);
      const sw = coherenceArcStrokeWidth(arc.metrics.gamma2, arc.isActivePair);
      const dash = arc.metrics.isSignificant ? "" : ' stroke-dasharray="5,4"';
      parts.push(
        `<path d="${arc.pathD}" fill="none" stroke="${col}" stroke-width="${sw.toFixed(2)}"${dash} opacity="${arc.isActivePair ? "0.95" : "0.72"}"/>`,
      );
      if (arc.labelText && arc.labelX !== null && arc.labelY !== null) {
        const boxW = Math.max(88, arc.labelText.length * 6.2 + 12);
        parts.push(
          `<rect x="${(arc.labelX - boxW * 0.5).toFixed(1)}" y="${(arc.labelY - 10).toFixed(1)}" width="${boxW.toFixed(1)}" height="16" rx="3" fill="${isLight ? "#ffffff" : "#0f172a"}" fill-opacity="${isLight ? "0.92" : "0.9"}" stroke="${col}" stroke-width="0.9"/>`,
        );
        parts.push(
          `<text x="${arc.labelX.toFixed(1)}" y="${(arc.labelY + 1.5).toFixed(1)}" text-anchor="middle" fill="${textColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5" font-weight="600">${escapeXml(arc.labelText)}</text>`,
        );
      }
    }
  }

  // OULU Neutron Monitor diamond
  if (scene.ouluPos) {
    const [ox, oy] = scene.ouluPos;
    const pts = `${ox.toFixed(1)},${(oy - 5.5).toFixed(1)} ${(ox + 5.5).toFixed(1)},${oy.toFixed(1)} ${ox.toFixed(1)},${(oy + 5.5).toFixed(1)} ${(ox - 5.5).toFixed(1)},${oy.toFixed(1)}`;
    parts.push(
      `<polygon points="${pts}" fill="${isLight ? "#0f766e" : "#2dd4bf"}" stroke="${textColor}" stroke-width="1"/>`,
    );
  }

  // 49 Magnetometer stations & Stokes ellipses
  for (const st of scene.stations) {
    const g = st.geometry;
    if (st.isStationA) {
      parts.push(
        `<circle cx="${st.x.toFixed(1)}" cy="${st.y.toFixed(1)}" r="${Math.max(10, g.majorRadius + 4).toFixed(1)}" fill="none" stroke="${ccwColor}" stroke-width="1.8"/>`,
      );
    } else if (st.isStationB) {
      parts.push(
        `<circle cx="${st.x.toFixed(1)}" cy="${st.y.toFixed(1)}" r="${Math.max(10, g.majorRadius + 4).toFixed(1)}" fill="none" stroke="${mlt12Stroke}" stroke-width="1.8"/>`,
      );
    } else if (st.inFlrChain) {
      parts.push(
        `<circle cx="${st.x.toFixed(1)}" cy="${st.y.toFixed(1)}" r="${Math.max(8, g.majorRadius + 3).toFixed(1)}" fill="none" stroke="${linColor}" stroke-width="1.2" stroke-dasharray="3,2"/>`,
      );
    }

    if (scene.showEllipses && g.status === "significant") {
      const col = g.sense === "CW" ? cwColor : g.sense === "CCW" ? ccwColor : linColor;
      const fillOpacity = isLight ? "0.16" : "0.22";

      parts.push(
        `<g transform="translate(${st.x.toFixed(2)}, ${st.y.toFixed(2)}) rotate(${g.rotationDeg.toFixed(2)})">`,
      );
      parts.push(
        `<ellipse cx="0" cy="0" rx="${Math.max(0.8, g.minorRadius).toFixed(2)}" ry="${g.majorRadius.toFixed(2)}" fill="${col}" fill-opacity="${fillOpacity}" stroke="${col}" stroke-width="1.7"/>`,
      );
      parts.push(
        `<line x1="0" y1="${(-g.majorRadius).toFixed(2)}" x2="0" y2="${g.majorRadius.toFixed(2)}" stroke="${col}" stroke-width="0.9" opacity="0.65"/>`,
      );
      if (g.showArrow) {
        parts.push(`<polygon points="${g.topArrowPoints}" fill="${col}"/>`);
        parts.push(`<polygon points="${g.bottomArrowPoints}" fill="${col}"/>`);
      }
      parts.push(`<circle cx="0" cy="0" r="1.8" fill="${col}"/>`);
      parts.push("</g>");
    } else if (g.status === "unavailable") {
      parts.push(
        `<circle cx="${st.x.toFixed(1)}" cy="${st.y.toFixed(1)}" r="3.6" fill="none" stroke="${insigColor}" stroke-width="1.2" stroke-dasharray="2,2"/>`,
      );
    } else {
      parts.push(
        `<circle cx="${st.x.toFixed(1)}" cy="${st.y.toFixed(1)}" r="3.8" fill="${insigColor}" stroke="${isLight ? "#ffffff" : "#0f172a"}" stroke-width="0.9" opacity="0.88"/>`,
      );
    }
  }

  // Labels
  for (const lbl of scene.labels) {
    parts.push(
      `<text x="${lbl.x.toFixed(1)}" y="${lbl.y.toFixed(1)}" text-anchor="${lbl.anchor}" fill="${textColor}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5" font-weight="600">${escapeXml(lbl.code)}</text>`,
    );
  }
  parts.push("</g>");

  // Border frame around map area
  parts.push(
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" fill="none" stroke="${frameStroke}" stroke-width="1"/>`,
  );

  // Legend & publication caption bar below map
  const legY = height + 16;
  const capY = height + 34;
  parts.push(
    `<rect x="0" y="${height}" width="${width}" height="${legendH}" fill="${isLight ? "#f1f5f9" : "#111827"}"/>`,
  );
  parts.push(
    `<line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="${frameStroke}" stroke-width="1"/>`,
  );

  const legItems = [
    { x: 14, color: cwColor, label: "CW (V>0, ε>0)" },
    { x: 130, color: ccwColor, label: "CCW (V<0, ε<0)" },
    { x: 256, color: insigColor, label: "p ≤ p_crit / no data" },
    { x: 422, color: mlt12Stroke, label: "MLT = 12" },
    { x: 504, color: termStroke, label: "Terminator" },
  ];
  for (const item of legItems) {
    parts.push(`<circle cx="${item.x}" cy="${legY - 3}" r="4" fill="${item.color}"/>`);
    parts.push(
      `<text x="${item.x + 8}" y="${legY}" fill="${textColor}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="10">${escapeXml(item.label)}</text>`,
    );
  }

  parts.push(
    `<text x="12" y="${capY}" fill="${mutedText}" font-family="JetBrains Mono, Consolas, monospace" font-size="9.5">${escapeXml(scene.caption)}</text>`,
  );

  parts.push("</svg>");
  return sanitizeStandaloneSvg(parts.join("\n"));
}
