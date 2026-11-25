import type * as echarts from "echarts";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildMltMeridianLine,
  buildNightHemispherePolygon,
  buildTerminatorLine,
} from "../astronomy";
import {
  buildExportFilename,
  buildHeatmapPublicationSvg,
  buildStationMapSvg,
  crc32,
  formatCsvWithMetadata,
  injectPngDpiChunk,
  renderEChartsToSvg,
  sanitizeStandaloneSvg,
} from "../exportUtils";
import {
  createFittedMapProjection,
  createMapPathGenerator,
  getWorldGeoData,
} from "../mapProjection";
import { buildStokesEllipseGeometry, computeArcCoherenceMetrics } from "../stokesEllipse";
import { ALL_49_STATIONS } from "./fixtures/stations49";

function parseXmlSvg(svg: string): Document {
  expect(svg).not.toMatch(/(?:fill|stroke)="rgba\(/i);
  const parser = new DOMParser();
  const doc = parser.parseFromString(svg, "image/svg+xml");
  expect(doc.querySelector("parsererror")).toBeNull();
  expect(doc.documentElement.tagName.toLowerCase()).toBe("svg");
  expect(doc.documentElement.getAttribute("xmlns")).toBe("http://www.w3.org/2000/svg");
  return doc;
}

describe("Publication export subsystem (SVG, PNG 300 DPI, CSV, Windows filenames)", () => {
  beforeEach(() => {
    const fakeCtx = {
      createImageData: (w: number, h: number) => ({
        width: w,
        height: h,
        data: new Uint8ClampedArray(w * h * 4),
      }),
      putImageData: () => {},
      measureText: (txt: string) => ({ width: txt.length * 6.5 }),
      fillRect: () => {},
      clearRect: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      fill: () => {},
      save: () => {},
      restore: () => {},
      scale: () => {},
      translate: () => {},
      rotate: () => {},
      setTransform: () => {},
      fillText: () => {},
      strokeText: () => {},
      drawImage: () => {},
      clip: () => {},
      rect: () => {},
    } as unknown as CanvasRenderingContext2D;

    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
      (() => fakeCtx) as unknown as typeof HTMLCanvasElement.prototype.getContext,
    );
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
      "data:image/png;base64,iVBORw0KGgo=",
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sanitizes Windows-illegal filename characters cleanly", () => {
    const name = buildExportFilename(
      ["map", "storm_20261009_11", "2026-10-10T12:30:00Z", "A/B*C?"],
      "svg",
    );
    expect(name).not.toMatch(/[<>:"/\\|?*\s]/);
    expect(name).toBe("map_storm_20261009_11_2026-10-10T12_30_00Z_A_B_C.svg");
  });

  it("formats UTF-8 CSV with # metadata lines and RFC 4180 escaping", () => {
    const csv = formatCsvWithMetadata(
      { episode_id: "storm_20261009_11", alpha: 0.05, note: "line1\nline2" },
      ["station", "value", "comment"],
      [
        ["LVV", 12.5, 'normal, "quoted"'],
        ["SUA", null, Number.NaN],
      ],
    );
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("# episode_id: storm_20261009_11");
    expect(lines[1]).toBe("# alpha: 0.05");
    expect(lines[2]).toBe("# note: line1 line2");
    expect(lines[3]).toBe("station,value,comment");
    expect(lines[4]).toBe('LVV,12.5,"normal, ""quoted"""');
    expect(lines[5]).toBe("SUA,,");
  });

  it("injects standard PNG pHYs 300 DPI chunk (11811 px/m) with valid IEEE CRC-32", () => {
    // Minimal valid PNG byte sequence: signature + 13-byte IHDR + 0-byte IEND
    const raw = new Uint8Array(8 + 25 + 12);
    raw.set([137, 80, 78, 71, 13, 10, 26, 10], 0);
    const view = new DataView(raw.buffer);
    view.setUint32(8, 13, false);
    raw.set([73, 72, 68, 82], 12); // IHDR
    view.setUint32(16, 100, false);
    view.setUint32(20, 100, false);
    raw[24] = 8;
    raw[25] = 6;
    view.setUint32(29, crc32(raw.subarray(12, 29)), false);
    view.setUint32(33, 0, false);
    raw.set([73, 69, 78, 68], 37); // IEND
    view.setUint32(41, crc32(raw.subarray(37, 41)), false);

    const out = injectPngDpiChunk(raw, 300);
    expect(out.length).toBe(raw.length + 21);

    const outView = new DataView(out.buffer);
    // pHYs sits right after IHDR (8 + 25 = 33)
    expect(outView.getUint32(33, false)).toBe(9);
    expect(String.fromCharCode(out[37], out[38], out[39], out[40])).toBe("pHYs");
    expect(outView.getUint32(41, false)).toBe(11811);
    expect(outView.getUint32(45, false)).toBe(11811);
    expect(out[49]).toBe(1); // meters
    expect(outView.getUint32(50, false)).toBe(crc32(out.subarray(37, 50)));
  });

  it("exports ECharts line/Stokes option to standalone Light-theme vector SVG via SSR", () => {
    const t0 = Date.UTC(2026, 9, 10, 0, 0);
    const t1 = Date.UTC(2026, 9, 10, 1, 0);
    const option: echarts.EChartsCoreOption = {
      animation: false,
      tooltip: { trigger: "axis" },
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      grid: { left: 60, right: 40, top: 20, bottom: 25 },
      xAxis: { type: "time", min: t0, max: t1, axisLabel: { color: "#94a3b8" } },
      yAxis: { type: "value", name: "нТл", axisLabel: { color: "#38bdf8" } },
      series: [
        {
          name: "X",
          type: "line",
          showSymbol: false,
          lineStyle: { color: "#38bdf8", width: 1.5 },
          data: [
            [t0, 12.5],
            [t1, -8.3],
          ],
        },
      ],
    };

    const svg = renderEChartsToSvg(option, {
      width: 800,
      height: 300,
      theme: "light",
      viewRangeMs: [t0, t1],
      title: "Test Magnetogram",
      desc: "storm_20261009_11",
    });

    expect(svg).not.toContain("var(--");
    expect(svg).toContain("#ffffff");
    const doc = parseXmlSvg(svg);
    expect(doc.querySelectorAll("path").length).toBeGreaterThan(0);
  });

  it("enforces UTC X-axis tick labels in renderEChartsToSvg regardless of host timezone (F-02)", () => {
    const t0 = Date.UTC(2026, 9, 10, 1, 0, 0);
    const t1 = Date.UTC(2026, 9, 10, 5, 0, 0);
    const option: echarts.EChartsCoreOption = {
      animation: false,
      grid: { left: 60, right: 40, top: 20, bottom: 30 },
      xAxis: { type: "time", min: t0, max: t1, splitNumber: 4 },
      yAxis: { type: "value" },
      series: [
        {
          type: "line",
          data: [
            [t0, 1.0],
            [Date.UTC(2026, 9, 10, 3, 0, 0), 2.5],
            [t1, 1.5],
          ],
        },
      ],
    };

    const svg = renderEChartsToSvg(option, {
      width: 800,
      height: 260,
      theme: "light",
      viewRangeMs: [t0, t1],
      title: "UTC Axis Test",
    });

    const doc = parseXmlSvg(svg);
    const textLabels = Array.from(doc.querySelectorAll("text")).map((el) => el.textContent ?? "");
    const timeLabels = textLabels.filter((txt) => /\b\d{2}:\d{2}\b/.test(txt));
    expect(timeLabels).toContain("01:00");
    expect(timeLabels).toContain("02:00");
    expect(timeLabels).toContain("03:00");
    expect(timeLabels).toContain("04:00");
    expect(timeLabels.some((txt) => /0[6-9]:00/.test(txt))).toBe(false);
  });

  it("aligns 2D heatmap raster <image> bin centers exactly to zoomed viewRangeMs (F-03)", () => {
    const ts = Array.from({ length: 10 }, (_, h) =>
      new Date(Date.UTC(2026, 9, 10, h, 0, 0)).toISOString(),
    );
    const periods = [150, 300, 600];
    const zMatrix = periods.map(() => ts.map((_, c) => c + 1));
    const vStart = Date.UTC(2026, 9, 10, 1, 45, 0);
    const vEnd = Date.UTC(2026, 9, 10, 5, 15, 0);
    const width = 880;
    const plotLeft = 78;
    const plotRight = width - 74;
    const plotW = plotRight - plotLeft;

    const svg = buildHeatmapPublicationSvg({
      title: "Aligned Raster",
      theme: "light",
      width,
      height: 300,
      timestampsUtc: ts,
      yValues: periods,
      yLabel: "Period [s]",
      yScaleMode: "log",
      invertY: true,
      zMatrix,
      matrixLayout: "y-by-time",
      zMin: 0,
      zMax: 10,
      colormap: "viridis",
      viewRangeMs: [vStart, vEnd],
    });

    const doc = parseXmlSvg(svg);
    const imgEl = doc.querySelector("image");
    expect(imgEl).not.toBeNull();
    const imgX = Number(imgEl?.getAttribute("x"));
    const imgW = Number(imgEl?.getAttribute("width"));
    // Sub-raster covers 4 whole columns: 02:00, 03:00, 04:00, 05:00 -> [01:30Z, 05:30Z]
    const centerOf02ZInImg = imgX + (0.5 / 4) * imgW;
    const expectedAxisX02Z =
      plotLeft + ((Date.UTC(2026, 9, 10, 2, 0, 0) - vStart) / (vEnd - vStart)) * plotW;
    expect(Math.abs(centerOf02ZInImg - expectedAxisX02Z)).toBeLessThan(0.05);

    const centerOf05ZInImg = imgX + (3.5 / 4) * imgW;
    const expectedAxisX05Z =
      plotLeft + ((Date.UTC(2026, 9, 10, 5, 0, 0) - vStart) / (vEnd - vStart)) * plotW;
    expect(Math.abs(centerOf05ZInImg - expectedAxisX05Z)).toBeLessThan(0.05);
  });

  it("exports 2D WTC heatmap to standalone vector SVG with COI, contours, and phase arrows", () => {
    const ts = ["2026-10-10T00:00:00Z", "2026-10-10T01:00:00Z", "2026-10-10T02:00:00Z"];
    const periods = [150, 300, 600, 1200];
    const r2 = [
      [0.2, 0.9, 0.3],
      [0.3, 0.95, 0.4],
      [0.1, 0.4, 0.2],
      [0.1, 0.2, 0.1],
    ];
    const sigGrid = [
      [false, true, false],
      [false, true, false],
      [false, false, false],
      [false, false, false],
    ];
    const svg = buildHeatmapPublicationSvg({
      title: "WTC LVV-BEL",
      subtitle: "storm_20261009_11 | N_MC=300",
      theme: "light",
      timestampsUtc: ts,
      yValues: periods,
      yLabel: "Period [s]",
      yScaleMode: "log",
      invertY: true,
      zMatrix: r2,
      matrixLayout: "y-by-time",
      zMin: 0,
      zMax: 1,
      zUnit: "R^2",
      colormap: "turbo",
      viewRangeMs: null,
      coiPeriods: [200, 900, 200],
      contourGrid: { sigGrid },
      phaseVectors: {
        phaseDeg: [
          [0, 45, 90],
          [0, 45, 90],
          [0, 0, 0],
          [0, 0, 0],
        ],
        sigMask: sigGrid,
        stepX: 1,
        stepY: 1,
      },
    });

    expect(svg).not.toContain("<marker");
    expect(svg).not.toContain("var(--");
    const doc = parseXmlSvg(svg);
    expect(doc.querySelectorAll("polygon").length).toBeGreaterThanOrEqual(2); // COI + phase arrows
    expect(doc.querySelectorAll("path").length).toBeGreaterThanOrEqual(2); // COI dash + contour
  });

  it("builds standalone Light & Dark 49-station vector SVG maps cleanly without CSS vars or <marker>", () => {
    const w = 960;
    const h = 440;
    const world = getWorldGeoData();
    const proj = createFittedMapProjection({
      projectionType: "lcc",
      regionPreset: "all",
      width: w,
      height: h,
      stations: ALL_49_STATIONS.map((s) => ({ lat_deg: s.lat, lon_deg: s.lon })),
    });
    const pathGen = createMapPathGenerator(proj);
    const utc = Date.UTC(2026, 9, 10, 12, 0, 0);

    const stations = ALL_49_STATIONS.map((s, idx) => {
      const [x, y] = proj([s.lon, s.lat]) as [number, number];
      const isSig = idx % 3 !== 2;
      const isCw = idx % 2 === 0;
      const geom = buildStokesEllipseGeometry({
        stationIaga: s.iaga,
        available: true,
        isValid: true,
        stokes: {
          I: 120,
          Q: 20,
          U: 15,
          V: isCw ? 50 : -50,
          A: 25,
          R: 55.9,
          p: isSig ? 0.75 : 0.12,
          p_L: 0.2,
          p_C: 0.4,
          psi_deg: 18,
          chi_deg: isCw ? 20 : -20,
          epsilon: isCw ? 0.5 : -0.5,
          gamma2: 0.6,
          arg_sxy_deg: 45,
          n_eff: 60,
          p_crit: 0.25,
          gamma2_crit: 0.05,
          is_significant_p: isSig,
          is_significant_gamma2: true,
        },
        coordFrame: "XYZ",
        d0Deg: 0,
        localNorthDeg: 0,
        scaleMinI: 10,
        scaleMaxI: 500,
      });
      return {
        iagaCode: s.iaga,
        x,
        y,
        geometry: geom,
        isStationA: s.iaga === "LVV",
        isStationB: s.iaga === "BEL",
        inFlrChain: false,
      };
    });

    const arcPath =
      pathGen({
        type: "LineString",
        coordinates: [
          [20.789, 51.836],
          [23.8, 49.9],
        ],
      }) ?? "";

    const baseScene = {
      width: w,
      height: h,
      title: "GeoMag Analyzer Map",
      caption: "storm_20261009_11 | Pc5 | alpha=0.05",
      graticulePathD: pathGen(world.graticule) ?? "",
      landPathsD: world.land.features.map((f) => pathGen(f) ?? ""),
      bordersPathD: pathGen(world.borders) ?? "",
      nightPolyPathD: pathGen(buildNightHemispherePolygon(utc, 2)) ?? null,
      terminatorPathD: pathGen(buildTerminatorLine(utc, 3)) ?? null,
      mlt12PathD: pathGen(buildMltMeridianLine(12, utc, 3)) ?? null,
      mlt12LabelPos: [480, 140] as [number, number],
      mlt00PathD: null,
      subsolarPos: null,
      nmpPos: null,
      arcs: [
        {
          pathD: arcPath,
          metrics: computeArcCoherenceMetrics({
            stationI: "BEL",
            latI: 51.836,
            lonI: 20.789,
            stationJ: "LVV",
            latJ: 49.9,
            lonJ: 23.8,
            gamma2: 0.78,
            gamma2Crit: 0.1,
            phaseDeg: 25,
          }),
          isActivePair: true,
          labelX: 600,
          labelY: 220,
          labelText: "BEL-LVV: γ²=0.78",
        },
      ],
      stations,
      ouluPos: proj([25.468, 65.054]) as [number, number],
      labels: [{ code: "LVV", x: 610, y: 210, anchor: "start" as const }],
      showTerminator: true,
      showEllipses: true,
      showArcs: true,
    };

    for (const theme of ["light", "dark"] as const) {
      const svg = buildStationMapSvg({ ...baseScene, theme });
      expect(svg).not.toContain("var(--");
      expect(svg).not.toContain("<marker");
      expect(svg).toContain("MLT = 12");
      const doc = parseXmlSvg(svg);
      expect(doc.querySelectorAll("ellipse").length).toBeGreaterThan(25);
      expect(doc.querySelectorAll("polygon").length).toBeGreaterThan(20);
      expect(svg).toContain('fill-opacity="');
      expect(svg).toContain('stroke-opacity="');
    }
  });

  it("emits strictly SVG 1.1 fill-opacity/stroke-opacity without rgba(...) in heatmap/map SVGs and sanitizeStandaloneSvg (F-01)", () => {
    const rawSvg =
      '<svg width="100" height="50"><rect x="0" y="0" width="20" height="20" fill="rgba(15, 23, 42, 0.5)" fill-opacity="0.8" stroke="rgba(255, 0, 128, 0.25)"/></svg>';
    const normalized = sanitizeStandaloneSvg(rawSvg);
    const doc = parseXmlSvg(normalized);
    const rect = doc.querySelector("rect[x='0'][width='20']");
    expect(rect).not.toBeNull();
    expect(rect?.getAttribute("fill")).toBe("rgb(15,23,42)");
    expect(rect?.getAttribute("fill-opacity")).toBe("0.4");
    expect(rect?.getAttribute("stroke")).toBe("rgb(255,0,128)");
    expect(rect?.getAttribute("stroke-opacity")).toBe("0.25");

    for (const theme of ["light", "dark"] as const) {
      const heatmapSvg = buildHeatmapPublicationSvg({
        title: "Dark/Light COI & Guides",
        theme,
        timestampsUtc: ["2026-10-10T00:00:00Z", "2026-10-10T01:00:00Z"],
        yValues: [0.002, 0.005],
        yLabel: "Frequency [mHz]",
        yFormat: (y) => (y * 1000).toFixed(2),
        yScaleMode: "linear",
        zMatrix: [
          [1, 2],
          [3, 4],
        ],
        matrixLayout: "time-by-y",
        colormap: "viridis",
        viewRangeMs: null,
        coiPeriods: [0.003, 0.003],
        bandGuidesY: [{ value: 0.004, label: "Pc5", color: "rgba(56, 189, 248, 0.85)" }],
      });
      parseXmlSvg(heatmapSvg);
      expect(heatmapSvg).toContain('fill-opacity="');
      if (theme === "dark") {
        expect(heatmapSvg).toContain('stroke="rgb(56,189,248)" stroke-opacity="0.85"');
      }
    }
  });
});
