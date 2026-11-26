import { geoContains } from "d3-geo";
import { describe, expect, it } from "vitest";
import {
  buildMltMeridianLine,
  buildNightHemispherePolygon,
  computeSubsolarPoint,
} from "../astronomy";
import {
  MAX_MAP_ZOOM,
  createFittedMapProjection,
  createMapPathGenerator,
  getWorldGeoData,
  measureLocalScalePxPerDeg,
  resolveLabelVisibility,
  scalePanOffsetForViewport,
  zoomMapViewportAtAnchor,
} from "../mapProjection";
import { ALL_49_STATIONS } from "./fixtures/stations49";

describe("Offline world-110m.topojson & Lambert Conformal Conic D3-geo projection", () => {
  it("parses embedded world-110m.topojson offline with non-empty land, countries, and borders", () => {
    const world = getWorldGeoData();
    expect(world.land.features.length).toBeGreaterThan(0);
    expect(world.countries.features.length).toBeGreaterThan(100);
    expect(world.borders.coordinates.length).toBeGreaterThan(50);
  });

  it("projects all 49 stations (-123.42 VIC to +29.70 IZN, 153.12 deg span) inside viewport in LCC", () => {
    const width = 960;
    const height = 460;
    const stationsLatLon = ALL_49_STATIONS.map((s) => ({ lat_deg: s.lat, lon_deg: s.lon }));

    const proj = createFittedMapProjection({
      regionPreset: "all",
      width,
      height,
      stations: stationsLatLon,
    });

    for (const st of ALL_49_STATIONS) {
      const pt = proj([st.lon, st.lat]);
      expect(pt).not.toBeNull();
      const [x, y] = pt as [number, number];
      expect(Number.isFinite(x)).toBe(true);
      expect(Number.isFinite(y)).toBe(true);
      expect(x).toBeGreaterThan(5);
      expect(x).toBeLessThan(width - 5);
      expect(y).toBeGreaterThan(5);
      expect(y).toBeLessThan(height - 5);
    }

    // High-latitude scale check vs Mercator blow-up (HRN 77 N, RES 74.69 N vs LVV 49.9 N)
    const sHrn = measureLocalScalePxPerDeg(proj, 15.55, 77.0);
    const sRes = measureLocalScalePxPerDeg(proj, -94.895, 74.69);
    const sLvv = measureLocalScalePxPerDeg(proj, 23.8, 49.9);
    expect(sHrn / sLvv).toBeLessThan(1.8);
    expect(sRes / sLvv).toBeLessThan(1.8);
  });

  it("renders valid SVG path strings for world land, night hemisphere, and MLT=12 meridian", () => {
    const world = getWorldGeoData();
    const proj = createFittedMapProjection({
      regionPreset: "all",
      width: 960,
      height: 460,
    });
    const pathGen = createMapPathGenerator(proj);

    const landD = pathGen(world.land) ?? "";
    expect(landD.length).toBeGreaterThan(500);

    const utc = Date.UTC(2026, 9, 10, 15, 0, 0);
    const sub = computeSubsolarPoint(utc);
    const antiLon = ((((sub.lonDeg + 180 + 180) % 360) + 360) % 360) - 180;
    const nightPoly = buildNightHemispherePolygon(utc, 2);

    expect(geoContains(nightPoly, [antiLon, -sub.declinationDeg])).toBe(true);
    expect(geoContains(nightPoly, [sub.lonDeg, sub.declinationDeg])).toBe(false);

    const nightD = pathGen(nightPoly) ?? "";
    const mlt12D = pathGen(buildMltMeridianLine(12, utc, 2)) ?? "";
    expect(nightD.startsWith("M")).toBe(true);
    expect(mlt12D.startsWith("M")).toBe(true);
  });

  it("places priority station labels cleanly without bounding-box collisions", () => {
    const proj = createFittedMapProjection({
      regionPreset: "all",
      width: 880,
      height: 400,
    });
    const candidates = ALL_49_STATIONS.map((s) => {
      const [x, y] = proj([s.lon, s.lat]) as [number, number];
      return {
        code: s.iaga,
        x,
        y,
        priority: s.iaga === "LVV" ? 1000 : s.iaga === "BEL" ? 960 : 150,
        radiusPx: 10,
      };
    });
    const placed = resolveLabelVisibility(candidates, 880, 400, false);
    const placedCodes = placed.map((p) => p.code);
    expect(placedCodes).toContain("LVV");
    expect(placedCodes).toContain("BEL");
    expect(placed.length).toBeGreaterThan(8);
  });

  it("scales panOffset proportionally from screen stage dimensions to export dimensions (F-05)", () => {
    const stations = ALL_49_STATIONS.map((s) => ({ lat_deg: s.lat, lon_deg: s.lon }));
    const pan = { x: -240, y: 110 };
    const zoomScale = 3.0;
    const screenW = 1400;
    const screenH = 520;
    const expW = 960;
    const expH = 440;

    const screenProj = createFittedMapProjection({
      regionPreset: "all",
      width: screenW,
      height: screenH,
      stations,
      zoomScale,
      panX: pan.x,
      panY: pan.y,
    });
    const centerGeo = screenProj.invert!([screenW * 0.5, screenH * 0.5])!;
    expect(centerGeo).not.toBeNull();

    const scaledPan = scalePanOffsetForViewport(pan, {
      regionPreset: "all",
      stations,
      fromWidth: screenW,
      fromHeight: screenH,
      toWidth: expW,
      toHeight: expH,
    });
    const expProj = createFittedMapProjection({
      regionPreset: "all",
      width: expW,
      height: expH,
      stations,
      zoomScale,
      panX: scaledPan.x,
      panY: scaledPan.y,
    });
    const projectedCenter = expProj(centerGeo)!;
    expect(Math.hypot(projectedCenter[0] - expW * 0.5, projectedCenter[1] - expH * 0.5)).toBeLessThan(0.1);
  });

  it("pins the geographic anchor point during zoomMapViewportAtAnchor in LCC without max-zoom drift", () => {
    const stations = ALL_49_STATIONS.map((s) => ({ lat_deg: s.lat, lon_deg: s.lon }));
    const width = 880;
    const height = 380;
    const anchorX = 215;
    const anchorY = 120;

    const start = { zoom: 1.5, panX: -45, panY: 28 };
    const projBefore = createFittedMapProjection({
      regionPreset: "all",
      width,
      height,
      stations,
      zoomScale: start.zoom,
      panX: start.panX,
      panY: start.panY,
    });
    const anchorGeo = projBefore.invert!([anchorX, anchorY])!;
    expect(anchorGeo).not.toBeNull();

    const zoomedIn = zoomMapViewportAtAnchor(start, {
      factor: 1.18,
      anchorX,
      anchorY,
      width,
      height,
    });
    const projIn = createFittedMapProjection({
      regionPreset: "all",
      width,
      height,
      stations,
      zoomScale: zoomedIn.zoom,
      panX: zoomedIn.panX,
      panY: zoomedIn.panY,
    });
    const ptIn = projIn(anchorGeo)!;
    expect(Math.hypot(ptIn[0] - anchorX, ptIn[1] - anchorY)).toBeLessThan(1e-6);

    const zoomedOut = zoomMapViewportAtAnchor(zoomedIn, {
      factor: 1 / 1.18,
      anchorX,
      anchorY,
      width,
      height,
    });
    const projOut = createFittedMapProjection({
      regionPreset: "all",
      width,
      height,
      stations,
      zoomScale: zoomedOut.zoom,
      panX: zoomedOut.panX,
      panY: zoomedOut.panY,
    });
    const ptOut = projOut(anchorGeo)!;
    expect(Math.hypot(ptOut[0] - anchorX, ptOut[1] - anchorY)).toBeLessThan(1e-6);

    // Symmetric zoom-in then zoom-out from {1, 0, 0} returns cleanly to {1, 0, 0}
    const zIn = zoomMapViewportAtAnchor(
      { zoom: 1, panX: 0, panY: 0 },
      { factor: 1.18, anchorX, anchorY, width, height },
    );
    const zBack = zoomMapViewportAtAnchor(zIn, {
      factor: 1 / 1.18,
      anchorX,
      anchorY,
      width,
      height,
    });
    expect(zBack.zoom).toBe(1);
    expect(Math.abs(zBack.panX)).toBeLessThan(1e-6);
    expect(Math.abs(zBack.panY)).toBeLessThan(1e-6);

    // Clamped at MAX_MAP_ZOOM: extra zoom-in does not drift pan
    const atMax = { zoom: MAX_MAP_ZOOM, panX: 120, panY: -80 };
    expect(
      zoomMapViewportAtAnchor(atMax, { factor: 1.18, anchorX, anchorY, width, height }),
    ).toEqual(atMax);
  });
});
