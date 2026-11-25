/**
 * 100% offline cartographic projection helpers using D3-geo + embedded world-110m.topojson.
 * Covers all 49 INTERMAGNET stations across ~153.12 deg longitude (-123.42 VIC to +29.70 IZN)
 * and 28.3 N to 77.0 N without Mercator polar blow-up or external map tiles.
 */

import {
  geoAzimuthalEqualArea,
  geoConicConformal,
  geoGraticule10,
  geoPath,
  type GeoProjection,
} from "d3-geo";
import type { Feature, FeatureCollection, Geometry, MultiLineString, MultiPoint } from "geojson";
import { feature, mesh } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";
import worldTopoRaw from "../assets/world-110m.topojson?raw";

export type MapProjectionType = "lcc" | "laea";
export type MapRegionPreset = "all" | "europe" | "flr" | "canada";

export interface RegionPresetConfig {
  id: MapRegionPreset;
  label: string;
  rotateLon: number;
  centerLat: number;
  parallels: [number, number];
  lonBounds: [number, number];
  latBounds: [number, number];
}

export const MAP_REGION_PRESETS: Record<MapRegionPreset, RegionPresetConfig> = {
  all: {
    id: "all",
    label: "All 49 (153°)",
    rotateLon: 47,
    centerLat: 56,
    parallels: [38, 68],
    lonBounds: [-126, 33],
    latBounds: [26, 79],
  },
  europe: {
    id: "europe",
    label: "Europe (37)",
    rotateLon: -7,
    centerLat: 53,
    parallels: [38, 68],
    lonBounds: [-19, 33],
    latBounds: [26, 79],
  },
  flr: {
    id: "flr",
    label: "FLR Chain",
    rotateLon: -20,
    centerLat: 58,
    parallels: [44, 70],
    lonBounds: [6, 33],
    latBounds: [36, 79],
  },
  canada: {
    id: "canada",
    label: "Canada (12)",
    rotateLon: 90,
    centerLat: 59,
    parallels: [46, 70],
    lonBounds: [-127, -49],
    latBounds: [41, 77],
  },
};

export interface WorldGeoData {
  land: FeatureCollection<Geometry>;
  countries: FeatureCollection<Geometry>;
  borders: MultiLineString;
  graticule: ReturnType<typeof geoGraticule10>;
}

let cachedWorldGeo: WorldGeoData | null = null;

/** Parse and cache embedded world-110m.topojson synchronously (zero network calls). */
export function getWorldGeoData(): WorldGeoData {
  if (cachedWorldGeo) return cachedWorldGeo;
  const topo = JSON.parse(worldTopoRaw) as Topology<{
    land?: GeometryCollection;
    countries: GeometryCollection;
  }>;

  const countriesCol = feature(topo, topo.objects.countries) as FeatureCollection<Geometry>;
  const landCol = topo.objects.land
    ? (feature(topo, topo.objects.land) as FeatureCollection<Geometry>)
    : countriesCol;
  const bordersMesh = mesh(topo, topo.objects.countries, (a, b) => a !== b);

  cachedWorldGeo = {
    land: landCol,
    countries: countriesCol,
    borders: bordersMesh,
    graticule: geoGraticule10(),
  };
  return cachedWorldGeo;
}

export interface CreateProjectionParams {
  projectionType: MapProjectionType;
  regionPreset: MapRegionPreset;
  width: number;
  height: number;
  stations?: ReadonlyArray<{ lat_deg: number; lon_deg: number }>;
  zoomScale?: number;
  panX?: number;
  panY?: number;
  paddingX?: number;
  paddingY?: number;
}

function buildRegionFitGeometry(
  cfg: RegionPresetConfig,
  stations?: ReadonlyArray<{ lat_deg: number; lon_deg: number }>,
): Feature<MultiPoint> {
  const [lonMin, lonMax] = cfg.lonBounds;
  const [latMin, latMax] = cfg.latBounds;
  const coords: [number, number][] = [];

  for (let lon = lonMin; lon <= lonMax; lon += 4) {
    coords.push([lon, latMin]);
    coords.push([lon, latMax]);
  }
  coords.push([lonMax, latMin], [lonMax, latMax]);

  for (let lat = latMin; lat <= latMax; lat += 4) {
    coords.push([lonMin, lat]);
    coords.push([lonMax, lat]);
  }

  if (stations) {
    for (const st of stations) {
      if (
        st.lon_deg >= lonMin - 2 &&
        st.lon_deg <= lonMax + 2 &&
        st.lat_deg >= latMin - 2 &&
        st.lat_deg <= latMax + 2
      ) {
        coords.push([st.lon_deg, st.lat_deg]);
      }
    }
  }

  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "MultiPoint",
      coordinates: coords,
    },
  };
}

/**
 * Create a fitted Lambert Conformal Conic ('lcc') or Lambert Azimuthal Equal-Area ('laea')
 * D3 projection enclosing the chosen region and stations without high-latitude Mercator distortion.
 */
export function createFittedMapProjection({
  projectionType,
  regionPreset,
  width,
  height,
  stations,
  zoomScale = 1,
  panX = 0,
  panY = 0,
  paddingX = 28,
  paddingY = 24,
}: CreateProjectionParams): GeoProjection {
  const cfg = MAP_REGION_PRESETS[regionPreset];
  const w = Math.max(240, width);
  const h = Math.max(180, height);
  const padX = Math.min(paddingX, Math.floor(w * 0.14));
  const padY = Math.min(paddingY, Math.floor(h * 0.14));

  const fitObj = buildRegionFitGeometry(cfg, stations);
  const proj: GeoProjection =
    projectionType === "laea"
      ? geoAzimuthalEqualArea()
          .rotate([cfg.rotateLon, -cfg.centerLat])
          .clipAngle(110)
      : geoConicConformal()
          .rotate([cfg.rotateLon, 0])
          .center([0, cfg.centerLat])
          .parallels(cfg.parallels);

  proj.fitExtent(
    [
      [padX, padY],
      [w - padX, h - padY],
    ],
    fitObj,
  );

  const safeZoom = Math.min(12, Math.max(0.6, zoomScale));
  if (Math.abs(safeZoom - 1) > 1e-6 || Math.abs(panX) > 1e-6 || Math.abs(panY) > 1e-6) {
    const baseScale = proj.scale();
    const [tx, ty] = proj.translate();
    const cx = w * 0.5;
    const cy = h * 0.5;
    proj
      .scale(baseScale * safeZoom)
      .translate([
        cx + (tx - cx) * safeZoom + panX,
        cy + (ty - cy) * safeZoom + panY,
      ]);
  }

  proj.clipExtent([
    [0, 0],
    [w, h],
  ]);
  return proj;
}

/**
 * Proportionally rescale screen pan offset [px] when rendering the same zoomed/panned
 * map view into a different viewport size (e.g. responsive screen stage -> fixed 960px SVG).
 */
export function scalePanOffsetForViewport(
  pan: { x: number; y: number },
  params: {
    projectionType: MapProjectionType;
    regionPreset: MapRegionPreset;
    stations?: ReadonlyArray<{ lat_deg: number; lon_deg: number }>;
    fromWidth: number;
    fromHeight: number;
    toWidth: number;
    toHeight: number;
  },
): { x: number; y: number } {
  if (Math.abs(pan.x) < 1e-9 && Math.abs(pan.y) < 1e-9) return pan;
  const fw = Math.max(240, params.fromWidth);
  const fh = Math.max(180, params.fromHeight);
  const tw = Math.max(240, params.toWidth);
  const th = Math.max(180, params.toHeight);
  if (fw === tw && fh === th) return pan;

  const fromBase = createFittedMapProjection({
    projectionType: params.projectionType,
    regionPreset: params.regionPreset,
    width: fw,
    height: fh,
    stations: params.stations,
  });
  const toBase = createFittedMapProjection({
    projectionType: params.projectionType,
    regionPreset: params.regionPreset,
    width: tw,
    height: th,
    stations: params.stations,
  });
  const ratio = toBase.scale() / Math.max(1e-9, fromBase.scale());
  return { x: pan.x * ratio, y: pan.y * ratio };
}

export function createMapPathGenerator(projection: GeoProjection) {
  return geoPath(projection);
}

/**
 * Measure local isotropic pixel scale (px per km) at (lonDeg, latDeg) to verify bounded
 * distortion across high-latitude stations (HRN 77 N, RES 74.7 N) vs mid-latitude LVV.
 */
export function measureLocalScalePxPerDeg(
  projection: GeoProjection,
  lonDeg: number,
  latDeg: number,
): number {
  const dLat = 0.1;
  const p0 = projection([lonDeg, latDeg]);
  const p1 = projection([lonDeg, Math.min(89.9, latDeg + dLat)]);
  if (!p0 || !p1) return Number.NaN;
  return Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dLat;
}

export interface LabelCandidate {
  code: string;
  x: number;
  y: number;
  priority: number;
  radiusPx: number;
}

export interface PlacedLabel {
  code: string;
  x: number;
  y: number;
  anchor: "start" | "end" | "middle";
}

interface LabelBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function boxesOverlap(a: LabelBox, b: LabelBox): boolean {
  return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
}

/**
 * Deterministic greedy label placement preventing overlaps on dense European networks
 * while always preserving priority stations (A, B, FLR chain, hovered, anchors).
 */
export function resolveLabelVisibility(
  candidates: LabelCandidate[],
  width: number,
  height: number,
  showAll = false,
): PlacedLabel[] {
  const sorted = candidates.slice().sort((a, b) => b.priority - a.priority);
  const placed: PlacedLabel[] = [];
  const boxes: LabelBox[] = [];
  const labelW = 30;
  const labelH = 12;

  for (const c of sorted) {
    if (c.x < 4 || c.x > width - 4 || c.y < 4 || c.y > height - 4) {
      continue;
    }
    const r = Math.max(4, c.radiusPx);
    const offsets: Array<{
      x: number;
      y: number;
      anchor: "start" | "end" | "middle";
      box: LabelBox;
    }> = [
      {
        x: c.x + r + 4,
        y: c.y + 3,
        anchor: "start",
        box: { x0: c.x + r + 3, y0: c.y - 7, x1: c.x + r + 3 + labelW, y1: c.y + 5 },
      },
      {
        x: c.x - r - 4,
        y: c.y + 3,
        anchor: "end",
        box: { x0: c.x - r - 3 - labelW, y0: c.y - 7, x1: c.x - r - 3, y1: c.y + 5 },
      },
      {
        x: c.x,
        y: c.y - r - 4,
        anchor: "middle",
        box: { x0: c.x - labelW * 0.5, y0: c.y - r - 4 - labelH, x1: c.x + labelW * 0.5, y1: c.y - r - 2 },
      },
      {
        x: c.x,
        y: c.y + r + 11,
        anchor: "middle",
        box: { x0: c.x - labelW * 0.5, y0: c.y + r + 1, x1: c.x + labelW * 0.5, y1: c.y + r + 1 + labelH },
      },
    ];

    if (showAll) {
      placed.push({ code: c.code, x: offsets[0].x, y: offsets[0].y, anchor: offsets[0].anchor });
      continue;
    }

    for (const opt of offsets) {
      const { box } = opt;
      if (box.x0 < 2 || box.x1 > width - 2 || box.y0 < 2 || box.y1 > height - 2) {
        continue;
      }
      if (!boxes.some((existing) => boxesOverlap(existing, box))) {
        boxes.push(box);
        placed.push({ code: c.code, x: opt.x, y: opt.y, anchor: opt.anchor });
        break;
      }
    }
  }

  return placed;
}
