/**
 * Stokes polarization ellipse geometry, dynamic snapshot alpha masking,
 * and cross-station coherence arc metrics matching .docx §§1, 5.3, 5.9 and cross_welch.py.
 */

import { geoDistance, type GeoProjection } from "d3-geo";
import type {
  CoordFrame,
  NetworkSnapshotResponse,
  NetworkStationSnapshot,
  Stokes2DOut,
} from "../api/types";
import { gamma2Crit, pCrit } from "./thresholds";

export type PolarizationSense = "CW" | "CCW" | "LINEAR" | "NONE";
export type StationEllipseStatus = "significant" | "insignificant" | "unavailable";

export interface StokesEllipseInput {
  stationIaga: string;
  available: boolean;
  isValid: boolean;
  stokes: Stokes2DOut | null;
  coordFrame: CoordFrame;
  d0Deg: number;
  localNorthDeg: number;
  scaleMinI: number;
  scaleMaxI: number;
  sizeMode?: "log" | "sqrt";
  minRadiusPx?: number;
  maxRadiusPx?: number;
}

export interface StokesEllipseGeometry {
  stationIaga: string;
  status: StationEllipseStatus;
  sense: PolarizationSense;
  majorRadius: number;
  minorRadius: number;
  axisRatio: number;
  psiGeoDeg: number;
  rotationDeg: number;
  showArrow: boolean;
  topArrowPoints: string;
  bottomArrowPoints: string;
}

/**
 * Recompute p_crit, gamma2_crit, and significance flags across a NetworkSnapshotResponse
 * instantaneously on the client when alpha changes.
 */
export function applyNetworkSnapshotAlphaMask(
  snapshot: NetworkSnapshotResponse,
  alpha: number,
): NetworkSnapshotResponse {
  const memo = new Map<number, { pc: number; gc: number }>();
  const getCrit = (nEff: number) => {
    let entry = memo.get(nEff);
    if (!entry) {
      entry = {
        pc: pCrit(nEff, alpha),
        gc: gamma2Crit(nEff, alpha),
      };
      memo.set(nEff, entry);
    }
    return entry;
  };

  const stations: NetworkStationSnapshot[] = snapshot.stations.map((st) => {
    if (!st.stokes_2d) return st;
    const nEff = st.stokes_2d.n_eff;
    if (nEff === null || nEff === undefined || !Number.isFinite(nEff) || nEff <= 1) {
      return st;
    }
    const { pc, gc } = getCrit(nEff);
    const nextStokes: Stokes2DOut = {
      ...st.stokes_2d,
      p_crit: pc,
      gamma2_crit: gc,
      is_significant_p: st.stokes_2d.p !== null ? st.stokes_2d.p > pc : false,
      is_significant_gamma2: st.stokes_2d.gamma2 !== null ? st.stokes_2d.gamma2 > gc : false,
    };
    return {
      ...st,
      stokes_2d: nextStokes,
    };
  });

  return {
    ...snapshot,
    stations,
  };
}

/**
 * Compute UTC [startIso, endIso] window of duration `windowSamples` minutes centered on
 * `cursorUtcMs` and clamped inside `boundsMs` so boundary cursors retain full window length.
 */
export function computeSnapshotWindowUtc(
  cursorUtcMs: number,
  windowSamples: number,
  boundsMs: [number, number] | null,
): { startIso: string; endIso: string; centerMs: number } {
  const spanMin = Math.max(128, Math.round(windowSamples));
  const halfSpanMs = 0.5 * spanMin * 60_000;
  let startMs = Math.round((cursorUtcMs - halfSpanMs) / 60_000) * 60_000;
  let endMs = startMs + spanMin * 60_000;

  if (boundsMs) {
    const bStart = Math.min(boundsMs[0], boundsMs[1]);
    const bEnd = Math.max(boundsMs[0], boundsMs[1]);
    if (endMs - startMs >= bEnd - bStart) {
      startMs = bStart;
      endMs = bEnd;
    } else {
      if (startMs < bStart) {
        startMs = bStart;
        endMs = bStart + spanMin * 60_000;
      }
      if (endMs > bEnd) {
        endMs = bEnd;
        startMs = Math.max(bStart, bEnd - spanMin * 60_000);
      }
    }
  }

  return {
    startIso: new Date(startMs).toISOString(),
    endIso: new Date(endMs).toISOString(),
    centerMs: Math.round(0.5 * (startMs + endMs)),
  };
}

/**
 * Compute clockwise screen bearing [deg] of local geographic North from SVG -y (screen up)
 * at (lonDeg, latDeg) under the active D3 projection.
 */
export function computeLocalNorthBearingDeg(
  projection: GeoProjection,
  lonDeg: number,
  latDeg: number,
): number {
  const p0 = projection([lonDeg, latDeg]);
  const pN = projection([lonDeg, Math.min(89.9, latDeg + 0.05)]);
  if (!p0 || !pN) return 0;
  const dx = pN[0] - p0[0];
  const dy = pN[1] - p0[1];
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || Math.hypot(dx, dy) < 1e-9) {
    return 0;
  }
  return (Math.atan2(dx, -dy) * 180) / Math.PI;
}

/**
 * Map total horizontal pulsation power I [nT^2] monotonically into ellipse major semi-axis [px].
 */
export function computeEllipseRadius(
  powerI: number,
  scaleMinI: number,
  scaleMaxI: number,
  sizeMode: "log" | "sqrt" = "log",
  minRadiusPx = 6,
  maxRadiusPx = 20,
): number {
  if (!Number.isFinite(powerI) || powerI <= 0) return minRadiusPx;
  const lo = Math.max(0, scaleMinI);
  const hi = Math.max(lo, scaleMaxI);
  if (hi - lo < 1e-12) {
    return 0.5 * (minRadiusPx + maxRadiusPx);
  }
  const transform = (v: number) =>
    sizeMode === "sqrt" ? Math.sqrt(Math.max(0, v)) : Math.log10(1 + Math.max(0, v));
  const tVal = transform(powerI);
  const tLo = transform(lo);
  const tHi = transform(hi);
  const frac = tHi > tLo ? Math.min(1, Math.max(0, (tVal - tLo) / (tHi - tLo))) : 0.5;
  return minRadiusPx + frac * (maxRadiusPx - minRadiusPx);
}

/**
 * Build Stokes ellipse geometry:
 * - V > 0 (epsilon > 0) => CW rotation (X -> Y clockwise viewed from above, +Z down)
 * - V < 0 (epsilon < 0) => CCW rotation
 * - |epsilon| < 0.05 => LINEAR (no circular sense arrow)
 * - p <= p_crit(alpha) or missing/invalid => neutral gray marker
 * - majorRadius along local North + psiGeoDeg; minorRadius = majorRadius * |epsilon|
 */
export function buildStokesEllipseGeometry(input: StokesEllipseInput): StokesEllipseGeometry {
  const {
    stationIaga,
    available,
    isValid,
    stokes,
    coordFrame,
    d0Deg,
    localNorthDeg,
    scaleMinI,
    scaleMaxI,
    sizeMode = "log",
    minRadiusPx = 6,
    maxRadiusPx = 20,
  } = input;

  if (!available) {
    return {
      stationIaga,
      status: "unavailable",
      sense: "NONE",
      majorRadius: 0,
      minorRadius: 0,
      axisRatio: 0,
      psiGeoDeg: 0,
      rotationDeg: localNorthDeg,
      showArrow: false,
      topArrowPoints: "",
      bottomArrowPoints: "",
    };
  }

  if (
    !isValid ||
    !stokes ||
    !stokes.is_significant_p ||
    stokes.I === null ||
    !Number.isFinite(stokes.I) ||
    stokes.I <= 0
  ) {
    return {
      stationIaga,
      status: "insignificant",
      sense: "NONE",
      majorRadius: 0,
      minorRadius: 0,
      axisRatio: 0,
      psiGeoDeg: 0,
      rotationDeg: localNorthDeg,
      showArrow: false,
      topArrowPoints: "",
      bottomArrowPoints: "",
    };
  }

  const majorRadius = computeEllipseRadius(
    stokes.I,
    scaleMinI,
    scaleMaxI,
    sizeMode,
    minRadiusPx,
    maxRadiusPx,
  );
  const rawEps =
    stokes.epsilon !== null && Number.isFinite(stokes.epsilon) ? stokes.epsilon : 0;
  const absEps = Math.min(1, Math.max(0, Math.abs(rawEps)));
  const minorRadius = majorRadius * absEps;

  const vVal = stokes.V !== null && Number.isFinite(stokes.V) ? stokes.V : rawEps;
  let sense: PolarizationSense = "LINEAR";
  if (absEps >= 0.05 && Math.abs(vVal) > 0) {
    sense = vVal > 0 ? "CW" : "CCW";
  }

  const rawPsi =
    stokes.psi_deg !== null && Number.isFinite(stokes.psi_deg) ? stokes.psi_deg : 0;
  const declOffset = coordFrame === "XmYm" && Number.isFinite(d0Deg) ? d0Deg : 0;
  const psiGeoDeg = rawPsi + declOffset;
  const rotationDeg = localNorthDeg + psiGeoDeg;

  const showArrow = sense === "CW" || sense === "CCW";
  let topArrowPoints = "";
  let bottomArrowPoints = "";

  if (showArrow) {
    const dir = sense === "CW" ? 1 : -1;
    const len = Math.max(4.4, Math.min(7.2, majorRadius * 0.44));
    const halfH = len * 0.56;
    const tipX = (dir * len * 0.65).toFixed(2);
    const baseX = (-dir * len * 0.35).toFixed(2);
    const topY = (-majorRadius).toFixed(2);
    const topYLo = (-majorRadius - halfH).toFixed(2);
    const topYHi = (-majorRadius + halfH).toFixed(2);
    topArrowPoints = `${tipX},${topY} ${baseX},${topYLo} ${baseX},${topYHi}`;

    const botTipX = (-dir * len * 0.65).toFixed(2);
    const botBaseX = (dir * len * 0.35).toFixed(2);
    const botY = majorRadius.toFixed(2);
    const botYLo = (majorRadius - halfH).toFixed(2);
    const botYHi = (majorRadius + halfH).toFixed(2);
    bottomArrowPoints = `${botTipX},${botY} ${botBaseX},${botYLo} ${botBaseX},${botYHi}`;
  }

  return {
    stationIaga,
    status: "significant",
    sense,
    majorRadius,
    minorRadius,
    axisRatio: absEps,
    psiGeoDeg,
    rotationDeg,
    showArrow,
    topArrowPoints,
    bottomArrowPoints,
  };
}

export interface ArcCoherenceMetrics {
  stationI: string;
  stationJ: string;
  deltaLonDeg: number;
  distanceKm: number;
  gamma2: number | null;
  gamma2Crit: number | null;
  phaseDeg: number | null;
  phaseDegMasked: number | null;
  azimuthalM: number | null;
  mAlias: number | null;
  isSignificant: boolean;
  isMeridionalGuarded: boolean;
}

const EARTH_MEAN_RADIUS_KM = 6371.0088;

/**
 * Compute geodesic distance, longitudinal separation, m-number, and meridional guard
 * matching cross_welch.py conventions (delta_lon = lon_i - lon_j wrapped to [-180, 180)).
 */
export function computeArcCoherenceMetrics(params: {
  stationI: string;
  latI: number;
  lonI: number;
  stationJ: string;
  latJ: number;
  lonJ: number;
  gamma2: number | null;
  gamma2Crit: number | null;
  phaseDeg: number | null;
  minDeltaLonDeg?: number;
  azimuthalMOverride?: number | null;
  mAliasOverride?: number | null;
  isMeridionalGuardedOverride?: boolean;
}): ArcCoherenceMetrics {
  const minDeltaLonDeg = params.minDeltaLonDeg ?? 3.0;
  const deltaLonDeg = ((((params.lonI - params.lonJ + 180) % 360) + 360) % 360) - 180;
  const distanceKm =
    geoDistance([params.lonI, params.latI], [params.lonJ, params.latJ]) *
    EARTH_MEAN_RADIUS_KM;

  const isMeridionalGuarded =
    params.isMeridionalGuardedOverride ?? Math.abs(deltaLonDeg) < minDeltaLonDeg;

  const gamma2 =
    params.gamma2 !== null && Number.isFinite(params.gamma2) ? params.gamma2 : null;
  const gamma2Crit =
    params.gamma2Crit !== null && Number.isFinite(params.gamma2Crit)
      ? params.gamma2Crit
      : null;
  const phaseDeg =
    params.phaseDeg !== null && Number.isFinite(params.phaseDeg) ? params.phaseDeg : null;

  const isSignificant = Boolean(
    gamma2 !== null && gamma2Crit !== null && gamma2 > gamma2Crit,
  );
  const phaseDegMasked = isSignificant ? phaseDeg : null;

  const dLonRad = (deltaLonDeg * Math.PI) / 180;
  const mAlias =
    params.mAliasOverride !== undefined
      ? params.mAliasOverride
      : isMeridionalGuarded || Math.abs(dLonRad) < 1e-12
        ? null
        : Math.PI / Math.abs(dLonRad);

  let azimuthalM: number | null = null;
  if (!isMeridionalGuarded && isSignificant) {
    if (
      params.azimuthalMOverride !== undefined &&
      params.azimuthalMOverride !== null &&
      Number.isFinite(params.azimuthalMOverride)
    ) {
      azimuthalM = params.azimuthalMOverride;
    } else if (phaseDeg !== null && Math.abs(dLonRad) >= 1e-12) {
      azimuthalM = ((phaseDeg * Math.PI) / 180) / dLonRad;
    }
  }

  return {
    stationI: params.stationI,
    stationJ: params.stationJ,
    deltaLonDeg,
    distanceKm,
    gamma2,
    gamma2Crit,
    phaseDeg,
    phaseDegMasked,
    azimuthalM,
    mAlias,
    isSignificant,
    isMeridionalGuarded,
  };
}

export function coherenceArcStrokeWidth(gamma2: number | null, isActivePair: boolean): number {
  const base = gamma2 !== null && Number.isFinite(gamma2) ? 1.5 + 4.0 * Math.min(1, Math.max(0, gamma2)) : 1.5;
  return isActivePair ? Math.max(2.4, base) : Math.max(1.2, base * 0.75);
}

export function coherenceArcColor(
  gamma2: number | null,
  isSignificant: boolean,
  theme: "light" | "dark" = "dark",
): string {
  if (!isSignificant || gamma2 === null || !Number.isFinite(gamma2)) {
    return theme === "light" ? "#94a3b8" : "#64748b";
  }
  if (gamma2 >= 0.7) {
    return theme === "light" ? "#059669" : "#34d399";
  }
  if (gamma2 >= 0.4) {
    return theme === "light" ? "#0284c7" : "#38bdf8";
  }
  return theme === "light" ? "#d97706" : "#fbbf24";
}
