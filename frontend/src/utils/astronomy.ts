/**
 * Pure astronomical and centered-dipole geomagnetic coordinate utilities (100% offline).
 * Solar position: NOAA / Meeus algorithms.
 * Geomagnetic Dipole & MLT: IGRF-14 centered dipole (epoch ~2026.78, Laundal & Richmond 2017).
 */

import { geoCircle } from "d3-geo";
import type { LineString, Polygon } from "geojson";

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

export interface SubsolarPoint {
  latDeg: number;
  lonDeg: number;
  declinationDeg: number;
  eotMin: number;
  julianDate: number;
}

export interface DipolePole {
  latDeg: number;
  lonDeg: number;
}

/**
 * Centered dipole North Geomagnetic Pole for IGRF-14 epoch 2026.78
 * (from secularly extrapolated g10 = -29328.6 nT, g11 = -1393.0 nT, h11 = +4507.2 nT).
 */
export const IGRF14_NORTH_POLE: DipolePole = {
  latDeg: 80.86,
  lonDeg: -72.83,
};

/** Wrap longitude in degrees into (-180, 180]. */
export function wrapLon180(lonDeg: number): number {
  const wrapped = ((((lonDeg + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 ? 180 : wrapped;
}

function clampUnit(x: number): number {
  return Math.max(-1, Math.min(1, x));
}

/**
 * Compute solar declination, Equation of Time, and subsolar [lon, lat] at UTC ms.
 */
export function computeSubsolarPoint(utcMs: number): SubsolarPoint {
  const julianDate = utcMs / 86_400_000 + 2_440_587.5;
  const T = (julianDate - 2_451_545.0) / 36_525.0;

  const l0Deg = (((280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360) + 360) % 360;
  const mDeg = 357.52911 + T * (35999.05029 - 0.0001537 * T);
  const mRad = mDeg * DEG2RAD;
  const ecc = 0.016708634 - T * (0.000042037 + 0.0000001267 * T);

  const cDeg =
    Math.sin(mRad) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
    Math.sin(2 * mRad) * (0.019993 - 0.000101 * T) +
    Math.sin(3 * mRad) * 0.000289;

  const sunTrueLonDeg = l0Deg + cDeg;
  const omegaDeg = 125.04 - 1934.136 * T;
  const omegaRad = omegaDeg * DEG2RAD;
  const lambdaAppDeg = sunTrueLonDeg - 0.00569 - 0.00478 * Math.sin(omegaRad);

  const eps0Deg =
    23 +
    (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60;
  const epsDeg = eps0Deg + 0.00256 * Math.cos(omegaRad);
  const epsRad = epsDeg * DEG2RAD;

  const sinDecl = Math.sin(epsRad) * Math.sin(lambdaAppDeg * DEG2RAD);
  const declinationDeg = Math.asin(clampUnit(sinDecl)) * RAD2DEG;

  const y = Math.tan(epsRad * 0.5) ** 2;
  const l0Rad = l0Deg * DEG2RAD;
  const eotMin =
    4 *
    RAD2DEG *
    (y * Math.sin(2 * l0Rad) -
      2 * ecc * Math.sin(mRad) +
      4 * ecc * y * Math.sin(mRad) * Math.cos(2 * l0Rad) -
      0.5 * y * y * Math.sin(4 * l0Rad) -
      1.25 * ecc * ecc * Math.sin(2 * mRad));

  const d = new Date(utcMs);
  const utcMin =
    d.getUTCHours() * 60 +
    d.getUTCMinutes() +
    (d.getUTCSeconds() + d.getUTCMilliseconds() / 1000) / 60;

  const lonDeg = wrapLon180(180 - (utcMin + eotMin) / 4);

  return {
    latDeg: declinationDeg,
    lonDeg,
    declinationDeg,
    eotMin,
    julianDate,
  };
}

/** Solar elevation angle [deg] at geographic (latDeg, lonDeg) and UTC ms. */
export function solarElevationDeg(latDeg: number, lonDeg: number, utcMs: number): number {
  const sub = computeSubsolarPoint(utcMs);
  const phi = latDeg * DEG2RAD;
  const decl = sub.declinationDeg * DEG2RAD;
  const dLon = (lonDeg - sub.lonDeg) * DEG2RAD;
  const sinElev =
    Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(dLon);
  return Math.asin(clampUnit(sinElev)) * RAD2DEG;
}

/** Build a GeoJSON Polygon covering the spherical night hemisphere (90° around antisolar point). */
export function buildNightHemispherePolygon(utcMs: number, precisionDeg = 1): Polygon {
  const sub = computeSubsolarPoint(utcMs);
  const antiLon = wrapLon180(sub.lonDeg + 180);
  const antiLat = -sub.declinationDeg;
  return geoCircle().center([antiLon, antiLat]).radius(90).precision(precisionDeg)() as Polygon;
}

/** Build the spherical day/night terminator LineString (0° solar elevation great circle). */
export function buildTerminatorLine(utcMs: number, stepDeg = 2): LineString {
  const sub = computeSubsolarPoint(utcMs);
  const circle = geoCircle()
    .center([sub.lonDeg, sub.declinationDeg])
    .radius(90)
    .precision(stepDeg)();
  const ring = circle.coordinates[0] ?? [];
  return {
    type: "LineString",
    coordinates: ring.map(([lon, lat]) => [wrapLon180(lon), lat]),
  };
}

type Vec3 = [number, number, number];

interface DipoleBasis {
  xm: Vec3;
  ym: Vec3;
  zm: Vec3;
}

function buildDipoleBasis(pole: DipolePole): DipoleBasis {
  const latP = pole.latDeg * DEG2RAD;
  const lonP = pole.lonDeg * DEG2RAD;
  const zm: Vec3 = [
    Math.cos(latP) * Math.cos(lonP),
    Math.cos(latP) * Math.sin(lonP),
    Math.sin(latP),
  ];
  // y_m = normalize(z_g x z_m) = [-zm_y, zm_x, 0] / norm
  const yRaw: Vec3 = [-zm[1], zm[0], 0];
  const yNorm = Math.hypot(yRaw[0], yRaw[1], yRaw[2]) || 1;
  const ym: Vec3 = [yRaw[0] / yNorm, yRaw[1] / yNorm, yRaw[2] / yNorm];
  // x_m = y_m x z_m
  const xm: Vec3 = [
    ym[1] * zm[2] - ym[2] * zm[1],
    ym[2] * zm[0] - ym[0] * zm[2],
    ym[0] * zm[1] - ym[1] * zm[0],
  ];
  return { xm, ym, zm };
}

const DEFAULT_DIPOLE_BASIS = buildDipoleBasis(IGRF14_NORTH_POLE);

function getBasis(pole?: DipolePole): DipoleBasis {
  if (
    !pole ||
    (pole.latDeg === IGRF14_NORTH_POLE.latDeg && pole.lonDeg === IGRF14_NORTH_POLE.lonDeg)
  ) {
    return DEFAULT_DIPOLE_BASIS;
  }
  return buildDipoleBasis(pole);
}

/** Convert geographic (latDeg, lonDeg) to centered-dipole geomagnetic (mlatDeg, mlonDeg). */
export function geoToMagneticDipole(
  latDeg: number,
  lonDeg: number,
  pole: DipolePole = IGRF14_NORTH_POLE,
): { mlatDeg: number; mlonDeg: number } {
  const { xm, ym, zm } = getBasis(pole);
  const phi = latDeg * DEG2RAD;
  const lam = lonDeg * DEG2RAD;
  const vx = Math.cos(phi) * Math.cos(lam);
  const vy = Math.cos(phi) * Math.sin(lam);
  const vz = Math.sin(phi);

  const mx = xm[0] * vx + xm[1] * vy + xm[2] * vz;
  const my = ym[0] * vx + ym[1] * vy + ym[2] * vz;
  const mz = zm[0] * vx + zm[1] * vy + zm[2] * vz;

  const mlatDeg = Math.atan2(mz, Math.hypot(mx, my)) * RAD2DEG;
  const mlonDeg = Math.hypot(mx, my) < 1e-12 ? 0 : wrapLon180(Math.atan2(my, mx) * RAD2DEG);
  return { mlatDeg, mlonDeg };
}

/** Convert centered-dipole geomagnetic (mlatDeg, mlonDeg) back to geographic [lonDeg, latDeg]. */
export function magneticDipoleToGeo(
  mlatDeg: number,
  mlonDeg: number,
  pole: DipolePole = IGRF14_NORTH_POLE,
): [number, number] {
  const { xm, ym, zm } = getBasis(pole);
  const mlat = mlatDeg * DEG2RAD;
  const mlon = mlonDeg * DEG2RAD;
  const mx = Math.cos(mlat) * Math.cos(mlon);
  const my = Math.cos(mlat) * Math.sin(mlon);
  const mz = Math.sin(mlat);

  const vx = xm[0] * mx + ym[0] * my + zm[0] * mz;
  const vy = xm[1] * mx + ym[1] * my + zm[1] * mz;
  const vz = xm[2] * mx + ym[2] * my + zm[2] * mz;

  const latDeg = Math.atan2(vz, Math.hypot(vx, vy)) * RAD2DEG;
  const lonDeg = Math.hypot(vx, vy) < 1e-12 ? 0 : wrapLon180(Math.atan2(vy, vx) * RAD2DEG);
  return [lonDeg, latDeg];
}

/** Compute centered-dipole Magnetic Local Time (MLT in [0, 24) hours) at UTC ms. */
export function computeMLT(
  latDeg: number,
  lonDeg: number,
  utcMs: number,
  pole: DipolePole = IGRF14_NORTH_POLE,
): number {
  const sub = computeSubsolarPoint(utcMs);
  const stMag = geoToMagneticDipole(latDeg, lonDeg, pole);
  const sunMag = geoToMagneticDipole(sub.latDeg, sub.lonDeg, pole);
  const raw = 12 + (stMag.mlonDeg - sunMag.mlonDeg) / 15;
  return ((raw % 24) + 24) % 24;
}

/** Build a GeoJSON LineString along a constant Magnetic Local Time dipole meridian (e.g. MLT = 12). */
export function buildMltMeridianLine(
  targetMltHours: number,
  utcMs: number,
  stepDeg = 2,
  pole: DipolePole = IGRF14_NORTH_POLE,
): LineString {
  const sub = computeSubsolarPoint(utcMs);
  const sunMag = geoToMagneticDipole(sub.latDeg, sub.lonDeg, pole);
  const targetMlon = wrapLon180(sunMag.mlonDeg + (targetMltHours - 12) * 15);
  const coords: [number, number][] = [];
  const step = Math.max(0.5, stepDeg);
  for (let mlat = -85; mlat <= 89; mlat += step) {
    coords.push(magneticDipoleToGeo(mlat, targetMlon, pole));
  }
  if (coords.length === 0 || Math.abs(89 - (-85)) % step > 1e-6) {
    coords.push(magneticDipoleToGeo(89, targetMlon, pole));
  }
  return {
    type: "LineString",
    coordinates: coords,
  };
}

/** Format MLT decimal hours as HH:MM string. */
export function formatMltHours(mltHours: number): string {
  const norm = ((mltHours % 24) + 24) % 24;
  let hh = Math.floor(norm);
  let mm = Math.round((norm - hh) * 60);
  if (mm >= 60) {
    mm = 0;
    hh = (hh + 1) % 24;
  }
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}
