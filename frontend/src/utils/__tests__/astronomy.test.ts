import { describe, expect, it } from "vitest";
import mltReference from "../../../../tests/verification/fixtures/mlt_dipole_reference.json";
import {
  IGRF14_NORTH_POLE,
  buildMltMeridianLine,
  buildNightHemispherePolygon,
  buildTerminatorLine,
  computeMLT,
  computeSubsolarPoint,
  formatMltHours,
  geoToMagneticDipole,
  magneticDipoleToGeo,
  solarElevationDeg,
} from "../astronomy";

describe("Astronomy, solar terminator & centered-dipole MLT (offline)", () => {
  it("computes accurate solar declination across solstices, equinoxes, and Oct 10, 2026", () => {
    const juneSolstice = Date.UTC(2026, 5, 21, 12, 0, 0);
    const decSolstice = Date.UTC(2026, 11, 21, 12, 0, 0);
    const vernalEquinox = Date.UTC(2026, 2, 20, 15, 0, 0);
    const stormOct10 = Date.UTC(2026, 9, 10, 12, 0, 0);

    expect(computeSubsolarPoint(juneSolstice).declinationDeg).toBeCloseTo(23.44, 1);
    expect(computeSubsolarPoint(decSolstice).declinationDeg).toBeCloseTo(-23.44, 1);
    expect(Math.abs(computeSubsolarPoint(vernalEquinox).declinationDeg)).toBeLessThan(0.35);

    const sOct = computeSubsolarPoint(stormOct10);
    expect(sOct.declinationDeg).toBeGreaterThan(-7.5);
    expect(sOct.declinationDeg).toBeLessThan(-6.0);
  });

  it("moves subsolar longitude ~90 deg westward every 6 hours UTC", () => {
    const t00 = Date.UTC(2026, 9, 10, 0, 0, 0);
    const t06 = Date.UTC(2026, 9, 10, 6, 0, 0);
    const t12 = Date.UTC(2026, 9, 10, 12, 0, 0);
    const t18 = Date.UTC(2026, 9, 10, 18, 0, 0);

    const lon00 = computeSubsolarPoint(t00).lonDeg;
    const lon06 = computeSubsolarPoint(t06).lonDeg;
    const lon12 = computeSubsolarPoint(t12).lonDeg;
    const lon18 = computeSubsolarPoint(t18).lonDeg;

    const wrapDiff = (a: number, b: number) => ((((a - b + 180) % 360) + 360) % 360) - 180;
    expect(wrapDiff(lon00, lon06)).toBeCloseTo(90, 0);
    expect(wrapDiff(lon06, lon12)).toBeCloseTo(90, 0);
    expect(wrapDiff(lon12, lon18)).toBeCloseTo(90, 0);
  });

  it("yields +90 deg elevation at subsolar, -90 deg at antisolar, and ~0 deg along terminator", () => {
    const utc = Date.UTC(2026, 9, 10, 9, 30, 0);
    const sub = computeSubsolarPoint(utc);
    const antiLon = ((((sub.lonDeg + 180 + 180) % 360) + 360) % 360) - 180;

    expect(solarElevationDeg(sub.latDeg, sub.lonDeg, utc)).toBeCloseTo(90, 3);
    expect(solarElevationDeg(-sub.latDeg, antiLon, utc)).toBeCloseTo(-90, 3);

    const term = buildTerminatorLine(utc, 3);
    expect(term.coordinates.length).toBeGreaterThan(60);
    for (const [lon, lat] of term.coordinates) {
      expect(Math.abs(solarElevationDeg(lat, lon, utc))).toBeLessThan(0.2);
    }

    const nightPoly = buildNightHemispherePolygon(utc, 2);
    expect(nightPoly.type).toBe("Polygon");
    expect(nightPoly.coordinates[0].length).toBeGreaterThan(60);
  });

  it("inverts geo <-> magnetic dipole coordinates to < 1e-6 deg and maps NMP to +90 mlat", () => {
    const poleMag = geoToMagneticDipole(IGRF14_NORTH_POLE.latDeg, IGRF14_NORTH_POLE.lonDeg);
    expect(poleMag.mlatDeg).toBeCloseTo(90, 6);

    const testPts: [number, number][] = [
      [49.9, 23.8], // LVV
      [77.0, 15.55], // HRN
      [48.52, -123.42], // VIC
      [28.3, -16.4], // GUI
      [-35.0, 149.0],
    ];
    for (const [lat, lon] of testPts) {
      const mag = geoToMagneticDipole(lat, lon);
      const [lonRt, latRt] = magneticDipoleToGeo(mag.mlatDeg, mag.mlonDeg);
      expect(latRt).toBeCloseTo(lat, 6);
      expect(lonRt).toBeCloseTo(lon, 6);
    }
  });

  it("pins MLT = 12.00 h along buildMltMeridianLine(12) and separates Europe vs Canada by 6-10 h MLT", () => {
    const utc = Date.UTC(2026, 9, 10, 11, 0, 0);
    const mlt12 = buildMltMeridianLine(12, utc, 4);
    expect(mlt12.coordinates.length).toBeGreaterThan(30);

    for (const [lon, lat] of mlt12.coordinates) {
      const mlt = computeMLT(lat, lon, utc);
      expect(mlt).toBeCloseTo(12.0, 4);
    }

    const mltLvv = computeMLT(49.9, 23.8, utc);
    const mltOtt = computeMLT(45.403, -75.552, utc);
    const mltVic = computeMLT(48.52, -123.42, utc);

    const diffLvvOtt = ((mltLvv - mltOtt) % 24 + 24) % 24;
    const diffLvvVic = ((mltLvv - mltVic) % 24 + 24) % 24;
    expect(diffLvvOtt).toBeGreaterThan(5.5);
    expect(diffLvvOtt).toBeLessThan(8.5);
    expect(diffLvvVic).toBeGreaterThan(8.0);
    expect(diffLvvVic).toBeLessThan(11.5);

    expect(formatMltHours(12)).toBe("12:00");
    expect(formatMltHours(23.999)).toBe("00:00");
  });

  it("matches Python geomag_core.mlt identically across all checkpoints in shared mlt_dipole_reference.json", () => {
    expect(mltReference.model.pole_lat_deg).toBeCloseTo(IGRF14_NORTH_POLE.latDeg, 9);
    expect(mltReference.model.pole_lon_deg).toBeCloseTo(IGRF14_NORTH_POLE.lonDeg, 9);

    for (const c of mltReference.subsolar_cases) {
      const sub = computeSubsolarPoint(Date.parse(c.utc_iso));
      expect(sub.declinationDeg).toBeCloseTo(c.declination_deg, 6);
      expect(sub.lonDeg).toBeCloseTo(c.lon_deg, 6);
      expect(sub.eotMin).toBeCloseTo(c.eot_min, 6);
      expect(sub.julianDate).toBeCloseTo(c.julian_date, 6);
    }

    for (const c of mltReference.dipole_cases) {
      const mag = geoToMagneticDipole(c.lat_deg, c.lon_deg);
      expect(mag.mlatDeg).toBeCloseTo(c.mlat_deg, 6);
      expect(mag.mlonDeg).toBeCloseTo(c.mlon_deg, 6);
      const [lonRt, latRt] = magneticDipoleToGeo(mag.mlatDeg, mag.mlonDeg);
      expect(latRt).toBeCloseTo(c.lat_deg, 6);
      if (Math.abs(c.lat_deg) < 89.99) {
        expect(lonRt).toBeCloseTo(c.lon_deg, 6);
      }
    }

    for (const c of mltReference.mlt_cases) {
      const utcMs = Date.parse(c.utc_iso);
      const mltH = computeMLT(c.lat_deg, c.lon_deg, utcMs);
      const elev = solarElevationDeg(c.lat_deg, c.lon_deg, utcMs);
      expect(mltH).toBeCloseTo(c.mlt_hours, 6);
      expect(mltH < 12 ? "morning" : "afternoon").toBe(c.sector);
      expect(elev).toBeCloseTo(c.solar_elevation_deg, 6);
    }
  });
});
