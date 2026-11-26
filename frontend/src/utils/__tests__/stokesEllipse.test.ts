import { describe, expect, it } from "vitest";
import type { NetworkSnapshotResponse, Stokes2DOut } from "../../api/types";
import {
  applyNetworkSnapshotAlphaMask,
  buildStokesEllipseGeometry,
  computeArcCoherenceMetrics,
  computeEllipseRadius,
  computeSnapshotWindowUtc,
} from "../stokesEllipse";
import { gamma2Crit, pCrit } from "../thresholds";

function makeStokes(overrides: Partial<Stokes2DOut> = {}): Stokes2DOut {
  const nEff = 75;
  const pc = pCrit(nEff, 0.05);
  const gc = gamma2Crit(nEff, 0.05);
  return {
    I: 250,
    Q: 75,
    U: 129.9,
    V: 200,
    A: 150,
    R: 250,
    p: 1.0,
    p_L: 0.6,
    p_C: 0.8,
    psi_deg: 30,
    chi_deg: 26.565,
    epsilon: 0.5,
    gamma2: 1.0,
    arg_sxy_deg: 56.3,
    n_eff: nEff,
    p_crit: pc,
    gamma2_crit: gc,
    is_significant_p: true,
    is_significant_gamma2: true,
    ...overrides,
  };
}

describe("Stokes ellipse geometry, CW/CCW sign, masking, and coherence arcs", () => {
  it("models .docx Case A (epsilon=+0.5, psi=+30, V>0) as CW with rx/ry = 0.5 and +x top arrow", () => {
    const geom = buildStokesEllipseGeometry({
      stationIaga: "LVV",
      available: true,
      isValid: true,
      stokes: makeStokes(),
      coordFrame: "XYZ",
      d0Deg: 6.5,
      localNorthDeg: -4.0,
      scaleMinI: 10,
      scaleMaxI: 500,
    });

    expect(geom.status).toBe("significant");
    expect(geom.sense).toBe("CW");
    expect(geom.minorRadius / geom.majorRadius).toBeCloseTo(0.5, 6);
    expect(geom.psiGeoDeg).toBeCloseTo(30, 6);
    expect(geom.rotationDeg).toBeCloseTo(26.0, 6); // localNorth (-4) + psi (30)
    expect(geom.showArrow).toBe(true);

    // Top tip arrow at (0, -majorRadius) must point right (+x) for CW
    const [tipXStr, tipYStr] = geom.topArrowPoints.split(" ")[0].split(",");
    expect(Number(tipXStr)).toBeGreaterThan(0);
    expect(Number(tipYStr)).toBeCloseTo(-geom.majorRadius, 1);
  });

  it("models .docx Case E Pc6 (epsilon=-0.3846, psi=-50, V<0) as CCW with -x top arrow", () => {
    const geom = buildStokesEllipseGeometry({
      stationIaga: "BEL",
      available: true,
      isValid: true,
      stokes: makeStokes({
        V: -150,
        epsilon: -0.3846,
        psi_deg: -50,
        chi_deg: -21.037,
      }),
      coordFrame: "XYZ",
      d0Deg: 5.0,
      localNorthDeg: 0,
      scaleMinI: 10,
      scaleMaxI: 500,
    });

    expect(geom.status).toBe("significant");
    expect(geom.sense).toBe("CCW");
    expect(geom.minorRadius / geom.majorRadius).toBeCloseTo(0.3846, 4);
    expect(geom.rotationDeg).toBeCloseTo(-50, 6);
    const [tipXStr] = geom.topArrowPoints.split(" ")[0].split(",");
    expect(Number(tipXStr)).toBeLessThan(0);
  });

  it("handles circular (Case 0, psi=null), linear (epsilon=0), XmYm D0 offset, and p <= p_crit masking", () => {
    const circular = buildStokesEllipseGeometry({
      stationIaga: "HRN",
      available: true,
      isValid: true,
      stokes: makeStokes({ Q: 0, U: 0, A: 0, psi_deg: null, epsilon: 1.0, chi_deg: 45 }),
      coordFrame: "XmYm",
      d0Deg: 11.2,
      localNorthDeg: 3.0,
      scaleMinI: 100,
      scaleMaxI: 100,
    });
    expect(circular.sense).toBe("CW");
    expect(circular.minorRadius).toBeCloseTo(circular.majorRadius, 6);
    expect(circular.rotationDeg).toBeCloseTo(14.2, 6);

    const linear = buildStokesEllipseGeometry({
      stationIaga: "OTT",
      available: true,
      isValid: true,
      stokes: makeStokes({ V: 0, epsilon: 0, chi_deg: 0, psi_deg: 15 }),
      coordFrame: "XmYm",
      d0Deg: -13.5,
      localNorthDeg: 0,
      scaleMinI: 10,
      scaleMaxI: 500,
    });
    expect(linear.sense).toBe("LINEAR");
    expect(linear.minorRadius).toBe(0);
    expect(linear.showArrow).toBe(false);
    expect(linear.psiGeoDeg).toBeCloseTo(1.5, 6);

    const masked = buildStokesEllipseGeometry({
      stationIaga: "PAG",
      available: true,
      isValid: true,
      stokes: makeStokes({ p: 0.12, is_significant_p: false }),
      coordFrame: "XYZ",
      d0Deg: 0,
      localNorthDeg: 0,
      scaleMinI: 10,
      scaleMaxI: 500,
    });
    expect(masked.status).toBe("insignificant");
    expect(masked.sense).toBe("NONE");
    expect(masked.majorRadius).toBe(0);

    const absent = buildStokesEllipseGeometry({
      stationIaga: "SUA",
      available: false,
      isValid: false,
      stokes: null,
      coordFrame: "XYZ",
      d0Deg: 0,
      localNorthDeg: 0,
      scaleMinI: 10,
      scaleMaxI: 500,
    });
    expect(absent.status).toBe("unavailable");
  });

  it("re-masks NetworkSnapshotResponse instantaneously when alpha tightens 0.05 -> 0.01", () => {
    const nEff = 45;
    const pc05 = pCrit(nEff, 0.05);
    const pc01 = pCrit(nEff, 0.01);
    const borderlineP = 0.5 * (pc05 + pc01);

    const raw: NetworkSnapshotResponse = {
      episode_id: "storm_20261009_11",
      band: "Pc5",
      coord_frame: "XYZ",
      stations: [
        {
          station_iaga: "LVV",
          lat_deg: 49.9,
          lon_deg: 23.8,
          has_baseline: true,
          available: true,
          is_valid: true,
          d0_deg: 6.2,
          stokes_2d: makeStokes({ n_eff: nEff, p: borderlineP, p_crit: pc05, is_significant_p: true }),
          warning: null,
        },
        {
          station_iaga: "BEL",
          lat_deg: 51.84,
          lon_deg: 20.79,
          has_baseline: true,
          available: true,
          is_valid: true,
          d0_deg: 5.5,
          stokes_2d: makeStokes({ n_eff: nEff, p: 0.9, p_crit: pc05, is_significant_p: true }),
          warning: null,
        },
      ],
    };

    const masked01 = applyNetworkSnapshotAlphaMask(raw, 0.01);
    expect(raw.stations[0].stokes_2d?.is_significant_p).toBe(true); // immutable
    expect(masked01.stations[0].stokes_2d?.p_crit).toBeCloseTo(pc01, 6);
    expect(masked01.stations[0].stokes_2d?.is_significant_p).toBe(false);
    expect(masked01.stations[1].stokes_2d?.is_significant_p).toBe(true);
  });

  it("computes coherence arc m = delta_phi / delta_lon, aliasing bound, and meridional guard", () => {
    // Zonal pair LVV (23.8 E) - BEL (20.789 E) => |delta_lon| = 3.011 > 3.0
    const zonal = computeArcCoherenceMetrics({
      stationI: "LVV",
      latI: 49.9,
      lonI: 23.8,
      stationJ: "BEL",
      latJ: 51.836,
      lonJ: 20.789,
      gamma2: 0.72,
      gamma2Crit: 0.12,
      phaseDeg: 30,
    });
    expect(zonal.isMeridionalGuarded).toBe(false);
    expect(zonal.isSignificant).toBe(true);
    expect(zonal.phaseDegMasked).toBe(30);
    expect(zonal.azimuthalM).toBeCloseTo(30 / 3.011, 2);
    expect(zonal.mAlias).toBeCloseTo(180 / 3.011, 2);

    // Meridional pair LVV (23.8 E) - PEG (23.9 E) => |delta_lon| = 0.1 < 3.0
    const merid = computeArcCoherenceMetrics({
      stationI: "LVV",
      latI: 49.9,
      lonI: 23.8,
      stationJ: "PEG",
      latJ: 38.1,
      lonJ: 23.9,
      gamma2: 0.85,
      gamma2Crit: 0.12,
      phaseDeg: 40,
    });
    expect(merid.isMeridionalGuarded).toBe(true);
    expect(merid.azimuthalM).toBeNull();
    expect(merid.mAlias).toBeNull();

    // Insignificant pair masks phase and m
    const insig = computeArcCoherenceMetrics({
      stationI: "LVV",
      latI: 49.9,
      lonI: 23.8,
      stationJ: "OTT",
      latJ: 45.403,
      lonJ: -75.552,
      gamma2: 0.05,
      gamma2Crit: 0.15,
      phaseDeg: 45,
    });
    expect(insig.isSignificant).toBe(false);
    expect(insig.phaseDegMasked).toBeNull();
    expect(insig.azimuthalM).toBeNull();
  });

  it("scales ellipse radii monotonically and clamps snapshot windows to episode bounds", () => {
    const rLow = computeEllipseRadius(10, 10, 1000, 6, 20);
    const rMid = computeEllipseRadius(100, 10, 1000, 6, 20);
    const rHigh = computeEllipseRadius(1000, 10, 1000, 6, 20);
    expect(rLow).toBeCloseTo(6, 5);
    expect(rHigh).toBeCloseTo(20, 5);
    expect(rMid).toBeGreaterThan(rLow);
    expect(rMid).toBeLessThan(rHigh);

    const b0 = Date.UTC(2026, 9, 4, 0, 0, 0);
    const b1 = Date.UTC(2026, 9, 5, 23, 59, 0);
    const winStart = computeSnapshotWindowUtc(b0 + 30 * 60_000, 360, [b0, b1]);
    expect(Date.parse(winStart.startIso)).toBe(b0);
    expect(Date.parse(winStart.endIso)).toBe(b0 + 360 * 60_000);
  });
});
