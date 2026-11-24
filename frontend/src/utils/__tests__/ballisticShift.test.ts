import { describe, expect, it } from "vitest";
import type { Solar1Series } from "../../api/types";
import {
  applySolar1Shift,
  computeBallisticArrivalSec,
  resampleMonotonicSeries,
} from "../ballisticShift";

describe("ballisticShift (L1 -> BSN propagation)", () => {
  it("computes exact smoothed ballistic delay and clamps extreme speeds to [200, 2000]", () => {
    const n = 20;
    const tSec = Array.from({ length: n }, (_, i) => 1_700_000_000 + i * 60);
    const vx = Array.from({ length: n }, () => -500);

    const { arrivalSec, effectiveDelayMin } = computeBallisticArrivalSec(tSec, vx, {
      refDistanceKm: 1_400_000,
      smoothWindowMin: 5,
    });
    const expectedMin = 1_400_000 / 500 / 60;
    for (let i = 0; i < n; i++) {
      expect(effectiveDelayMin[i]).toBeCloseTo(expectedMin, 6);
      if (i > 0) {
        expect(arrivalSec[i]).toBeGreaterThan(arrivalSec[i - 1]);
      }
    }

    // Clamp low speed (-80 -> 200 km/s) and high speed (-5000 -> 2000 km/s)
    const lowClamp = computeBallisticArrivalSec(tSec, Array(n).fill(-80), {
      refDistanceKm: 1_400_000,
      smoothWindowMin: 1,
    });
    expect(lowClamp.effectiveDelayMin[0]).toBeCloseTo(1_400_000 / 200 / 60, 6);

    const highClamp = computeBallisticArrivalSec(tSec, Array(n).fill(-5000), {
      refDistanceKm: 1_400_000,
      smoothWindowMin: 1,
    });
    expect(highClamp.effectiveDelayMin[0]).toBeCloseTo(1_400_000 / 2000 / 60, 6);
  });

  it("enforces strict forward monotonicity across abrupt shock velocity jumps", () => {
    const n = 120;
    const tSec = Array.from({ length: n }, (_, i) => i * 60);
    // Sharp CME shock: speed jumps from 350 km/s to 950 km/s at t = 60 min (delay drops ~67 -> ~25 min)
    const vx = Array.from({ length: n }, (_, i) => (i < 60 ? -350 : -950));

    const { arrivalSec } = computeBallisticArrivalSec(tSec, vx, {
      refDistanceKm: 1_400_000,
      smoothWindowMin: 1,
    });

    for (let i = 1; i < n; i++) {
      expect(arrivalSec[i]).toBeGreaterThan(arrivalSec[i - 1]);
    }
  });

  it("falls back to 450 km/s when all speeds are missing and never bridges null gaps", () => {
    const tSec = [0, 60, 120, 180, 240];
    const allNullVx = [null, null, null, null, null];
    const { effectiveDelayMin } = computeBallisticArrivalSec(tSec, allNullVx, {
      refDistanceKm: 1_400_000,
      smoothWindowMin: 3,
    });
    expect(effectiveDelayMin[0]).toBeCloseTo(1_400_000 / 450 / 60, 6);

    const shifted = resampleMonotonicSeries([0, 60, 120, 180], [10, null, null, 40], [30, 60, 120, 150, 180]);
    expect(shifted[0]).toBeNull(); // brackets 10 & null -> do not bridge
    expect(shifted[1]).toBeNull(); // exact null sample -> null
    expect(shifted[3]).toBeNull(); // brackets null & 40 -> null
    expect(shifted[4]).toBe(40); // exact finite sample -> 40
  });

  it("shifts rawSolar1 in manual lag mode cleanly and preserves identity in l1_raw", () => {
    const n = 10;
    const ts = Array.from({ length: n }, (_, i) =>
      new Date(Date.UTC(2026, 9, 10, 0, i, 0)).toISOString(),
    );
    const raw: Solar1Series = {
      timestamps_utc: ts,
      time_frame: "L1",
      shift_mode: "l1_raw",
      ref_distance_km: 1_400_000,
      propagation_delay_min: Array(n).fill(0),
      series: {
        bz_gsm_nT: ts.map((_, i) => (i === 2 ? -15 : 0)),
        vx_gse_kms: Array(n).fill(-500),
        proton_n_cm3: Array(n).fill(8),
      },
      swips_flags: Array(n).fill(0),
      is_swips_valid: Array(n).fill(true),
      warning: "",
    };

    const idRes = applySolar1Shift(raw, "l1_raw");
    expect(idRes.solar1.series.bz_gsm_nT[2]).toBe(-15);
    expect(idRes.meanDelayMin).toBe(0);

    const manRes = applySolar1Shift(raw, "manual", { manualLagMin: 3 });
    expect(manRes.meanDelayMin).toBeCloseTo(3, 6);
    expect(manRes.solar1.series.bz_gsm_nT[2]).toBeNull(); // before arrival
    expect(manRes.solar1.series.bz_gsm_nT[5]).toBe(-15); // index 2 + 3 min lag
  });
});
