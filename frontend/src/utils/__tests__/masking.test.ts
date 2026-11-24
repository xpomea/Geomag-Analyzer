import { describe, expect, it } from "vitest";
import type { PolarizationResponse, WelchPairResponse } from "../../api/types";
import { applyPolarizationAlphaMask, applyWelchAlphaMask } from "../masking";
import { gamma2Crit, pCrit } from "../thresholds";

function buildSyntheticPolarization(nWin = 6): PolarizationResponse {
  const nEff = 75;
  const p05 = pCrit(nEff, 0.05);
  const g05 = gamma2Crit(nEff, 0.05);
  // Windows chosen to bracket thresholds at alpha = 0.10, 0.05, 0.01:
  // pCrit(75, 0.10) ~ 0.205, pCrit(75, 0.05) ~ 0.227, pCrit(75, 0.01) ~ 0.271
  const pVals = [0.15, 0.215, 0.25, 0.45, 0.24, null];
  const validFlags = [true, true, true, true, true, false];
  const ts = Array.from({ length: nWin }, (_, i) =>
    new Date(Date.UTC(2026, 9, 10, i, 0, 0)).toISOString(),
  );

  return {
    episode_id: "storm_20261009_11",
    station_iaga: "LVV",
    method: "welch",
    band: "Pc5",
    coord_frame: "XYZ",
    d0_deg: 0,
    is_valid: true,
    n_eff: 395,
    trim_hours: null,
    stokes_2d: {
      I: 100,
      Q: 20,
      U: 10,
      V: 12,
      A: 22.36,
      R: 25.38,
      p: 0.25,
      p_L: 0.22,
      p_C: 0.12,
      psi_deg: 13.3,
      chi_deg: 14.0,
      epsilon: 0.25,
      gamma2: 0.1,
      arg_sxy_deg: 50,
      n_eff: nEff,
      p_crit: p05,
      gamma2_crit: g05,
      is_significant_p: true,
      is_significant_gamma2: true,
    },
    pol_3d: null,
    dynamic: {
      timestamps_utc: ts,
      I: ts.map((_, i) => (validFlags[i] ? 50 + i * 10 : null)),
      p: pVals.slice(0, nWin),
      p_crit: validFlags.map((ok) => (ok ? p05 : null)),
      psi_deg: validFlags.map((ok, i) => (ok ? 25 + i : null)),
      chi_deg: validFlags.map((ok, i) => (ok ? 10 + i : null)),
      epsilon: validFlags.map((ok, i) => (ok ? 0.3 + i * 0.02 : null)),
      gamma2: validFlags.map((ok) => (ok ? 0.2 : null)),
      gamma2_crit: validFlags.map((ok) => (ok ? g05 : null)),
      n_eff: validFlags.map((ok) => (ok ? nEff : null)),
      is_significant_p: pVals.map((v, i) => validFlags[i] && v !== null && v > p05),
      is_valid: validFlags,
    },
    filtered: null,
    spectrogram: null,
    warnings: [],
  };
}

describe("client dynamic alpha masking (p < p_crit & gamma2 < gamma2_crit)", () => {
  it("remasks dynamic angles and summary KPI when alpha switches between 0.10, 0.05, 0.01", () => {
    const raw = buildSyntheticPolarization();

    const m10 = applyPolarizationAlphaMask(raw, 0.1);
    const m05 = applyPolarizationAlphaMask(raw, 0.05);
    const m01 = applyPolarizationAlphaMask(raw, 0.01);

    // Summary p = 0.25 at n_eff = 75: significant at 0.05, insignificant at 0.01
    expect(m05.stokes2d?.is_significant_p).toBe(true);
    expect(m01.stokes2d?.is_significant_p).toBe(false);

    // Window 1 (p=0.215): significant at 0.10, masked at 0.05 and 0.01
    expect(m10.dynamic?.is_significant_p[1]).toBe(true);
    expect(m10.psiSig[1]).not.toBeNull();
    expect(m10.psiNoiseGhost[1]).toBeNull();

    expect(m05.dynamic?.is_significant_p[1]).toBe(false);
    expect(m05.psiSig[1]).toBeNull();
    expect(m05.chiSig[1]).toBeNull();
    expect(m05.epsSig[1]).toBeNull();
    expect(m05.psiNoiseGhost[1]).toBe(26);

    // Windows 2 & 4 (p=0.25, 0.24): significant at 0.05, masked at 0.01
    expect(m05.psiSig[2]).toBe(27);
    expect(m05.psiSig[4]).toBe(29);
    expect(m01.psiSig[2]).toBeNull();
    expect(m01.psiSig[4]).toBeNull();

    // Window 3 (p=0.45): always significant
    expect(m01.psiSig[3]).toBe(28);

    // Noise intervals count merges contiguous noise windows
    expect(m10.noiseIntervals.length).toBe(1); // only window 0
    expect(m05.noiseIntervals.length).toBe(1); // windows 0+1 contiguous
    expect(m01.noiseIntervals.length).toBe(2); // [0,1,2] and [4]
  });

  it("remasks Welch 1D phase, azimuthal m, and 2D TF phase in < 50 ms on full storm size", () => {
    const nWin = 288;
    const nFreq = 39;
    const nEff = 45;
    const gc05 = gamma2Crit(nEff, 0.05);
    const ts = Array.from({ length: nWin }, (_, i) =>
      new Date(Date.UTC(2026, 9, 10, 0, 0, 0) + i * 30 * 60_000).toISOString(),
    );

    const welch: WelchPairResponse = {
      episode_id: "storm_20261009_11",
      station_i: "LVV",
      station_j: "BEL",
      component: "X",
      coord_frame: "XYZ",
      band: "Pc5",
      available: true,
      cache_hits: { LVV: true, BEL: true },
      window_timestamps_utc: ts,
      freqs_hz: Array.from({ length: nFreq }, (_, k) => 0.0017 + k * 0.00012),
      K_ij: Array(nWin).fill(4),
      n_eff: Array(nWin).fill(nEff),
      gamma2: ts.map((_, i) => (i % 2 === 0 ? 0.08 : 0.22)),
      gamma2_crit: Array(nWin).fill(gc05),
      phase_deg: ts.map((_, i) => (i % 2 === 0 ? 45 : -60)),
      phase_deg_masked: [],
      is_significant: [],
      is_valid: Array(nWin).fill(true),
      gamma2_tf: ts.map(() =>
        Array.from({ length: nFreq }, (_, f) => (f < 15 ? 0.2 : f < 25 ? 0.7 : 0.85)),
      ),
      gamma2_crit_tf: Array(nWin).fill(gamma2Crit(4, 0.05)),
      phase_deg_tf: ts.map(() => Array.from({ length: nFreq }, (_, f) => -90 + f * 4)),
      phase_deg_tf_masked: [],
      azimuthal: {
        m: Array(nWin).fill(4.5),
        m_alias: 18,
        delta_lon_deg: 10,
        is_meridional_guarded: false,
      },
      warnings: [],
    };

    const t0 = performance.now();
    const res05 = applyWelchAlphaMask(welch, 0.05, 0.05);
    const res01 = applyWelchAlphaMask(welch, 0.01, 0.05);
    const elapsedMs = performance.now() - t0;

    expect(elapsedMs).toBeLessThan(50);
    // For nEff=45: gamma2_crit(45, 0.05) ~ 0.0658, gamma2_crit(45, 0.01) ~ 0.0994
    // Window with gamma2=0.08 is significant at 0.05 but masked at 0.01
    expect(res05.isSignificant[0]).toBe(true);
    expect(res05.phaseDegMasked[0]).toBe(45);
    expect(res05.azimuthalMMasked[0]).toBeCloseTo(4.5, 6);

    expect(res01.isSignificant[0]).toBe(false);
    expect(res01.phaseDegMasked[0]).toBeNull();
    expect(res01.azimuthalMMasked[0]).toBeNull();
    expect(res01.isSignificant[1]).toBe(true);

    // 2D TF phase (K=4: g2_crit(0.05)~0.6316, g2_crit(0.01)~0.7846)
    expect(res05.phaseDegTfMasked[0][0]).toBeNull(); // 0.20 < 0.6316
    expect(res05.phaseDegTfMasked[0][20]).toBe(-90 + 20 * 4); // 0.70 > 0.6316
    expect(res01.phaseDegTfMasked[0][20]).toBeNull(); // 0.70 < 0.7846
    expect(res01.phaseDegTfMasked[0][35]).toBe(-90 + 35 * 4); // 0.85 > 0.7846
  });

  it("reveals and unwraps azimuthal m when relaxing alpha to 0.10 even when backend m was null at 0.05", () => {
    const nWin = 3;
    const nEff = 45; // gamma2Crit(45, 0.05) ~ 0.0658, gamma2Crit(45, 0.10) ~ 0.0510
    const ts = [
      "2026-10-10T01:00:00Z",
      "2026-10-10T02:00:00Z",
      "2026-10-10T03:00:00Z",
    ];
    const welch: WelchPairResponse = {
      episode_id: "storm_20261009_11",
      station_i: "LVV",
      station_j: "OTT",
      component: "X",
      coord_frame: "XYZ",
      band: "Pc5",
      available: true,
      cache_hits: { LVV: true, OTT: true },
      window_timestamps_utc: ts,
      freqs_hz: [],
      K_ij: [4, 4, 4],
      n_eff: [nEff, nEff, nEff],
      // Window 0 (0.060) fails alpha=0.05 but passes alpha=0.10
      gamma2: [0.06, 0.2, 0.25],
      gamma2_crit: Array(nWin).fill(gamma2Crit(nEff, 0.05)),
      phase_deg: [170, -170, -150],
      phase_deg_masked: [null, -170, -150],
      is_significant: [false, true, true],
      is_valid: [true, true, true],
      gamma2_tf: [],
      gamma2_crit_tf: [],
      phase_deg_tf: [],
      phase_deg_tf_masked: [],
      azimuthal: {
        // Backend nulled index 0 at alpha=0.05; client recomputes from phase_deg!
        m: [null, -17, -15],
        m_alias: 18,
        delta_lon_deg: 10,
        is_meridional_guarded: false,
      },
      warnings: [],
    };

    const at05 = applyWelchAlphaMask(welch, 0.05, 0.05);
    expect(at05.azimuthalMMasked[0]).toBeNull();
    expect(at05.azimuthalMMasked[1]).toBeCloseTo(-17, 6);
    expect(at05.azimuthalMMasked[2]).toBeCloseTo(-15, 6);

    const at10 = applyWelchAlphaMask(welch, 0.1, 0.05);
    // All 3 windows significant -> 170 deg unwrapped across +/-180 -> 170, 190, 210 deg => m = 17, 19, 21
    expect(at10.isSignificant).toEqual([true, true, true]);
    expect(at10.phaseDegMasked[0]).toBe(170);
    expect(at10.azimuthalMMasked[0]).toBeCloseTo(17, 6);
    expect(at10.azimuthalMMasked[1]).toBeCloseTo(19, 6);
    expect(at10.azimuthalMMasked[2]).toBeCloseTo(21, 6);
  });
});
