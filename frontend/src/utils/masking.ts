import type {
  DynamicPolarizationOut,
  MeridionalFLRResponse,
  PolarizationResponse,
  Stokes2DOut,
  WelchPairResponse,
} from "../api/types";
import { gamma2Crit, pCrit, scaleGamma2CritForAlpha } from "./thresholds";

/**
 * Extract contiguous [startMs, endMs] intervals where mask[i] is true
 * for rendering ECharts markArea spans cleanly.
 */
export function extractContiguousRanges(
  timestampsMs: number[],
  mask: boolean[],
  stepMs?: number,
): [number, number][] {
  const n = Math.min(timestampsMs.length, mask.length);
  if (n === 0) return [];

  let inferredStep = stepMs ?? 0;
  if (!(inferredStep > 0) && n > 1) {
    for (let i = 1; i < n; i++) {
      const dt = timestampsMs[i] - timestampsMs[i - 1];
      if (dt > 0) {
        inferredStep = dt;
        break;
      }
    }
  }
  const halfStep = inferredStep > 0 ? inferredStep * 0.5 : 30_000;

  const ranges: [number, number][] = [];
  let inRun = false;
  let runStart = 0;

  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      if (!inRun) {
        inRun = true;
        runStart = i;
      }
    } else if (inRun) {
      inRun = false;
      const t0 = timestampsMs[runStart];
      const t1 = timestampsMs[i - 1];
      ranges.push(runStart === i - 1 ? [t0 - halfStep, t0 + halfStep] : [t0, t1]);
    }
  }

  if (inRun) {
    const t0 = timestampsMs[runStart];
    const t1 = timestampsMs[n - 1];
    ranges.push(runStart === n - 1 ? [t0 - halfStep, t0 + halfStep] : [t0, t1]);
  }

  return ranges;
}

export interface MaskedPolarizationView {
  stokes2d: Stokes2DOut | null;
  dynamic: DynamicPolarizationOut | null;
  psiSig: (number | null)[];
  psiNoiseGhost: (number | null)[];
  chiSig: (number | null)[];
  chiNoiseGhost: (number | null)[];
  epsSig: (number | null)[];
  epsNoiseGhost: (number | null)[];
  noiseIntervals: [number, number][];
}

export function applyPolarizationAlphaMask(
  resp: PolarizationResponse,
  alpha: number,
): MaskedPolarizationView {
  const threshMemo = new Map<number, { pc: number; gc: number }>();
  const getCrit = (nEff: number) => {
    let cached = threshMemo.get(nEff);
    if (!cached) {
      cached = {
        pc: pCrit(nEff, alpha),
        gc: gamma2Crit(nEff, alpha),
      };
      threshMemo.set(nEff, cached);
    }
    return cached;
  };

  let stokes2d: Stokes2DOut | null = resp.stokes_2d ? { ...resp.stokes_2d } : null;
  if (stokes2d) {
    const nEff = stokes2d.n_eff ?? resp.n_eff;
    if (nEff !== null && nEff !== undefined && nEff > 1) {
      const { pc, gc } = getCrit(nEff);
      stokes2d.p_crit = pc;
      stokes2d.gamma2_crit = gc;
      stokes2d.is_significant_p = stokes2d.p !== null ? stokes2d.p > pc : false;
      stokes2d.is_significant_gamma2 =
        stokes2d.gamma2 !== null ? stokes2d.gamma2 > gc : false;
    }
  }

  if (!resp.dynamic) {
    return {
      stokes2d,
      dynamic: null,
      psiSig: [],
      psiNoiseGhost: [],
      chiSig: [],
      chiNoiseGhost: [],
      epsSig: [],
      epsNoiseGhost: [],
      noiseIntervals: [],
    };
  }

  const dyn = resp.dynamic;
  const nWin = dyn.timestamps_utc.length;
  const pCritArr = new Array<number | null>(nWin);
  const g2CritArr = new Array<number | null>(nWin);
  const sigArr = new Array<boolean>(nWin);
  const psiSig = new Array<number | null>(nWin);
  const psiNoiseGhost = new Array<number | null>(nWin);
  const chiSig = new Array<number | null>(nWin);
  const chiNoiseGhost = new Array<number | null>(nWin);
  const epsSig = new Array<number | null>(nWin);
  const epsNoiseGhost = new Array<number | null>(nWin);
  const noiseMask = new Array<boolean>(nWin);
  const timestampsMs = new Array<number>(nWin);

  for (let i = 0; i < nWin; i++) {
    timestampsMs[i] = Date.parse(dyn.timestamps_utc[i]);
    const valid = Boolean(dyn.is_valid[i]);
    const pVal = dyn.p[i];
    const nEff = dyn.n_eff[i];

    let pc: number | null = dyn.p_crit[i] ?? null;
    let gc: number | null = dyn.gamma2_crit[i] ?? null;
    if (nEff !== null && nEff !== undefined && nEff > 1) {
      const crit = getCrit(nEff);
      pc = crit.pc;
      gc = crit.gc;
    }

    pCritArr[i] = valid ? pc : null;
    g2CritArr[i] = valid ? gc : null;

    const sig = Boolean(
      valid &&
        pVal !== null &&
        pVal !== undefined &&
        pc !== null &&
        pVal > pc,
    );
    sigArr[i] = sig;

    const psiVal = valid ? dyn.psi_deg[i] : null;
    const chiVal = valid ? dyn.chi_deg[i] : null;
    const epsVal = valid ? dyn.epsilon[i] : null;

    psiSig[i] = sig ? psiVal : null;
    psiNoiseGhost[i] = valid && !sig ? psiVal : null;
    chiSig[i] = sig ? chiVal : null;
    chiNoiseGhost[i] = valid && !sig ? chiVal : null;
    epsSig[i] = sig ? epsVal : null;
    epsNoiseGhost[i] = valid && !sig ? epsVal : null;
    noiseMask[i] = valid && !sig;
  }

  const maskedDynamic: DynamicPolarizationOut = {
    ...dyn,
    p_crit: pCritArr,
    gamma2_crit: g2CritArr,
    is_significant_p: sigArr,
  };

  return {
    stokes2d,
    dynamic: maskedDynamic,
    psiSig,
    psiNoiseGhost,
    chiSig,
    chiNoiseGhost,
    epsSig,
    epsNoiseGhost,
    noiseIntervals: extractContiguousRanges(timestampsMs, noiseMask),
  };
}

export interface MaskedWelchView {
  gamma2Crit: (number | null)[];
  isSignificant: boolean[];
  phaseDegMasked: (number | null)[];
  azimuthalMMasked: (number | null)[];
  gamma2CritTf: (number | null)[];
  phaseDegTfMasked: (number | null)[][];
}

export function applyWelchAlphaMask(
  resp: WelchPairResponse,
  alpha: number,
  initialAlpha = 0.05,
): MaskedWelchView {
  const threshMemo = new Map<number, number>();
  const nWin = resp.window_timestamps_utc.length;
  const gamma2CritArr = new Array<number | null>(nWin);
  const isSigArr = new Array<boolean>(nWin);
  const phaseMasked = new Array<number | null>(nWin);
  const mGuarded = Boolean(resp.azimuthal?.is_meridional_guarded);
  const deltaLonDeg = resp.azimuthal?.delta_lon_deg ?? 0;
  const deltaLonRad = (deltaLonDeg * Math.PI) / 180;
  const mMasked = new Array<number | null>(nWin).fill(null);

  for (let i = 0; i < nWin; i++) {
    const valid = Boolean(resp.is_valid[i]);
    const g2 = resp.gamma2[i];
    const nEff = resp.n_eff[i];

    let gc: number | null = resp.gamma2_crit[i] ?? null;
    if (nEff !== null && nEff !== undefined && nEff > 1) {
      let cached = threshMemo.get(nEff);
      if (cached === undefined) {
        cached = gamma2Crit(nEff, alpha);
        threshMemo.set(nEff, cached);
      }
      gc = cached;
    }

    gamma2CritArr[i] = valid ? gc : null;
    const sig = Boolean(valid && g2 !== null && gc !== null && g2 > gc);
    isSigArr[i] = sig;
    phaseMasked[i] = sig ? (resp.phase_deg[i] ?? null) : null;
  }

  // Recompute azimuthal wave number m directly from unmasked phase_deg over significant segments
  if (!mGuarded && Math.abs(deltaLonRad) > 1e-12) {
    let segStart = -1;
    for (let i = 0; i <= nWin; i++) {
      const pDeg = i < nWin ? resp.phase_deg[i] : null;
      const inSeg =
        i < nWin && isSigArr[i] && pDeg !== null && pDeg !== undefined && Number.isFinite(pDeg);
      if (inSeg) {
        if (segStart === -1) segStart = i;
      } else if (segStart !== -1) {
        let prevRaw = ((resp.phase_deg[segStart] as number) * Math.PI) / 180;
        let prevUnwrapped = prevRaw;
        mMasked[segStart] = prevUnwrapped / deltaLonRad;
        for (let k = segStart + 1; k < i; k++) {
          const curRaw = ((resp.phase_deg[k] as number) * Math.PI) / 180;
          const rawDiff = curRaw - prevRaw;
          let wrappedDiff =
            (((rawDiff + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
          if (wrappedDiff === -Math.PI && rawDiff > 0) {
            wrappedDiff = Math.PI;
          }
          prevUnwrapped += wrappedDiff;
          prevRaw = curRaw;
          mMasked[k] = prevUnwrapped / deltaLonRad;
        }
        segStart = -1;
      }
    }
  }

  const gamma2CritTf = resp.gamma2_crit_tf.map((gc0) =>
    gc0 !== null && gc0 !== undefined
      ? scaleGamma2CritForAlpha(gc0, initialAlpha, alpha)
      : null,
  );

  const hasRawPhaseTf = resp.phase_deg_tf && resp.phase_deg_tf.length === nWin;
  const phaseDegTfMasked: (number | null)[][] = hasRawPhaseTf
    ? resp.gamma2_tf.map((row, tIdx) => {
        const gc = gamma2CritTf[tIdx];
        const phaseRow = resp.phase_deg_tf[tIdx];
        return row.map((gVal, fIdx) => {
          if (gVal !== null && gc !== null && gVal > gc) {
            return phaseRow[fIdx] ?? null;
          }
          return null;
        });
      })
    : resp.phase_deg_tf_masked;

  return {
    gamma2Crit: gamma2CritArr,
    isSignificant: isSigArr,
    phaseDegMasked: phaseMasked,
    azimuthalMMasked: mMasked,
    gamma2CritTf,
    phaseDegTfMasked,
  };
}

export function applyFlrAlphaMask(
  flr: MeridionalFLRResponse,
  alpha: number,
  initialAlpha = 0.05,
): MeridionalFLRResponse {
  const nPairs = flr.pair_coherence.length;
  const nextCrit: (number | null)[] = new Array(nPairs);
  const nextSig: boolean[] = new Array(nPairs);
  const nextCum: (number | null)[] = new Array(flr.station_codes.length).fill(null);
  const nextGrad: (number | null)[] = new Array(nPairs).fill(null);

  if (nextCum.length > 0) {
    nextCum[0] = 0;
  }

  for (let i = 0; i < nPairs; i++) {
    const gc0 = flr.pair_gamma2_crit[i];
    const gc =
      gc0 !== null && gc0 !== undefined && Number.isFinite(gc0)
        ? scaleGamma2CritForAlpha(gc0, initialAlpha, alpha)
        : null;
    nextCrit[i] = gc;

    const g2 = flr.pair_coherence[i];
    const ph = flr.pair_phase_deg[i];
    const sig = Boolean(
      g2 !== null &&
        g2 !== undefined &&
        Number.isFinite(g2) &&
        gc !== null &&
        Number.isFinite(gc) &&
        ph !== null &&
        ph !== undefined &&
        Number.isFinite(ph) &&
        g2 > gc,
    );
    nextSig[i] = sig;

    const dLat = (flr.latitudes_deg[i + 1] ?? 0) - (flr.latitudes_deg[i] ?? 0);
    if (sig && ph !== null && ph !== undefined && Number.isFinite(ph)) {
      if (i + 1 < nextCum.length && nextCum[i] !== null && Number.isFinite(nextCum[i])) {
        nextCum[i + 1] = (nextCum[i] as number) - ph;
      }
      if (Math.abs(dLat) > 1e-6) {
        nextGrad[i] = -ph / dLat;
      }
    }
  }

  return {
    ...flr,
    pair_gamma2_crit: nextCrit,
    pair_is_significant: nextSig,
    cumulative_phase_deg: nextCum,
    phase_gradient_deg_per_lat: nextGrad,
  };
}
