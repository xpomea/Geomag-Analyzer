import type { Solar1Series, Solar1ShiftMode } from "../api/types";

export interface BallisticShiftOptions {
  refDistanceKm?: number;
  smoothWindowMin?: number;
}

export interface BallisticShiftResult {
  arrivalSec: Float64Array;
  effectiveDelayMin: Float64Array;
}

function medianOfSorted(sorted: Float64Array): number {
  const len = sorted.length;
  if (len === 0) return Number.NaN;
  const mid = len >> 1;
  if ((len & 1) === 1) {
    return sorted[mid];
  }
  return 0.5 * (sorted[mid - 1] + sorted[mid]);
}

function rollingCenteredMedian(values: Float64Array, windowSize: number): Float64Array {
  const n = values.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  const w = Math.max(1, Math.floor(windowSize));
  const halfLeft = Math.floor(w / 2);
  const halfRight = w - 1 - halfLeft;

  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - halfLeft);
    const hi = Math.min(n - 1, i + halfRight);
    const finiteVals: number[] = [];
    for (let j = lo; j <= hi; j++) {
      const v = values[j];
      if (Number.isFinite(v)) {
        finiteVals.push(v);
      }
    }
    if (finiteVals.length === 0) {
      out[i] = Number.NaN;
    } else {
      finiteVals.sort((a, b) => a - b);
      out[i] = medianOfSorted(Float64Array.from(finiteVals));
    }
  }
  return out;
}

/**
 * Exact client-side equivalent of backend geomag_api/data_store.py::compute_ballistic_shift:
 * smoothed speed V_x, clamped to [200, 2000] km/s, with strict forward monotonicity (+1 s step).
 */
export function computeBallisticArrivalSec(
  timestampsSec: ArrayLike<number>,
  vxKms: ArrayLike<number | null | undefined>,
  options: BallisticShiftOptions = {},
): BallisticShiftResult {
  const refDistanceKm = options.refDistanceKm ?? 1_400_000;
  const smoothWindowMin = options.smoothWindowMin ?? 31;

  if (!Number.isFinite(refDistanceKm) || refDistanceKm <= 0) {
    throw new Error(`refDistanceKm must be positive, got ${refDistanceKm}`);
  }
  if (!Number.isFinite(smoothWindowMin) || smoothWindowMin < 1) {
    throw new Error(`smoothWindowMin must be >= 1, got ${smoothWindowMin}`);
  }

  const n = timestampsSec.length;
  const arrivalSec = new Float64Array(n);
  const effectiveDelayMin = new Float64Array(n);
  if (n === 0) {
    return { arrivalSec, effectiveDelayMin };
  }

  const finiteIdx: number[] = [];
  const finiteSpeed: number[] = [];
  for (let i = 0; i < n; i++) {
    const raw = vxKms[i];
    if (raw !== null && raw !== undefined && Number.isFinite(raw) && Math.abs(raw) > 10.0) {
      finiteIdx.push(i);
      finiteSpeed.push(Math.abs(raw));
    }
  }

  const speedFilled = new Float64Array(n);
  if (finiteIdx.length === 0) {
    speedFilled.fill(450.0);
  } else {
    let ptr = 0;
    for (let i = 0; i < n; i++) {
      if (i <= finiteIdx[0]) {
        speedFilled[i] = finiteSpeed[0];
      } else if (i >= finiteIdx[finiteIdx.length - 1]) {
        speedFilled[i] = finiteSpeed[finiteIdx.length - 1];
      } else {
        while (ptr + 1 < finiteIdx.length && finiteIdx[ptr + 1] < i) {
          ptr++;
        }
        const i0 = finiteIdx[ptr];
        const i1 = finiteIdx[ptr + 1];
        const frac = (i - i0) / (i1 - i0);
        speedFilled[i] = finiteSpeed[ptr] * (1 - frac) + finiteSpeed[ptr + 1] * frac;
      }
    }
  }

  const speedSmooth = rollingCenteredMedian(speedFilled, smoothWindowMin);
  for (let i = 0; i < n; i++) {
    const clamped = Math.min(2000.0, Math.max(200.0, speedSmooth[i]));
    const delaySec = refDistanceKm / clamped;
    arrivalSec[i] = timestampsSec[i] + delaySec;
  }

  for (let i = 1; i < n; i++) {
    if (arrivalSec[i] <= arrivalSec[i - 1]) {
      arrivalSec[i] = arrivalSec[i - 1] + 1.0;
    }
  }

  for (let i = 0; i < n; i++) {
    effectiveDelayMin[i] = (arrivalSec[i] - timestampsSec[i]) / 60.0;
  }

  return { arrivalSec, effectiveDelayMin };
}

/**
 * Interpolate a strictly increasing shifted time series onto target grid timestamps
 * without bridging across missing (null/NaN) intervals.
 */
export function resampleMonotonicSeries(
  tArrSec: ArrayLike<number>,
  values: ArrayLike<number | null | undefined>,
  tGridSec: ArrayLike<number>,
): (number | null)[] {
  const nSrc = tArrSec.length;
  const nGrid = tGridSec.length;
  const out = new Array<number | null>(nGrid).fill(null);
  if (nSrc === 0 || nGrid === 0) {
    return out;
  }

  const first = tArrSec[0];
  const last = tArrSec[nSrc - 1];

  for (let g = 0; g < nGrid; g++) {
    const t = tGridSec[g];
    if (t < first || t > last) {
      continue;
    }

    let lo = 0;
    let hi = nSrc - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (tArrSec[mid] <= t) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }

    const i0 = lo;
    const i1 = Math.min(nSrc - 1, i0 + 1);
    const t0 = tArrSec[i0];
    const t1 = tArrSec[i1];
    const frac = i1 === i0 || t1 === t0 ? 0 : (t - t0) / (t1 - t0);

    const v0 = values[i0];
    const v1 = values[i1];
    const ok0 = v0 !== null && v0 !== undefined && Number.isFinite(v0);
    const ok1 = v1 !== null && v1 !== undefined && Number.isFinite(v1);

    if (ok0 && ok1) {
      out[g] = v0 * (1 - frac) + v1 * frac;
    } else if (frac <= 1e-9 && ok0) {
      out[g] = v0;
    }
  }

  return out;
}

export interface ShiftedSolar1Output {
  solar1: Solar1Series;
  meanDelayMin: number;
  minDelayMin: number;
  maxDelayMin: number;
}

export function applySolar1Shift(
  rawSolar1: Solar1Series,
  mode: Solar1ShiftMode,
  options: {
    manualLagMin?: number;
    refDistanceKm?: number;
    smoothWindowMin?: number;
  } = {},
): ShiftedSolar1Output {
  const n = rawSolar1.timestamps_utc.length;
  const refDistanceKm = options.refDistanceKm ?? rawSolar1.ref_distance_km ?? 1_400_000;
  const smoothWindowMin = options.smoothWindowMin ?? 31;
  const manualLagMin = options.manualLagMin ?? 45;

  if (mode === "l1_raw" || n === 0) {
    return {
      solar1: {
        ...rawSolar1,
        shift_mode: "l1_raw",
        time_frame: "L1",
      },
      meanDelayMin: 0,
      minDelayMin: 0,
      maxDelayMin: 0,
    };
  }

  const tGridSec = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tGridSec[i] = Date.parse(rawSolar1.timestamps_utc[i]) / 1000.0;
  }

  let arrivalSec: Float64Array;
  let effDelayMin: Float64Array;

  if (mode === "ballistic_auto") {
    const vx = rawSolar1.series.vx_gse_kms ?? [];
    const res = computeBallisticArrivalSec(tGridSec, vx, {
      refDistanceKm,
      smoothWindowMin,
    });
    arrivalSec = res.arrivalSec;
    effDelayMin = res.effectiveDelayMin;
  } else {
    arrivalSec = new Float64Array(n);
    effDelayMin = new Float64Array(n);
    const lagSec = manualLagMin * 60.0;
    for (let i = 0; i < n; i++) {
      arrivalSec[i] = tGridSec[i] + lagSec;
      effDelayMin[i] = manualLagMin;
    }
  }

  const shiftedSeries: Record<string, (number | null)[]> = {};
  for (const [col, arr] of Object.entries(rawSolar1.series)) {
    shiftedSeries[col] = resampleMonotonicSeries(arrivalSec, arr, tGridSec);
  }

  const delayResampled = resampleMonotonicSeries(arrivalSec, effDelayMin, tGridSec);
  const npArr = shiftedSeries.proton_n_cm3 ?? [];
  const vxArr = shiftedSeries.vx_gse_kms ?? [];
  const validOut = new Array<boolean>(n);
  const flagsOut = new Array<number | null>(n);
  for (let i = 0; i < n; i++) {
    const npVal = npArr[i];
    const vxVal = vxArr[i];
    const ok =
      npVal !== null &&
      npVal !== undefined &&
      Number.isFinite(npVal) &&
      vxVal !== null &&
      vxVal !== undefined &&
      Number.isFinite(vxVal);
    validOut[i] = ok;
    flagsOut[i] = ok ? 0 : null;
  }

  let sumDelay = 0;
  let countDelay = 0;
  let minDelay = Number.POSITIVE_INFINITY;
  let maxDelay = Number.NEGATIVE_INFINITY;
  for (const d of delayResampled) {
    if (d !== null && Number.isFinite(d)) {
      sumDelay += d;
      countDelay++;
      if (d < minDelay) minDelay = d;
      if (d > maxDelay) maxDelay = d;
    }
  }

  return {
    solar1: {
      timestamps_utc: rawSolar1.timestamps_utc,
      time_frame: `L1_shifted_${mode}`,
      shift_mode: mode,
      ref_distance_km: refDistanceKm,
      propagation_delay_min: delayResampled,
      series: shiftedSeries,
      swips_flags: flagsOut,
      is_swips_valid: validOut,
      warning: rawSolar1.warning,
    },
    meanDelayMin: countDelay > 0 ? sumDelay / countDelay : 0,
    minDelayMin: countDelay > 0 ? minDelay : 0,
    maxDelayMin: countDelay > 0 ? maxDelay : 0,
  };
}
