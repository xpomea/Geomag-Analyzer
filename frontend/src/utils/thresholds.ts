/**
 * Client-side critical thresholds for polarization p_crit and coherence gamma^2_crit
 * matching Formulas (24)-(25) and Table 5 of Pc5_Pc6_Polarization_Formulas_INTERMAGNET_XYZ_RU.docx.
 */

const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function logGamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  const zm1 = z - 1;
  let x = LANCZOS_COEFFS[0];
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
    x += LANCZOS_COEFFS[i] / (zm1 + i);
  }
  const t = zm1 + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (zm1 + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const maxIter = 300;
  const eps = 3e-15;
  const fpMin = 1e-300;

  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < fpMin) d = fpMin;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIter; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) <= eps) {
      break;
    }
  }
  return h;
}

export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (!(a > 0 && b > 0)) {
    throw new Error(`Beta parameters must be positive, got a=${a}, b=${b}`);
  }
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const bt = Math.exp(
    logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log(1 - x),
  );

  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betaContinuedFraction(a, b, x)) / a;
  }
  return 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
}

export function betaInv(p: number, a: number, b: number): number {
  if (!(a > 0 && b > 0)) {
    throw new Error(`Beta parameters must be positive, got a=${a}, b=${b}`);
  }
  if (!Number.isFinite(p) || p < 0 || p > 1) {
    throw new Error(`Probability p must be in [0, 1], got ${p}`);
  }
  if (p === 0) return 0;
  if (p === 1) return 1;

  const logNorm = logGamma(a) + logGamma(b) - logGamma(a + b);
  let lo = 0;
  let hi = 1;
  let x = Math.min(1 - 1e-12, Math.max(1e-12, a / (a + b)));

  for (let iter = 0; iter < 120; iter++) {
    const cdf = regularizedIncompleteBeta(x, a, b);
    const err = cdf - p;
    if (err < 0) {
      lo = Math.max(lo, x);
    } else {
      hi = Math.min(hi, x);
    }
    if (Math.abs(err) <= 1e-14 || hi - lo <= 1e-14) {
      return x;
    }

    const logPdf = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logNorm;
    const pdf = Math.exp(logPdf);
    let next = pdf > 1e-300 ? x - err / pdf : 0.5 * (lo + hi);
    if (!Number.isFinite(next) || next <= lo || next >= hi) {
      next = 0.5 * (lo + hi);
    }
    const step = Math.abs(next - x);
    x = next;
    if (step <= 1e-14) {
      return x;
    }
  }
  return x;
}

export function pCrit(nEff: number, alpha: number): number {
  if (!Number.isFinite(nEff) || nEff <= 1) {
    throw new Error(`Effective degrees of freedom n_eff must be > 1, got ${nEff}`);
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new Error(`Significance level alpha must be in (0, 1), got ${alpha}`);
  }
  const uAlpha = betaInv(alpha, nEff - 1, 1.5);
  return Math.sqrt(Math.max(0, 1 - uAlpha));
}

export function gamma2Crit(nEff: number, alpha: number): number {
  if (!Number.isFinite(nEff) || nEff <= 1) {
    throw new Error(`Effective degrees of freedom n_eff must be > 1, got ${nEff}`);
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new Error(`Significance level alpha must be in (0, 1), got ${alpha}`);
  }
  return 1 - Math.pow(alpha, 1 / (nEff - 1));
}

export function scaleGamma2CritForAlpha(
  gamma2Crit0: number,
  alpha0: number,
  alphaNew: number,
): number {
  if (!Number.isFinite(alpha0) || alpha0 <= 0 || alpha0 >= 1) {
    throw new Error(`alpha0 must be in (0, 1), got ${alpha0}`);
  }
  if (!Number.isFinite(alphaNew) || alphaNew <= 0 || alphaNew >= 1) {
    throw new Error(`alphaNew must be in (0, 1), got ${alphaNew}`);
  }
  if (Math.abs(alphaNew - alpha0) < 1e-15) {
    return gamma2Crit0;
  }
  if (gamma2Crit0 <= 0) return 0;
  if (gamma2Crit0 >= 1) return 1;
  const ratio = Math.log(alphaNew) / Math.log(alpha0);
  return 1 - Math.pow(1 - gamma2Crit0, ratio);
}
