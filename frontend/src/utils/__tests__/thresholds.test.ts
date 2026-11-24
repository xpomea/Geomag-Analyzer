import { describe, expect, it } from "vitest";
import {
  betaInv,
  gamma2Crit,
  pCrit,
  regularizedIncompleteBeta,
  scaleGamma2CritForAlpha,
} from "../thresholds";

// Exact rows from GeoMag Storms/Pc5_Pc6_Polarization_Formulas_INTERMAGNET_XYZ_RU.docx Table 5
const DOCX_TABLE_5 = [
  { n: 8, p95: 0.646, p99: 0.737, g95: 0.348, g99: 0.482 },
  { n: 16, p95: 0.476, p99: 0.558, g95: 0.181, g99: 0.264 },
  { n: 32, p95: 0.343, p99: 0.407, g95: 0.092, g99: 0.138 },
  { n: 64, p95: 0.245, p99: 0.293, g95: 0.046, g99: 0.07 },
  { n: 128, p95: 0.174, p99: 0.209, g95: 0.023, g99: 0.036 },
  { n: 256, p95: 0.123, p99: 0.148, g95: 0.012, g99: 0.018 },
  { n: 512, p95: 0.087, p99: 0.105, g95: 0.006, g99: 0.009 },
];

describe("thresholds (pCrit, gamma2Crit, betaInv)", () => {
  it("matches all 7 rows of docx Table 5 for alpha=0.05 and alpha=0.01", () => {
    for (const row of DOCX_TABLE_5) {
      expect(Math.abs(pCrit(row.n, 0.05) - row.p95)).toBeLessThanOrEqual(0.0006);
      expect(Math.abs(pCrit(row.n, 0.01) - row.p99)).toBeLessThanOrEqual(0.0006);
      expect(Math.abs(gamma2Crit(row.n, 0.05) - row.g95)).toBeLessThanOrEqual(0.0006);
      expect(Math.abs(gamma2Crit(row.n, 0.01) - row.g99)).toBeLessThanOrEqual(0.0006);
    }
  });

  it("inverts regularizedIncompleteBeta to machine precision", () => {
    for (const p of [0.01, 0.05, 0.1, 0.5, 0.95]) {
      for (const a of [7, 74, 361]) {
        const u = betaInv(p, a, 1.5);
        expect(Math.abs(regularizedIncompleteBeta(u, a, 1.5) - p)).toBeLessThan(1e-12);
      }
    }
  });

  it("is strictly monotonic in n_eff and alpha and scales gamma2_crit exactly", () => {
    expect(pCrit(75, 0.01)).toBeGreaterThan(pCrit(75, 0.05));
    expect(pCrit(75, 0.05)).toBeGreaterThan(pCrit(75, 0.1));
    expect(pCrit(200, 0.05)).toBeLessThan(pCrit(75, 0.05));

    const gc05 = gamma2Crit(48, 0.05);
    const gc01 = gamma2Crit(48, 0.01);
    expect(Math.abs(scaleGamma2CritForAlpha(gc05, 0.05, 0.01) - gc01)).toBeLessThan(1e-13);
    expect(Math.abs(scaleGamma2CritForAlpha(gc05, 0.05, 0.05) - gc05)).toBeLessThan(1e-15);
  });

  it("rejects invalid degrees of freedom or alpha outside (0, 1)", () => {
    expect(() => pCrit(1, 0.05)).toThrow();
    expect(() => pCrit(10, 0)).toThrow();
    expect(() => gamma2Crit(10, 1)).toThrow();
  });
});
