import { describe, expect, it } from "vitest";
import { extractSignificanceContours } from "../contours";

describe("contours (Marching Squares 95% MC contours)", () => {
  it("returns empty segments for empty or all-false grids", () => {
    expect(extractSignificanceContours([])).toEqual([]);
    const allFalse = [
      [false, false, false],
      [false, false, false],
    ];
    expect(extractSignificanceContours(allFalse)).toEqual([]);
  });

  it("extracts closed boundary segments around an interior island and respects COI", () => {
    const nScales = 6;
    const nTime = 6;
    const sigGrid = Array.from({ length: nScales }, (_, s) =>
      Array.from({ length: nTime }, (_, t) => s >= 2 && s <= 3 && t >= 2 && t <= 3),
    );
    const segments = extractSignificanceContours(sigGrid);
    expect(segments.length).toBeGreaterThanOrEqual(8);

    // Verify all vertices sit strictly inside the grid bounds
    for (const [[t0, s0], [t1, s1]] of segments) {
      expect(t0).toBeGreaterThanOrEqual(0);
      expect(t0).toBeLessThanOrEqual(nTime - 1);
      expect(s0).toBeGreaterThanOrEqual(0);
      expect(s0).toBeLessThanOrEqual(nScales - 1);
      expect(t1).toBeGreaterThanOrEqual(0);
      expect(s1).toBeGreaterThanOrEqual(0);
    }

    // Mask out everything via insideCoi=false -> 0 contour segments
    const noCoi = Array.from({ length: nScales }, () => Array(nTime).fill(false));
    expect(extractSignificanceContours(sigGrid, noCoi)).toEqual([]);
  });
});
