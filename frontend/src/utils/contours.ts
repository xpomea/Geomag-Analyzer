/**
 * Marching Squares contour extraction over 2D boolean significance grids
 * and 256-entry RGB colormaps for Canvas 2D spectrogram / XWT / WTC heatmaps.
 */

export type LineSegment2D = [[number, number], [number, number]];

const EDGE_TABLE: Record<number, number[][]> = {
  0: [],
  1: [[3, 2]],
  2: [[2, 1]],
  3: [[3, 1]],
  4: [[1, 0]],
  5: [
    [3, 0],
    [1, 2],
  ],
  6: [[2, 0]],
  7: [[3, 0]],
  8: [[0, 3]],
  9: [[0, 2]],
  10: [
    [0, 1],
    [2, 3],
  ],
  11: [[0, 1]],
  12: [[1, 3]],
  13: [[1, 2]],
  14: [[2, 3]],
  15: [],
};

function edgePoint(edge: number, t: number, s: number): [number, number] {
  switch (edge) {
    case 0:
      return [t + 0.5, s]; // top
    case 1:
      return [t + 1.0, s + 0.5]; // right
    case 2:
      return [t + 0.5, s + 1.0]; // bottom
    case 3:
    default:
      return [t, s + 0.5]; // left
  }
}

/**
 * Extract 2D boundary contour segments around regions where
 * `sigGrid[s][t] && (!insideCoi || insideCoi[s][t])` is true.
 * Returned coordinates are [timeIdx, scaleIdx] clamped to grid bounds.
 */
export function extractSignificanceContours(
  sigGrid: boolean[][],
  insideCoi?: boolean[][],
): LineSegment2D[] {
  const nScales = sigGrid.length;
  if (nScales === 0) return [];
  const nTime = sigGrid[0]?.length ?? 0;
  if (nTime === 0) return [];

  const at = (s: number, t: number): boolean => {
    if (s < 0 || s >= nScales || t < 0 || t >= nTime) return false;
    if (!sigGrid[s]?.[t]) return false;
    if (insideCoi && !insideCoi[s]?.[t]) return false;
    return true;
  };

  const clampPt = (pt: [number, number]): [number, number] => [
    Math.max(0, Math.min(nTime - 1, pt[0])),
    Math.max(0, Math.min(nScales - 1, pt[1])),
  ];

  const segments: LineSegment2D[] = [];
  for (let s = -1; s < nScales; s++) {
    for (let t = -1; t < nTime; t++) {
      const tl = at(s, t) ? 8 : 0;
      const tr = at(s, t + 1) ? 4 : 0;
      const br = at(s + 1, t + 1) ? 2 : 0;
      const bl = at(s + 1, t) ? 1 : 0;
      const code = tl | tr | br | bl;
      const pairs = EDGE_TABLE[code];
      if (!pairs || pairs.length === 0) continue;

      for (const [e0, e1] of pairs) {
        segments.push([clampPt(edgePoint(e0, t, s)), clampPt(edgePoint(e1, t, s))]);
      }
    }
  }
  return segments;
}

export type ColormapName = "viridis" | "plasma" | "turbo" | "phase";

const COLOR_STOPS: Record<ColormapName, [number, number, number][]> = {
  viridis: [
    [68, 1, 84],
    [71, 44, 122],
    [59, 81, 139],
    [44, 113, 142],
    [33, 144, 141],
    [39, 173, 129],
    [92, 200, 99],
    [170, 220, 50],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [75, 3, 161],
    [125, 3, 168],
    [168, 34, 150],
    [203, 70, 121],
    [229, 107, 93],
    [248, 148, 65],
    [253, 195, 40],
    [240, 249, 33],
  ],
  turbo: [
    [48, 18, 59],
    [65, 69, 171],
    [42, 130, 235],
    [24, 192, 195],
    [74, 234, 115],
    [171, 241, 56],
    [242, 200, 44],
    [248, 117, 29],
    [194, 38, 13],
    [122, 4, 3],
  ],
  phase: [
    [28, 36, 76], // -180
    [49, 102, 186], // -120
    [94, 180, 212], // -60
    [232, 237, 242], // 0 deg
    [214, 136, 86], // +60
    [164, 54, 72], // +120
    [28, 36, 76], // +180 wraps seamlessly
  ],
};

export function buildColorLut(name: ColormapName): Uint8Array {
  const stops = COLOR_STOPS[name];
  const lut = new Uint8Array(256 * 3);
  const segCount = stops.length - 1;

  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * segCount;
    const idx = Math.min(segCount - 1, Math.floor(pos));
    const frac = pos - idx;
    const c0 = stops[idx];
    const c1 = stops[idx + 1];
    lut[i * 3] = Math.round(c0[0] + (c1[0] - c0[0]) * frac);
    lut[i * 3 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * frac);
    lut[i * 3 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * frac);
  }
  return lut;
}
