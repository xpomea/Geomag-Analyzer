import type { StationInfo } from "../api/types";

export interface CanonicalStationPair {
  stationA: string;
  stationB: string;
  swapped: boolean;
}

/**
 * Canonicalize station pair ordering (West-first lon_deg asc, tie-break North-first lat_deg desc,
 * then IAGA code asc) so WTC cache keys and cross-phase signs remain consistent across map clicks.
 */
export function canonicalizeStationPair(
  codeA: string,
  codeB: string,
  stationMetaMap?: ReadonlyMap<string, StationInfo>,
): CanonicalStationPair {
  const a = codeA.trim().toUpperCase();
  const b = codeB.trim().toUpperCase();
  if (a === b) {
    return { stationA: a, stationB: b, swapped: false };
  }

  const metaA = stationMetaMap?.get(a);
  const metaB = stationMetaMap?.get(b);
  if (metaA && metaB) {
    const lonDiff = metaA.lon_deg - metaB.lon_deg;
    if (Math.abs(lonDiff) > 1e-9) {
      return lonDiff < 0
        ? { stationA: a, stationB: b, swapped: false }
        : { stationA: b, stationB: a, swapped: true };
    }
    const latDiff = metaA.lat_deg - metaB.lat_deg;
    if (Math.abs(latDiff) > 1e-9) {
      return latDiff > 0
        ? { stationA: a, stationB: b, swapped: false }
        : { stationA: b, stationB: a, swapped: true };
    }
  }

  return a <= b
    ? { stationA: a, stationB: b, swapped: false }
    : { stationA: b, stationB: a, swapped: true };
}
