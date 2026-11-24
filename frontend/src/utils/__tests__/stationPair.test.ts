import { describe, expect, it } from "vitest";
import type { StationInfo } from "../../api/types";
import {
  computeVisibleArrowStepX,
  findNearestYIndex,
} from "../../components/SyncedCharts/ScientificHeatmap2D";
import { canonicalizeStationPair } from "../stationPair";

function makeStation(code: string, lon: number, lat: number): StationInfo {
  return {
    iaga_code: code,
    station_name: code,
    station_type: "magnetometer",
    region: lon < -20 ? "Canada" : "Europe",
    lat_deg: lat,
    lon_deg: lon,
    elev_m: 100,
    orig_format: "XYZF",
    has_baseline: true,
    mean_vec_norm_nT: 48000,
    days_present: 10,
    source_of_data: "INTERMAGNET",
    cutoff_rigidity_gv: null,
    barometric_coeff: null,
  };
}

describe("stationPair canonicalization and heatmap scale helpers", () => {
  it("orders pairs West-first, then North-first, then IAGA code", () => {
    const map = new Map<string, StationInfo>([
      ["OTT", makeStation("OTT", -75.55, 45.4)],
      ["LVV", makeStation("LVV", 23.75, 49.9)],
      ["NTH", makeStation("NTH", 20.0, 65.0)],
      ["STH", makeStation("STH", 20.0, 45.0)],
    ]);

    expect(canonicalizeStationPair("LVV", "OTT", map)).toEqual({
      stationA: "OTT",
      stationB: "LVV",
      swapped: true,
    });
    expect(canonicalizeStationPair("OTT", "LVV", map)).toEqual({
      stationA: "OTT",
      stationB: "LVV",
      swapped: false,
    });
    expect(canonicalizeStationPair("STH", "NTH", map)).toEqual({
      stationA: "NTH",
      stationB: "STH",
      swapped: true,
    });
  });

  it("computes dense arrow stepX under deep zoom and matches log-axis nearest Y", () => {
    expect(computeVisibleArrowStepX(360)).toBe(11);
    expect(computeVisibleArrowStepX(64)).toBe(2);
    expect(computeVisibleArrowStepX(20)).toBe(1);
    expect(computeVisibleArrowStepX(360, 15)).toBe(15);

    // Log-spaced periods [120, 240, 480, 960]: geometric midpoint of 240 and 480 is sqrt(240*480) ~= 339.4.
    // Target 350 > 339.4 is closer to 480 in log space, whereas |350-240|=110 < |350-480|=130 linearly!
    const periods = [120, 240, 480, 960];
    expect(findNearestYIndex(periods, 350, "log")).toBe(2);
    expect(findNearestYIndex(periods, 350, "linear")).toBe(1);
  });
});
