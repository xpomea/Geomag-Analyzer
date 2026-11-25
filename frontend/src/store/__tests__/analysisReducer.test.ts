import { describe, expect, it } from "vitest";
import type { EpisodeInfo, JobStatusResponse, StationInfo } from "../../api/types";
import {
  INITIAL_ANALYSIS_STATE,
  MIN_ZOOM_SPAN_MS,
  analysisReducer,
  getStationBaselineWarning,
  selectBackendFetchSignature,
} from "../analysisReducer";

const SAMPLE_EPISODE: EpisodeInfo = {
  episode_id: "storm_20261004_05",
  label: "Oct 4-5, 2026",
  start_utc: "2026-10-04T00:00:00Z",
  end_utc: "2026-10-05T23:59:00Z",
  n_minutes: 2880,
  n_days: 2,
  omni_available: true,
  oulu_baseline_cpm: 6000,
};

const BASELESS_GCK: StationInfo = {
  iaga_code: "GCK",
  station_name: "Grocka",
  station_type: "magnetometer",
  region: "Europe",
  lat_deg: 44.63,
  lon_deg: 20.77,
  elev_m: 231,
  orig_format: "XYZF",
  has_baseline: false,
  mean_vec_norm_nT: 469.3,
  days_present: 10,
  source_of_data: "INTERMAGNET",
  cutoff_rigidity_gv: null,
  barometric_coeff: null,
};

const FULLFIELD_RES: StationInfo = {
  ...BASELESS_GCK,
  iaga_code: "RES",
  station_name: "Resolute Bay",
  region: "Canada",
  lat_deg: 74.69,
  lon_deg: -94.89,
  has_baseline: true,
  mean_vec_norm_nT: 57372.7,
};

describe("analysisReducer & state guards", () => {
  it("resets view range and cursor UTC on episode switch and clears active WTC job", () => {
    const dirty = {
      ...INITIAL_ANALYSIS_STATE,
      activeWtcJobId: "job-123",
    };
    const next = analysisReducer(dirty, {
      type: "SET_EPISODE",
      episodeId: SAMPLE_EPISODE.episode_id,
      episode: SAMPLE_EPISODE,
    });
    expect(next.episodeId).toBe("storm_20261004_05");
    expect(next.activeWtcJobId).toBeNull();
    expect(next.viewRangeMs).toEqual([
      Date.parse(SAMPLE_EPISODE.start_utc),
      Date.parse(SAMPLE_EPISODE.end_utc),
    ]);
    expect(next.cursorUtcMs).not.toBeNull();
  });

  it("keeps backend fetch signature invariant under client-only alpha and solar1 shift actions", () => {
    const sig0 = selectBackendFetchSignature(INITIAL_ANALYSIS_STATE);
    const s1 = analysisReducer(INITIAL_ANALYSIS_STATE, { type: "SET_ALPHA", alpha: 0.01 });
    const s2 = analysisReducer(s1, { type: "SET_SOLAR1_SHIFT", mode: "manual" });
    const s3 = analysisReducer(s2, { type: "SET_MANUAL_LAG", lagMin: 55 });
    const s4 = analysisReducer(s3, {
      type: "SET_VIEW_RANGE",
      rangeMs: [Date.parse("2026-10-10T00:00:00Z"), Date.parse("2026-10-10T12:00:00Z")],
    });
    const s5 = analysisReducer(s4, { type: "SET_EXPORT_THEME", theme: "dark" });

    expect(s5.exportTheme).toBe("dark");
    expect(selectBackendFetchSignature(s5)).toBe(sig0);

    const sMethod = analysisReducer(s4, { type: "SET_METHOD", method: "variant_b_sos" });
    expect(selectBackendFetchSignature(sMethod)).not.toBe(sig0);
  });

  it("harmonizes crossComponent when toggling XYZ <-> XmYm and clamps zoom bounds", () => {
    const sXm = analysisReducer(INITIAL_ANALYSIS_STATE, {
      type: "SET_COORD_FRAME",
      coordFrame: "XmYm",
    });
    expect(sXm.crossComponent).toBe("Xm");

    const sZ = analysisReducer(sXm, { type: "SET_CROSS_COMPONENT", component: "Z" });
    const sXyz = analysisReducer(sZ, { type: "SET_COORD_FRAME", coordFrame: "XYZ" });
    expect(sXyz.crossComponent).toBe("Z");

    const b0 = Date.parse(SAMPLE_EPISODE.start_utc);
    const b1 = Date.parse(SAMPLE_EPISODE.end_utc);
    const zoomed = analysisReducer(sXyz, {
      type: "SET_VIEW_RANGE",
      rangeMs: [b0 - 10_000, b0 + 5_000],
      boundsMs: [b0, b1],
    });
    expect(zoomed.viewRangeMs?.[0]).toBe(b0);
    expect((zoomed.viewRangeMs?.[1] ?? 0) - (zoomed.viewRangeMs?.[0] ?? 0)).toBeGreaterThanOrEqual(
      MIN_ZOOM_SPAN_MS,
    );
  });

  it("guards baseless GCK/NCK against automatic XmYm rotation without external D0", () => {
    expect(getStationBaselineWarning(BASELESS_GCK, "XYZ", null)).toBeNull();
    expect(getStationBaselineWarning(BASELESS_GCK, "XmYm", null)).toContain("GCK");
    expect(getStationBaselineWarning(BASELESS_GCK, "XmYm", 5.2)).toBeNull();
    expect(getStationBaselineWarning(FULLFIELD_RES, "XmYm", null)).toBeNull();
  });

  it("upserts and updates job statuses cleanly", () => {
    const job1: JobStatusResponse = {
      job_id: "j-1",
      job_type: "wtc",
      cache_key: "abc",
      status: "running",
      stage: "Monte Carlo",
      progress_pct: 40,
      completed_steps: 4,
      total_steps: 10,
      eta_sec: 12,
      cache_hit: false,
      result_path: null,
      error_message: null,
      params: { station_i: "LVV", station_j: "BEL" },
      created_at: "2026-11-24T12:00:00Z",
      updated_at: "2026-11-24T12:00:10Z",
    };

    const s1 = analysisReducer(INITIAL_ANALYSIS_STATE, { type: "UPSERT_JOB", job: job1 });
    expect(s1.jobs).toHaveLength(1);
    const s2 = analysisReducer(s1, {
      type: "UPSERT_JOB",
      job: { ...job1, status: "completed", progress_pct: 100 },
    });
    expect(s2.jobs).toHaveLength(1);
    expect(s2.jobs[0].status).toBe("completed");
  });

  it("handles OPEN_WTC_JOB cross-episode switch, SET_WTC_STRIDE, SET_EXTERNAL_D0_B, and windowSamples >= L guard", () => {
    const b0 = Date.parse(SAMPLE_EPISODE.start_utc);
    const b1 = Date.parse(SAMPLE_EPISODE.end_utc);
    const opened = analysisReducer(INITIAL_ANALYSIS_STATE, {
      type: "OPEN_WTC_JOB",
      jobId: "wtc-old",
      episodeId: SAMPLE_EPISODE.episode_id,
      stationA: "bel",
      stationB: "wng",
      component: "Ym",
      coordFrame: "XmYm",
      boundsMs: [b0, b1],
    });
    expect(opened.episodeId).toBe(SAMPLE_EPISODE.episode_id);
    expect(opened.stationA).toBe("BEL");
    expect(opened.stationB).toBe("WNG");
    expect(opened.crossComponent).toBe("Ym");
    expect(opened.activeCrossTab).toBe("wtc");
    expect(opened.activeWtcJobId).toBe("wtc-old");
    expect(opened.viewRangeMs).toEqual([b0, b1]);

    const strided = analysisReducer(opened, { type: "SET_WTC_STRIDE", stride: 4 });
    expect(strided.wtcStride).toBe(4);

    const d0b = analysisReducer(strided, { type: "SET_EXTERNAL_D0_B", d0Deg: 4.5 });
    expect(d0b.externalD0DegB).toBe(4.5);

    const smallWin = analysisReducer(d0b, { type: "SET_WELCH_PARAMS", L: 256, windowSamples: 128 });
    expect(smallWin.L).toBe(256);
    expect(smallWin.windowSamples).toBe(256);

    // Switching L: 256 -> 128 with M = 128 must clamp M to floor(128 / 2) = 64 (not 127!)
    const withM128 = analysisReducer(smallWin, { type: "SET_WELCH_PARAMS", M: 128 });
    expect(withM128.M).toBe(128);
    const backTo128 = analysisReducer(withM128, { type: "SET_WELCH_PARAMS", L: 128 });
    expect(backTo128.L).toBe(128);
    expect(backTo128.M).toBe(64);

    // OPEN_WTC_JOB restores external D0 values when provided
    const openedWithD0 = analysisReducer(backTo128, {
      type: "OPEN_WTC_JOB",
      jobId: "wtc-gck",
      episodeId: SAMPLE_EPISODE.episode_id,
      stationA: "GCK",
      stationB: "BEL",
      component: "Xm",
      coordFrame: "XmYm",
      externalD0Deg: 5.5,
      externalD0DegB: null,
    });
    expect(openedWithD0.externalD0Deg).toBe(5.5);
    expect(openedWithD0.externalD0DegB).toBeNull();
  });

  it("syncs activeDeckTab and activeCrossTab cleanly without changing backend fetch signature", () => {
    const sig0 = selectBackendFetchSignature(INITIAL_ANALYSIS_STATE);
    const sWelch = analysisReducer(INITIAL_ANALYSIS_STATE, {
      type: "SET_ACTIVE_DECK_TAB",
      tab: "welch",
    });
    expect(sWelch.activeDeckTab).toBe("welch");
    expect(sWelch.activeCrossTab).toBe("welch");
    expect(selectBackendFetchSignature(sWelch)).toBe(sig0);

    const sStokes = analysisReducer(sWelch, {
      type: "SET_ACTIVE_DECK_TAB",
      tab: "stokes",
    });
    expect(sStokes.activeDeckTab).toBe("stokes");
    expect(sStokes.activeCrossTab).toBe("spectrogram");
    expect(selectBackendFetchSignature(sStokes)).toBe(sig0);

    const sWtc = analysisReducer(sStokes, {
      type: "OPEN_WTC_JOB",
      jobId: "wtc-sync",
      episodeId: INITIAL_ANALYSIS_STATE.episodeId,
      stationA: "LVV",
      stationB: "BEL",
      component: "X",
      coordFrame: "XYZ",
    });
    expect(sWtc.activeDeckTab).toBe("wtc");
    expect(sWtc.activeCrossTab).toBe("wtc");
  });
});
