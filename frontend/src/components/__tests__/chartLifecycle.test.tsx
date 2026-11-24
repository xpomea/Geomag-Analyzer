import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  CatalogResponse,
  GeomagTimeseriesResponse,
  HealthResponse,
  PolarizationResponse,
  SpaceWeatherResponse,
  WelchPairResponse,
} from "../../api/types";
import {
  AnalysisContext,
  AnalysisProvider,
  useAnalysis,
  type AnalysisContextValue,
} from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import { INITIAL_ANALYSIS_STATE } from "../../store/analysisReducer";
import { applySolar1Shift } from "../../utils/ballisticShift";
import { applyPolarizationAlphaMask, applyWelchAlphaMask } from "../../utils/masking";
import { gamma2Crit, pCrit } from "../../utils/thresholds";
import { WelchCrossPanel } from "../CrossSpectralPanel/WelchCrossPanel";
import { DraftNullableNumberInput } from "../Layout/DraftNumberInput";
import { MagnetogramPanel } from "../SyncedCharts/MagnetogramPanel";
import { SpaceWeatherPanel } from "../SyncedCharts/SpaceWeatherPanel";
import { StokesPanel } from "../SyncedCharts/StokesPanel";

interface MockChartRecord {
  el: HTMLElement;
  disposed: boolean;
  setOptionCalls: number;
  dispatchActionCalls: number;
  lastOption: unknown;
}

const mockCharts: MockChartRecord[] = [];

vi.mock("echarts", () => {
  return {
    init: vi.fn((el: HTMLElement) => {
      const rec: MockChartRecord = {
        el,
        disposed: false,
        setOptionCalls: 0,
        dispatchActionCalls: 0,
        lastOption: null,
      };
      mockCharts.push(rec);
      return {
        group: "",
        setOption: vi.fn((opt: unknown) => {
          rec.setOptionCalls += 1;
          rec.lastOption = opt;
        }),
        dispatchAction: vi.fn(() => {
          rec.dispatchActionCalls += 1;
        }),
        getOption: vi.fn(() => ({ dataZoom: [{ startValue: 0, endValue: 1 }] })),
        resize: vi.fn(),
        on: vi.fn(),
        off: vi.fn(),
        dispose: vi.fn(() => {
          rec.disposed = true;
        }),
      };
    }),
    connect: vi.fn(),
  };
});

function makeBaseContext(overrides: Partial<AnalysisContextValue> = {}): AnalysisContextValue {
  return {
    state: {
      ...INITIAL_ANALYSIS_STATE,
      viewRangeMs: [Date.UTC(2026, 9, 10, 0, 0), Date.UTC(2026, 9, 10, 12, 0)],
      cursorUtcMs: Date.UTC(2026, 9, 10, 6, 0),
    },
    dispatch: vi.fn(),
    auth: { checked: true, authenticated: true, requireAuth: true, error: null },
    health: null,
    catalog: null,
    activeEpisode: null,
    episodeBoundsMs: [Date.UTC(2026, 9, 10, 0, 0), Date.UTC(2026, 9, 10, 12, 0)],
    stationMetaMap: new Map(),
    stationAvailabilityMap: new Map(),
    geomag: null,
    rawSpaceWeather: null,
    shiftedSolar1: null,
    polarizationRaw: null,
    maskedPolarization: null,
    welchPairRaw: null,
    maskedWelch: null,
    flrProfile: null,
    wtcResult: null,
    loadingCatalog: false,
    loadingTimeseries: false,
    loadingPolarization: false,
    loadingWelch: false,
    loadingFlr: false,
    loadingWtcResult: false,
    globalError: null,
    polarizationError: null,
    welchError: null,
    flrError: null,
    wtcError: null,
    baselineWarning: null,
    authenticateWithToken: vi.fn(),
    refreshJobsAndCache: vi.fn(),
    launchWtcJob: vi.fn(),
    cancelWtcJob: vi.fn(),
    openWtcJobResult: vi.fn(),
    loadFlrAtCursor: vi.fn(),
    triggerShutdown: vi.fn(),
    ...overrides,
  };
}

function samplePolarization(): PolarizationResponse {
  const ts = [
    "2026-10-10T01:00:00Z",
    "2026-10-10T02:00:00Z",
    "2026-10-10T03:00:00Z",
  ];
  const nEff = 75;
  return {
    episode_id: "storm_20261009_11",
    station_iaga: "LVV",
    method: "welch",
    band: "Pc5",
    coord_frame: "XYZ",
    d0_deg: 0,
    is_valid: true,
    n_eff: nEff,
    trim_hours: null,
    stokes_2d: {
      I: 100,
      Q: 30,
      U: 20,
      V: 40,
      A: 36,
      R: 53.8,
      p: 0.54,
      p_L: 0.36,
      p_C: 0.4,
      psi_deg: 16.8,
      chi_deg: 24,
      epsilon: 0.44,
      gamma2: 0.3,
      arg_sxy_deg: 60,
      n_eff: nEff,
      p_crit: pCrit(nEff, 0.05),
      gamma2_crit: gamma2Crit(nEff, 0.05),
      is_significant_p: true,
      is_significant_gamma2: true,
    },
    pol_3d: null,
    dynamic: {
      timestamps_utc: ts,
      I: [80, 100, 90],
      p: [0.4, 0.6, 0.5],
      p_crit: [0.23, 0.23, 0.23],
      psi_deg: [15, 20, 18],
      chi_deg: [20, 25, 22],
      epsilon: [0.3, 0.4, 0.35],
      gamma2: [0.2, 0.4, 0.3],
      gamma2_crit: [0.04, 0.04, 0.04],
      n_eff: [nEff, nEff, nEff],
      is_significant_p: [true, true, true],
      is_valid: [true, true, true],
    },
    filtered: null,
    spectrogram: null,
    warnings: [],
  };
}

function sampleWelchPair(): WelchPairResponse {
  const ts = ["2026-10-10T01:00:00Z", "2026-10-10T02:00:00Z"];
  return {
    episode_id: "storm_20261009_11",
    station_i: "LVV",
    station_j: "BEL",
    component: "X",
    coord_frame: "XYZ",
    band: "Pc5",
    available: true,
    cache_hits: { LVV: true, BEL: true },
    window_timestamps_utc: ts,
    freqs_hz: [],
    K_ij: [4, 4],
    n_eff: [45, 45],
    gamma2: [0.3, 0.4],
    gamma2_crit: [0.066, 0.066],
    phase_deg: [10, 20],
    phase_deg_masked: [10, 20],
    is_significant: [true, true],
    is_valid: [true, true],
    gamma2_tf: [],
    gamma2_crit_tf: [],
    phase_deg_tf: [],
    phase_deg_tf_masked: [],
    azimuthal: {
      m: [1.5, 3.0],
      m_alias: 18,
      delta_lon_deg: 6.67,
      is_meridional_guarded: false,
    },
    warnings: [],
  };
}

function sampleGeomag(available: boolean): GeomagTimeseriesResponse {
  const ts = ["2026-10-10T01:00:00Z", "2026-10-10T01:01:00Z", "2026-10-10T01:02:00Z"];
  return {
    episode_id: "storm_20261009_11",
    station_iaga: available ? "LVV" : "SUA",
    station_meta: {
      iaga_code: available ? "LVV" : "SUA",
      station_name: available ? "Lviv" : "Surlari",
      station_type: "magnetometer",
      region: "Europe",
      lat_deg: 49.9,
      lon_deg: 23.75,
      elev_m: 400,
      orig_format: "XYZF",
      has_baseline: true,
      mean_vec_norm_nT: 49000,
      days_present: available ? 10 : 2,
      source_of_data: "INTERMAGNET",
      cutoff_rigidity_gv: null,
      barometric_coeff: null,
    },
    available,
    coord_frame: "XYZ",
    d0_deg: 0,
    valid_pct: available ? 100 : 0,
    timestamps_utc: ts,
    x_nT: available ? [20000, 20005, 20002] : [null, null, null],
    y_nT: available ? [1500, 1503, 1501] : [null, null, null],
    z_nT: available ? [45000, 45001, 45000] : [null, null, null],
    f_scalar_nT: [null, null, null],
    is_valid_xyz: [available, available, available],
    is_interpolated: [false, false, false],
    was_despiked: [false, false, false],
    warnings: available ? [] : ["SUA unavailable"],
  };
}

function sampleSpaceWeather(): SpaceWeatherResponse {
  const ts = ["2026-10-10T01:00:00Z", "2026-10-10T01:01:00Z"];
  return {
    episode_id: "storm_20261009_11",
    solar1: {
      timestamps_utc: ts,
      time_frame: "L1",
      shift_mode: "l1_raw",
      ref_distance_km: 1_400_000,
      propagation_delay_min: [0, 0],
      series: {
        bz_gsm_nT: [-5, -6],
        b_mag_gsm_nT: [10, 11],
        vx_gse_kms: [-500, -510],
        pdyn_nPa: [3.2, 3.4],
        proton_n_cm3: [7, 7.2],
      },
      swips_flags: [0, 0],
      is_swips_valid: [true, true],
      warning: "",
    },
    omni: {
      available: true,
      time_frame: "BSN",
      timestamps_utc: ts,
      series: { ae_nT: [300, 320], au_nT: [100, 110], al_nT: [-200, -210] },
      warning: null,
    },
    oulu: {
      available: true,
      timestamps_utc: ts,
      baseline_cpm: 6000,
      corrected_cpm: [5900, 5880],
      uncorrected_cpm: [5900, 5880],
      pressure_mbar: [1010, 1010],
      delta_n_pct: [-1.67, -2.0],
    },
  };
}

describe("Synced panel ECharts lifecycle (callback-ref regression C1)", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    class MockRO {
      observe() {}
      disconnect() {}
      unobserve() {}
    }
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = MockRO;
    mockCharts.length = 0;
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    host.remove();
  });

  it("initializes StokesPanel ECharts cleanly when option arrives asynchronously after initial mount", () => {
    const emptyCtx = makeBaseContext();
    act(() => {
      root.render(
        <AnalysisContext.Provider value={emptyCtx}>
          <StokesPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts).toHaveLength(0);

    const pol = samplePolarization();
    const readyCtx = makeBaseContext({
      polarizationRaw: pol,
      maskedPolarization: applyPolarizationAlphaMask(pol, 0.05),
    });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={readyCtx}>
          <StokesPanel />
        </AnalysisContext.Provider>,
      );
    });

    expect(mockCharts).toHaveLength(1);
    expect(mockCharts[0].disposed).toBe(false);
    expect(mockCharts[0].setOptionCalls).toBe(1);
  });

  it("initializes WelchCrossPanel 1D ECharts cleanly when welchPairRaw finishes loading", () => {
    const emptyCtx = makeBaseContext();
    act(() => {
      root.render(
        <AnalysisContext.Provider value={emptyCtx}>
          <WelchCrossPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts).toHaveLength(0);

    const welch = sampleWelchPair();
    const readyCtx = makeBaseContext({
      welchPairRaw: welch,
      maskedWelch: applyWelchAlphaMask(welch, 0.05, 0.05),
    });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={readyCtx}>
          <WelchCrossPanel />
        </AnalysisContext.Provider>,
      );
    });

    expect(mockCharts).toHaveLength(1);
    expect(mockCharts[0].disposed).toBe(false);
    expect(mockCharts[0].setOptionCalls).toBe(1);
  });

  it("disposes and re-initializes MagnetogramPanel ECharts across LVV -> SUA (absent) -> LVV", () => {
    const ctxLvv = makeBaseContext({ geomag: sampleGeomag(true) });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctxLvv}>
          <MagnetogramPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts).toHaveLength(1);
    const firstDiv = mockCharts[0].el;

    const ctxSua = makeBaseContext({ geomag: sampleGeomag(false) });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctxSua}>
          <MagnetogramPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts[0].disposed).toBe(true);

    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctxLvv}>
          <MagnetogramPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts).toHaveLength(2);
    expect(mockCharts[1].disposed).toBe(false);
    expect(mockCharts[1].el).not.toBe(firstDiv);
    expect(mockCharts[1].setOptionCalls).toBe(1);
  });

  it("initializes SpaceWeatherPanel and disposes cleanly on unmount", () => {
    const sw = sampleSpaceWeather();
    const ctx = makeBaseContext({
      rawSpaceWeather: sw,
      shiftedSolar1: applySolar1Shift(sw.solar1, "l1_raw"),
    });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctx}>
          <SpaceWeatherPanel />
        </AnalysisContext.Provider>,
      );
    });
    expect(mockCharts).toHaveLength(1);
    expect(mockCharts[0].setOptionCalls).toBe(1);

    act(() => {
      root.unmount();
    });
    expect(mockCharts[0].disposed).toBe(true);
  });

  it("renders T_cut markAreas in Variant B by default and plots both x_amp and y_amp envelopes when filtered", () => {
    const geo = sampleGeomag(true);
    const pol: PolarizationResponse = {
      ...samplePolarization(),
      method: "variant_b_sos",
      trim_hours: 3,
      filtered: {
        timestamps_utc: geo.timestamps_utc,
        x_filt: [1.0, 2.0, 3.0],
        y_filt: [-1.0, -2.0, -3.0],
        z_filt: [0.2, 0.3, 0.4],
        x_amp: [2.5, 2.6, 2.7],
        y_amp: [1.8, 1.9, 2.0],
        usable_mask: [false, true, false],
      },
    };

    const ctxDefault = makeBaseContext({
      state: {
        ...INITIAL_ANALYSIS_STATE,
        method: "variant_b_sos",
        showFilteredMagnetogram: false,
      },
      geomag: geo,
      polarizationRaw: pol,
      maskedPolarization: applyPolarizationAlphaMask(pol, 0.05),
    });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctxDefault}>
          <MagnetogramPanel />
        </AnalysisContext.Provider>,
      );
    });
    const optDefault = mockCharts[0].lastOption as {
      series: Array<{ name: string; markArea?: { data: unknown[] } }>;
    };
    expect(optDefault.series[0].markArea?.data.length).toBeGreaterThanOrEqual(2);

    const ctxFiltered = makeBaseContext({
      state: {
        ...INITIAL_ANALYSIS_STATE,
        method: "variant_b_sos",
        showFilteredMagnetogram: true,
      },
      geomag: geo,
      polarizationRaw: pol,
      maskedPolarization: applyPolarizationAlphaMask(pol, 0.05),
    });
    act(() => {
      root.render(
        <AnalysisContext.Provider value={ctxFiltered}>
          <MagnetogramPanel />
        </AnalysisContext.Provider>,
      );
    });
    const optFiltered = mockCharts[0].lastOption as {
      series: Array<{ name: string }>;
    };
    const names = optFiltered.series.map((s) => s.name);
    expect(names).toContain("Огибающая |Z_X|");
    expect(names).toContain("Огибающая |Z_Y|");
  });

  it("DraftNullableNumberInput commits negative decimals and empty->null on Enter/blur", () => {
    const onCommit = vi.fn();
    act(() => {
      root.render(
        <DraftNullableNumberInput
          value={null}
          onCommit={onCommit}
          min={-180}
          max={180}
          step={0.1}
          ariaLabel="D0"
        />,
      );
    });

    const input = host.querySelector("input") as HTMLInputElement;
    const setInputValue = (val: string) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(input, val);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    };

    act(() => {
      setInputValue("-13.8");
    });
    expect(onCommit).not.toHaveBeenCalled();

    act(() => {
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });
    expect(onCommit).toHaveBeenCalledWith(-13.8);

    act(() => {
      root.render(
        <DraftNullableNumberInput
          value={-13.8}
          onCommit={onCommit}
          min={-180}
          max={180}
          step={0.1}
          ariaLabel="D0"
        />,
      );
    });
    act(() => {
      setInputValue("");
      input.dispatchEvent(new FocusEvent("focusout", { bubbles: true }));
    });
    expect(onCommit).toHaveBeenLastCalledWith(null);
  });

  it("isolates CursorProvider updates from useAnalysis consumers and aborts/clears stale fetches", async () => {
    let sidebarRenders = 0;
    let cursorRenders = 0;
    let triggerCursor!: (ms: number) => void;
    let latestCtx!: AnalysisContextValue;

    const FakeSidebar = () => {
      useAnalysis();
      sidebarRenders += 1;
      return <div data-testid="sidebar" />;
    };
    const CursorConsumer = () => {
      const { cursorUtcMs, setCursorUtcMs } = useCursor();
      latestCtx = useAnalysis();
      triggerCursor = setCursorUtcMs;
      cursorRenders += 1;
      return <div data-testid="cursor">{cursorUtcMs}</div>;
    };

    const healthResp: HealthResponse = {
      status: "ok",
      core_version: "0.1.0",
      catalog_stations: 50,
      catalog_episodes: 4,
      require_auth: false,
    };
    const catResp: CatalogResponse = {
      episodes: [
        {
          episode_id: "storm_20261009_11",
          label: "Oct 9-11",
          start_utc: "2026-10-09T00:00:00Z",
          end_utc: "2026-10-11T23:59:00Z",
          n_minutes: 4320,
          n_days: 3,
          omni_available: true,
          oulu_baseline_cpm: 6000,
        },
      ],
      stations: [],
      availability: [],
      metadata: {},
    };

    const flrAborted: boolean[] = [];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/auth/session")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ authenticated: true, session_token: "t", require_auth: false }),
            { status: 200 },
          ),
        );
      }
      if (url.includes("/api/health")) {
        return Promise.resolve(new Response(JSON.stringify(healthResp), { status: 200 }));
      }
      if (url.includes("/api/catalog")) {
        return Promise.resolve(new Response(JSON.stringify(catResp), { status: 200 }));
      }
      if (url.includes("/api/jobs") && !url.includes("/result")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
      }
      if (url.includes("/api/cache")) {
        return Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
      }
      if (url.includes("/api/timeseries/space-weather")) {
        return Promise.resolve(new Response(JSON.stringify(sampleSpaceWeather()), { status: 200 }));
      }
      if (url.includes("/api/timeseries/geomag")) {
        return Promise.resolve(new Response(JSON.stringify(sampleGeomag(true)), { status: 200 }));
      }
      if (url.includes("/api/polarization")) {
        return Promise.resolve(new Response(JSON.stringify(samplePolarization()), { status: 200 }));
      }
      if (url.includes("/api/cross/welch/flr")) {
        const callIdx = flrAborted.length;
        flrAborted.push(false);
        init?.signal?.addEventListener("abort", () => {
          flrAborted[callIdx] = true;
        });
        return new Promise<Response>((resolve, reject) => {
          if (callIdx === 0) {
            init?.signal?.addEventListener("abort", () => {
              reject(new DOMException("Aborted", "AbortError"));
            });
          } else {
            resolve(
              new Response(
                JSON.stringify({
                  episode_id: "storm_20261009_11",
                  band: "Pc5",
                  window_timestamp_utc: "2026-10-10T06:00:00Z",
                  station_codes: ["LVV", "BEL"],
                  latitudes_deg: [49.9, 51.84],
                  band_power: [10, 12],
                  amplitude: [3.1, 3.4],
                  pair_coherence: [0.5],
                  pair_gamma2_crit: [0.06],
                  pair_phase_deg: [-15],
                  pair_is_significant: [true],
                  cumulative_phase_deg: [0, 15],
                  phase_gradient_deg_per_lat: [7.7],
                  peak_station: "BEL",
                  peak_latitude_deg: 51.84,
                }),
                { status: 200 },
              ),
            );
          }
        });
      }
      return Promise.resolve(new Response("{}", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    await act(async () => {
      root.render(
        <AnalysisProvider>
          <FakeSidebar />
          <CursorConsumer />
        </AnalysisProvider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20));
    });

    const sidebarBeforeCursor = sidebarRenders;
    const cursorBeforeHover = cursorRenders;
    act(() => {
      triggerCursor(Date.UTC(2026, 9, 10, 9, 0));
    });
    expect(cursorRenders).toBe(cursorBeforeHover + 1);
    expect(sidebarRenders).toBe(sidebarBeforeCursor);

    // Concurrent loadFlrAtCursor calls abort the first in-flight request cleanly
    await act(async () => {
      const p1 = latestCtx.loadFlrAtCursor(1);
      const p2 = latestCtx.loadFlrAtCursor(2);
      await Promise.all([p1, p2]);
    });
    expect(flrAborted[0]).toBe(true);
    expect(latestCtx.flrProfile?.peak_station).toBe("BEL");
    expect(latestCtx.flrError).toBeNull();

    vi.unstubAllGlobals();
  });
});
