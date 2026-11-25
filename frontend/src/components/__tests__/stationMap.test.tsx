import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  CatalogResponse,
  NetworkSnapshotResponse,
  StationInfo,
  WelchMatrixResponse,
  WelchPairResponse,
} from "../../api/types";
import { AnalysisContext, type AnalysisContextValue } from "../../store/AnalysisContext";
import { INITIAL_ANALYSIS_STATE } from "../../store/analysisReducer";
import * as exportUtils from "../../utils/exportUtils";
import { buildWelchPairRequestSignature, computeCursorWindowIndex } from "../../utils/masking";
import { pCrit, gamma2Crit, scaleGamma2CritForAlpha } from "../../utils/thresholds";
import { ALL_49_STATIONS } from "../../utils/__tests__/fixtures/stations49";
import { StationMapPanel } from "../StationMap/StationMapPanel";

function build49Catalog(): CatalogResponse {
  const stations: StationInfo[] = ALL_49_STATIONS.map((s) => ({
    iaga_code: s.iaga,
    station_name: s.iaga,
    station_type: "magnetometer",
    region: s.region,
    lat_deg: s.lat,
    lon_deg: s.lon,
    elev_m: 100,
    orig_format: "XYZF",
    has_baseline: s.hasBaseline,
    mean_vec_norm_nT: s.hasBaseline ? 48000 : 500,
    days_present: 10,
    source_of_data: "INTERMAGNET",
    cutoff_rigidity_gv: null,
    barometric_coeff: null,
  }));
  stations.push({
    iaga_code: "OULU",
    station_name: "Oulu NM",
    station_type: "neutron_monitor",
    region: "Europe",
    lat_deg: 65.0544,
    lon_deg: 25.4681,
    elev_m: 15,
    orig_format: "CSV",
    has_baseline: true,
    mean_vec_norm_nT: null,
    days_present: 10,
    source_of_data: "OULU",
    cutoff_rigidity_gv: 0.8,
    barometric_coeff: 0.74,
  });

  return {
    episodes: [
      {
        episode_id: "storm_20261009_11",
        label: "Oct 9-11, 2026",
        start_utc: "2026-10-09T00:00:00Z",
        end_utc: "2026-10-11T23:59:00Z",
        n_minutes: 4320,
        n_days: 3,
        omni_available: true,
        oulu_baseline_cpm: 6000,
      },
    ],
    stations,
    availability: [],
    metadata: {},
  };
}

function buildSnapshotResponse(): NetworkSnapshotResponse {
  const nEff = 45;
  const pc05 = pCrit(nEff, 0.05);
  const pc01 = pCrit(nEff, 0.01);
  const borderlineP = 0.5 * (pc05 + pc01); // significant at 0.05, masked at 0.01

  return {
    episode_id: "storm_20261009_11",
    band: "Pc5",
    coord_frame: "XYZ",
    stations: ALL_49_STATIONS.map((s) => {
      if (s.iaga === "SUA") {
        return {
          station_iaga: s.iaga,
          lat_deg: s.lat,
          lon_deg: s.lon,
          has_baseline: s.hasBaseline,
          available: false,
          is_valid: false,
          d0_deg: 0,
          stokes_2d: null,
          warning: "Absent in episode",
        };
      }
      const pVal =
        s.iaga === "PAG" ? borderlineP : s.iaga === "GCK" ? 0.1 : 0.82;
      const isCw = s.lon >= 0;
      return {
        station_iaga: s.iaga,
        lat_deg: s.lat,
        lon_deg: s.lon,
        has_baseline: s.hasBaseline,
        available: true,
        is_valid: true,
        d0_deg: 5.0,
        stokes_2d: {
          I: 180,
          Q: 40,
          U: 30,
          V: isCw ? 90 : -90,
          A: 50,
          R: 103,
          p: pVal,
          p_L: 0.28,
          p_C: 0.5,
          psi_deg: 18.4,
          chi_deg: isCw ? 30 : -30,
          epsilon: isCw ? 0.5 : -0.5,
          gamma2: 0.65,
          arg_sxy_deg: 55,
          n_eff: nEff,
          p_crit: pc05,
          gamma2_crit: gamma2Crit(nEff, 0.05),
          is_significant_p: pVal > pc05,
          is_significant_gamma2: true,
        },
        warning: null,
      };
    }),
  };
}

function makeCtx(
  catalog: CatalogResponse,
  overrides: Partial<AnalysisContextValue> = {},
): AnalysisContextValue {
  const stationMetaMap = new Map(catalog.stations.map((s) => [s.iaga_code, s]));
  const mergedState = {
    ...INITIAL_ANALYSIS_STATE,
    stationA: "LVV",
    stationB: "BEL",
    viewRangeMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)] as [number, number],
    cursorUtcMs: Date.UTC(2026, 9, 10, 12, 0),
    ...overrides.state,
  };
  const welch: WelchPairResponse = {
    episode_id: "storm_20261009_11",
    station_i: "LVV",
    station_j: "BEL",
    component: "X",
    coord_frame: "XYZ",
    band: "Pc5",
    available: true,
    cache_hits: { LVV: true, BEL: true },
    window_timestamps_utc: ["2026-10-10T12:00:00Z"],
    freqs_hz: [],
    K_ij: [4],
    n_eff: [45],
    gamma2: [0.74],
    gamma2_crit: [0.066],
    phase_deg: [22],
    phase_deg_masked: [22],
    is_significant: [true],
    is_valid: [true],
    gamma2_tf: [],
    gamma2_crit_tf: [],
    phase_deg_tf: [],
    phase_deg_tf_masked: [],
    azimuthal: { m: [2.5], m_alias: 18, delta_lon_deg: 3.01, is_meridional_guarded: false },
    warnings: [],
  };
  const defaultSig =
    overrides.welchPairSig !== undefined
      ? overrides.welchPairSig
      : overrides.welchPairRaw === null
        ? null
        : buildWelchPairRequestSignature(mergedState);

  return {
    state: mergedState,
    dispatch: vi.fn(),
    auth: { checked: true, authenticated: true, requireAuth: true, error: null },
    health: null,
    catalog,
    activeEpisode: catalog.episodes[0],
    episodeBoundsMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)],
    stationMetaMap,
    stationAvailabilityMap: new Map(),
    geomag: null,
    rawSpaceWeather: null,
    shiftedSolar1: null,
    polarizationRaw: null,
    maskedPolarization: null,
    welchPairRaw: welch,
    welchPairSig: defaultSig,
    maskedWelch: {
      gamma2Crit: [0.066],
      isSignificant: [true],
      phaseDegMasked: [22],
      azimuthalMMasked: [2.5],
      gamma2CritTf: [],
      phaseDegTfMasked: [],
    },
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

describe("StationMapPanel offline D3-geo map & interactive selection", () => {
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
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    host.remove();
    vi.unstubAllGlobals();
  });

  it("renders all 49 stations, MLT=12, terminator, OULU offline and re-masks alpha without extra backend calls", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    let snapshotCalls = 0;

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/polarization/network-snapshot")) {
        snapshotCalls += 1;
        return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
      }
      return Promise.resolve(new Response("{}", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const dispatch = vi.fn();
    const ctx05 = makeCtx(catalog, { dispatch });

    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctx05}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });

    // Zero external CDN / tile requests; strictly 1 local snapshot call
    expect(snapshotCalls).toBe(1);
    const stationNodes = host.querySelectorAll('[data-testid^="station-node-"]');
    expect(stationNodes.length).toBe(49);
    expect(host.querySelector('[data-testid="station-marker-OULU"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="map-terminator"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="map-mlt12"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="active-coherence-arc"]')).not.toBeNull();
    const htmlWithoutXmlns = host.innerHTML
      .replace(/http:\/\/www\.w3\.org\/2000\/svg/g, "")
      .replace(/http:\/\/www\.w3\.org\/1999\/xlink/g, "");
    expect(htmlWithoutXmlns).not.toMatch(/https?:\/\/(?!127\.0\.0\.1)/);

    // Check CW (Europe), CCW (Canada), unavailable (SUA), insignificant (GCK), borderline significant (PAG at alpha=0.05)
    expect(host.querySelector('[data-testid="station-node-LVV"]')?.getAttribute("data-sense")).toBe("CW");
    expect(host.querySelector('[data-testid="station-node-OTT"]')?.getAttribute("data-sense")).toBe("CCW");
    expect(host.querySelector('[data-testid="station-node-SUA"]')?.getAttribute("data-status")).toBe("unavailable");
    expect(host.querySelector('[data-testid="station-node-GCK"]')?.getAttribute("data-status")).toBe("insignificant");
    expect(host.querySelector('[data-testid="station-node-PAG"]')?.getAttribute("data-status")).toBe("significant");

    // 2-click pair selection: LVV (lon 23.8) then OTT (lon -75.552) -> canonical West-first OTT -> LVV
    const lvvEl = host.querySelector('[data-testid="station-node-LVV"]') as Element;
    const ottEl = host.querySelector('[data-testid="station-node-OTT"]') as Element;
    act(() => {
      lvvEl.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(dispatch).toHaveBeenCalledWith({
      type: "SET_STATION_A",
      stationIaga: "LVV",
      recordRecent: false,
    });

    act(() => {
      ottEl.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(dispatch).toHaveBeenCalledWith({
      type: "SET_PAIR",
      stationA: "OTT",
      stationB: "LVV",
    });

    // Instant client-side alpha change 0.05 -> 0.01 masks borderline PAG with ZERO extra network calls
    const ctx01 = makeCtx(catalog, {
      dispatch,
      state: { ...ctx05.state, alpha: 0.01 },
    });
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctx01}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });
    expect(snapshotCalls).toBe(1);
    expect(host.querySelector('[data-testid="station-node-PAG"]')?.getAttribute("data-status")).toBe("insignificant");
  });

  it("clears stale local pair metrics when switching to a baseless station in XmYm (F-01)", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    let pairCalls = 0;

    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/polarization/network-snapshot")) {
        return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
      }
      if (url.includes("/api/cross/welch/pair")) {
        pairCalls += 1;
        return Promise.resolve(
          new Response(
            JSON.stringify({
              episode_id: "storm_20261009_11",
              station_i: "LVV",
              station_j: "BEL",
              component: "X",
              coord_frame: "XYZ",
              band: "Pc5",
              available: true,
              cache_hits: { LVV: true, BEL: true },
              window_timestamps_utc: ["2026-10-10T12:00:00Z"],
              freqs_hz: [],
              K_ij: [4],
              n_eff: [45],
              gamma2: [0.74],
              gamma2_crit: [0.066],
              phase_deg: [22],
              phase_deg_masked: [22],
              is_significant: [true],
              is_valid: [true],
              gamma2_tf: [],
              gamma2_crit_tf: [],
              phase_deg_tf: [],
              phase_deg_tf_masked: [],
              azimuthal: { m: [2.5], m_alias: 18, delta_lon_deg: 3.01, is_meridional_guarded: false },
              warnings: [],
            } satisfies WelchPairResponse),
            { status: 200 },
          ),
        );
      }
      return Promise.resolve(new Response("{}", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    // 1. Spectrogram tab: local pair fetch LVV-BEL populates γ²=0.74
    const ctxXyz = makeCtx(catalog, {
      welchPairRaw: null,
      welchPairSig: null,
      maskedWelch: null,
      state: {
        ...INITIAL_ANALYSIS_STATE,
        activeCrossTab: "spectrogram",
        stationA: "LVV",
        stationB: "BEL",
        coordFrame: "XYZ",
        viewRangeMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)],
        cursorUtcMs: Date.UTC(2026, 9, 10, 12, 0),
      },
    });
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctxXyz}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });
    expect(pairCalls).toBe(1);
    expect(host.innerHTML).toContain("LVV–BEL: γ²=0.74");

    // 2. Switch to baseless GCK in XmYm without externalD0DegB -> must NOT leak LVV-BEL γ²=0.74
    const ctxBlocked = makeCtx(catalog, {
      welchPairRaw: null,
      welchPairSig: null,
      maskedWelch: null,
      state: {
        ...ctxXyz.state,
        stationA: "LVV",
        stationB: "GCK",
        coordFrame: "XmYm",
        crossComponent: "Xm",
        externalD0DegB: null,
      },
    });
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctxBlocked}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });
    expect(pairCalls).toBe(1);
    expect(host.innerHTML).not.toContain("γ²=0.74");
    expect(host.innerHTML).toContain("LVV ↔ GCK");
  });

  it("refetches local Welch pair when Welch params change on spectrogram tab despite stale context welchPairRaw (F-01)", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    let capturedBody: Record<string, unknown> | null = null;

    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/polarization/network-snapshot")) {
        return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
      }
      if (url.includes("/api/cross/welch/pair")) {
        capturedBody = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : null;
        return Promise.resolve(
          new Response(
            JSON.stringify({
              episode_id: "storm_20261009_11",
              station_i: "LVV",
              station_j: "BEL",
              component: "X",
              coord_frame: "XYZ",
              band: "Pc5",
              available: true,
              cache_hits: { LVV: true, BEL: true },
              window_timestamps_utc: ["2026-10-10T12:00:00Z"],
              freqs_hz: [],
              K_ij: [4],
              n_eff: [45],
              gamma2: [0.41],
              gamma2_crit: [0.095],
              phase_deg: [12],
              phase_deg_masked: [12],
              is_significant: [true],
              is_valid: [true],
              gamma2_tf: [],
              gamma2_crit_tf: [],
              phase_deg_tf: [],
              phase_deg_tf_masked: [],
              azimuthal: { m: [1.4], m_alias: 18, delta_lon_deg: 3.01, is_meridional_guarded: false },
              warnings: [],
            } satisfies WelchPairResponse),
            { status: 200 },
          ),
        );
      }
      return Promise.resolve(new Response("{}", { status: 200 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const oldState = {
      ...INITIAL_ANALYSIS_STATE,
      activeCrossTab: "welch" as const,
      stationA: "LVV",
      stationB: "BEL",
      L: 128,
      windowSamples: 360,
      tMaxPc6Sec: 1800,
      spectrumType: "flat" as const,
    };
    const newState = {
      ...oldState,
      activeCrossTab: "spectrogram" as const,
      L: 256,
      windowSamples: 720,
      tMaxPc6Sec: 900,
      spectrumType: "red" as const,
      viewRangeMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)] as [number, number],
      cursorUtcMs: Date.UTC(2026, 9, 10, 12, 0),
    };

    const ctx = makeCtx(catalog, {
      state: newState,
      welchPairSig: buildWelchPairRequestSignature(oldState),
    });

    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctx}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });

    expect(capturedBody).not.toBeNull();
    expect(capturedBody?.["L"]).toBe(256);
    expect(capturedBody?.["window_samples"]).toBe(720);
    expect(capturedBody?.["t_max_pc6_sec"]).toBe(900);
    expect(capturedBody?.["spectrum_type"]).toBe("red");
    expect(host.innerHTML).toContain("LVV–BEL: γ²=0.41");
    expect(host.innerHTML).not.toContain("γ²=0.74");
  });

  it("removes manual Compute FLR button, renders Updating status, and supports Retry with cursor window_index on flrError (F-04)", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/polarization/network-snapshot")) {
          return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
        }
        if (url.includes("/api/cross/welch/pair")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                episode_id: "storm_20261009_11",
                station_i: "LVV",
                station_j: "BEL",
                component: "X",
                coord_frame: "XYZ",
                band: "Pc5",
                available: false,
                cache_hits: {},
                window_timestamps_utc: [],
                freqs_hz: [],
                K_ij: [],
                n_eff: [],
                gamma2: [],
                gamma2_crit: [],
                phase_deg: [],
                phase_deg_masked: [],
                is_significant: [],
                is_valid: [],
                gamma2_tf: [],
                gamma2_crit_tf: [],
                phase_deg_tf: [],
                phase_deg_tf_masked: [],
                azimuthal: null,
                warnings: [],
              } satisfies WelchPairResponse),
              { status: 200 },
            ),
          );
        }
        return Promise.resolve(new Response("{}", { status: 200 }));
      }),
    );

    const loadFlrAtCursor = vi.fn().mockResolvedValue(undefined);
    const setMapFlrChainActive = vi.fn();
    const cursorMs = Date.UTC(2026, 9, 10, 12, 0, 0);
    const expectedIdx = computeCursorWindowIndex(
      cursorMs,
      [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)],
      360,
      30,
    );
    expect(expectedIdx).toBeGreaterThan(0);

    const ctx = makeCtx(catalog, {
      welchPairRaw: null,
      welchPairSig: null,
      maskedWelch: null,
      flrError: "Station HRN unavailable",
      loadingFlr: true,
      loadFlrAtCursor,
      setMapFlrChainActive,
      state: {
        ...INITIAL_ANALYSIS_STATE,
        activeCrossTab: "spectrogram",
        cursorUtcMs: cursorMs,
      },
    });

    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctx}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });

    expect(host.textContent).toContain("Updating…");
    const errBanner = host.querySelector('[data-testid="map-flr-error"]');
    expect(errBanner).not.toBeNull();
    expect(errBanner?.textContent).toContain("Station HRN unavailable");

    const chainTabBtn = Array.from(host.querySelectorAll("button")).find(
      (b) =>
        b.textContent?.trim() === "FLR Chain" &&
        b.getAttribute("title")?.includes("meridional FLR chain"),
    );
    expect(chainTabBtn).toBeDefined();
    await act(async () => {
      (chainTabBtn as HTMLButtonElement).click();
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 40));
    });

    expect(setMapFlrChainActive).toHaveBeenCalledWith(true);
    const flrBtn = Array.from(host.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Compute FLR"),
    );
    expect(flrBtn).toBeUndefined();

    const retryBtn = Array.from(errBanner?.querySelectorAll("button") ?? []).find(
      (b) => b.textContent?.trim() === "Retry",
    );
    expect(retryBtn).toBeDefined();
    // Re-render with loadingFlr: false so Retry button is enabled
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={{ ...ctx, loadingFlr: false }}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    const enabledRetry = Array.from(
      host.querySelector('[data-testid="map-flr-error"]')?.querySelectorAll("button") ?? [],
    ).find((b) => b.textContent?.trim() === "Retry") as HTMLButtonElement;
    await act(async () => {
      enabledRetry.click();
    });

    expect(loadFlrAtCursor).toHaveBeenCalledWith(expectedIdx);
  });

  it("preserves unmasked matrix phase_deg when relaxing alpha to 0.1 in network-from-A mode (F-05)", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    const g2Crit05 = 0.09;
    const g2Val = 0.08; // Masked at 0.05, significant at 0.1 (scaled crit ~0.070)
    expect(g2Val).toBeGreaterThan(scaleGamma2CritForAlpha(g2Crit05, 0.05, 0.1));

    const matResp: WelchMatrixResponse = {
      episode_id: "storm_20261009_11",
      component: "X",
      coord_frame: "XYZ",
      band: "Pc5",
      station_codes: ["LVV", "PAG"],
      window_timestamps_utc: ["2026-10-10T12:00:00Z"],
      gamma2: [
        [
          [1, g2Val],
          [g2Val, 1],
        ],
      ],
      gamma2_crit: [
        [
          [g2Crit05, g2Crit05],
          [g2Crit05, g2Crit05],
        ],
      ],
      phase_deg: [
        [
          [0, 28.5],
          [-28.5, 0],
        ],
      ],
      phase_deg_masked: [
        [
          [null, null],
          [null, null],
        ],
      ],
      K_ij: [
        [
          [4, 4],
          [4, 4],
        ],
      ],
      is_significant: [
        [
          [false, false],
          [false, false],
        ],
      ],
      cache_hits: { LVV: true, PAG: true },
      skipped_stations: {},
      elapsed_ms: 5,
    };

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/polarization/network-snapshot")) {
          return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
        }
        if (url.includes("/api/cross/welch/matrix")) {
          return Promise.resolve(new Response(JSON.stringify(matResp), { status: 200 }));
        }
        return Promise.resolve(new Response("{}", { status: 200 }));
      }),
    );

    const ctx10 = makeCtx(catalog, {
      state: {
        ...INITIAL_ANALYSIS_STATE,
        stationA: "LVV",
        stationB: "BEL",
        alpha: 0.1,
        viewRangeMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)],
        cursorUtcMs: Date.UTC(2026, 9, 10, 12, 0),
      },
    });

    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctx10}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });

    const netBtn = Array.from(host.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Network from LVV"),
    );
    expect(netBtn).toBeDefined();
    act(() => {
      netBtn!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    await act(async () => {
      await new Promise((r) => setTimeout(r, 290));
    });

    const matrixGroup = host.querySelector('[data-arc-key="arc:LVV-PAG"]');
    expect(matrixGroup).not.toBeNull();
    act(() => {
      matrixGroup!.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    });

    const tooltip = host.querySelector(".map-tooltip");
    expect(tooltip).not.toBeNull();
    expect(tooltip?.textContent).toContain("Δφ=28.5°");
    expect(tooltip?.textContent).not.toContain("masked");
  });

  it("drops stale snapshot and matrix arcs immediately when parameters change or snapshot request fails (F-02)", async () => {
    const catalog = build49Catalog();
    const snapshot = buildSnapshotResponse();
    // Make GCK significant in XYZ snapshot so transition to XmYm + 500 error proves no stale XYZ leak
    const gckEntry = snapshot.stations.find((s) => s.station_iaga === "GCK")!;
    gckEntry.stokes_2d!.p = 0.85;
    gckEntry.stokes_2d!.is_significant_p = true;

    const g2Crit05 = 0.09;
    const g2Val = 0.72;
    const matResp: WelchMatrixResponse = {
      episode_id: "storm_20261009_11",
      component: "X",
      coord_frame: "XYZ",
      band: "Pc5",
      station_codes: ["LVV", "PAG"],
      window_timestamps_utc: ["2026-10-10T12:00:00Z"],
      gamma2: [
        [
          [1, g2Val],
          [g2Val, 1],
        ],
      ],
      gamma2_crit: [
        [
          [g2Crit05, g2Crit05],
          [g2Crit05, g2Crit05],
        ],
      ],
      phase_deg: [
        [
          [0, 18.0],
          [-18.0, 0],
        ],
      ],
      phase_deg_masked: [
        [
          [0, 18.0],
          [-18.0, 0],
        ],
      ],
      K_ij: [
        [
          [4, 4],
          [4, 4],
        ],
      ],
      is_significant: [
        [
          [true, true],
          [true, true],
        ],
      ],
      cache_hits: { LVV: true, PAG: true },
      skipped_stations: {},
      elapsed_ms: 4,
    };

    let failSnapshot = false;
    let failMatrix = false;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/polarization/network-snapshot")) {
          if (failSnapshot) {
            return Promise.resolve(
              new Response(JSON.stringify({ detail: "Snapshot backend error 500" }), {
                status: 500,
              }),
            );
          }
          return Promise.resolve(new Response(JSON.stringify(snapshot), { status: 200 }));
        }
        if (url.includes("/api/cross/welch/matrix")) {
          if (failMatrix) {
            return Promise.resolve(
              new Response(JSON.stringify({ detail: "Matrix error 500" }), { status: 500 }),
            );
          }
          return Promise.resolve(new Response(JSON.stringify(matResp), { status: 200 }));
        }
        if (url.includes("/api/cross/welch/pair")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                episode_id: "storm_20261009_11",
                station_i: "LVV",
                station_j: "BEL",
                component: "Xm",
                coord_frame: "XmYm",
                band: "Pc6",
                available: false,
                cache_hits: {},
                window_timestamps_utc: [],
                freqs_hz: [],
                K_ij: [],
                n_eff: [],
                gamma2: [],
                gamma2_crit: [],
                phase_deg: [],
                phase_deg_masked: [],
                is_significant: [],
                is_valid: [],
                gamma2_tf: [],
                gamma2_crit_tf: [],
                phase_deg_tf: [],
                phase_deg_tf_masked: [],
                azimuthal: null,
                warnings: [],
              } satisfies WelchPairResponse),
              { status: 200 },
            ),
          );
        }
        return Promise.resolve(new Response("{}", { status: 200 }));
      }),
    );

    const csvSpy = vi.spyOn(exportUtils, "downloadTextFile").mockImplementation(() => {});

    // 1. Initial XYZ + Pc5 state: GCK ellipse is significant, matrix arc LVV-PAG renders
    const ctxXyz = makeCtx(catalog, {
      state: {
        ...INITIAL_ANALYSIS_STATE,
        stationA: "LVV",
        stationB: "BEL",
        coordFrame: "XYZ",
        band: "Pc5",
        viewRangeMs: [Date.UTC(2026, 9, 9, 0, 0), Date.UTC(2026, 9, 11, 23, 59)],
        cursorUtcMs: Date.UTC(2026, 9, 10, 12, 0),
      },
    });
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctxXyz}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    const netBtn = Array.from(host.querySelectorAll("button")).find((b) =>
      b.textContent?.includes("Network from LVV"),
    );
    expect(netBtn).toBeDefined();
    act(() => {
      netBtn!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 290));
    });
    expect(host.querySelector('[data-testid="station-node-GCK"]')?.getAttribute("data-status")).toBe("significant");
    expect(host.querySelector('[data-arc-key="arc:LVV-PAG"]')).not.toBeNull();

    // 2. Switch to XmYm + Pc6 while snapshot & matrix fetches fail -> must drop stale GCK ellipse & LVV-PAG arc immediately
    failSnapshot = true;
    failMatrix = true;
    const ctxXmyErr = makeCtx(catalog, {
      welchPairRaw: null,
      welchPairSig: null,
      maskedWelch: null,
      state: {
        ...ctxXyz.state,
        coordFrame: "XmYm",
        crossComponent: "Xm",
        band: "Pc6",
      },
    });
    await act(async () => {
      root.render(
        <AnalysisContext.Provider value={ctxXmyErr}>
          <StationMapPanel />
        </AnalysisContext.Provider>,
      );
    });
    // Immediate guard before debounce resolves: no stale GCK ellipse or LVV-PAG matrix arc
    expect(host.querySelector('[data-testid="station-node-GCK"]')?.getAttribute("data-status")).toBe("insignificant");
    expect(host.querySelector('[data-arc-key="arc:LVV-PAG"]')).toBeNull();

    await act(async () => {
      await new Promise((r) => setTimeout(r, 290));
    });
    expect(host.textContent).toContain("Snapshot backend error 500");
    expect(host.querySelector('[data-testid="station-node-GCK"]')?.getAttribute("data-status")).toBe("insignificant");
    expect(host.querySelector('[data-arc-key="arc:LVV-PAG"]')).toBeNull();

    // Exported CSV must also contain coord_frame: XmYm with NO stale XYZ Stokes numbers on GCK
    const csvBtn = Array.from(host.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "CSV",
    );
    expect(csvBtn).toBeDefined();
    act(() => {
      (csvBtn as HTMLButtonElement).click();
    });
    expect(csvSpy).toHaveBeenCalledTimes(1);
    const csvContent = String(csvSpy.mock.calls[0][1]);
    expect(csvContent).toContain("# coord_frame: XmYm");
    const gckLine = csvContent.split("\n").find((l) => l.startsWith("GCK,"));
    expect(gckLine).toBeDefined();
    expect(gckLine).not.toContain(",180,");
    csvSpy.mockRestore();
  });
});
