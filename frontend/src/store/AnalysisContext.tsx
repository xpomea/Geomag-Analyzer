import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import {
  ApiError,
  TERMINAL_JOB_STATUSES,
  cancelJob,
  computePolarization,
  computeWelchFlr,
  computeWelchPair,
  getCatalog,
  getGeomagTimeseries,
  getHealth,
  getSpaceWeatherTimeseries,
  getWtcJobResult,
  initAuthSession,
  listCacheEntries,
  listJobs,
  shutdownServer,
  submitBootstrapToken,
  submitWtcJob,
  subscribeJobProgress,
} from "../api/client";
import type {
  CatalogResponse,
  CoordFrame,
  CrossComponent,
  EpisodeInfo,
  GeomagTimeseriesResponse,
  HealthResponse,
  JobStatusResponse,
  MeridionalFLRResponse,
  PolarizationResponse,
  SpaceWeatherResponse,
  StationInfo,
  WTCJobRequest,
  WTCResultResponse,
  WelchPairResponse,
} from "../api/types";
import { applySolar1Shift, type ShiftedSolar1Output } from "../utils/ballisticShift";
import {
  applyFlrAlphaMask,
  applyPolarizationAlphaMask,
  applyWelchAlphaMask,
  buildWelchPairRequestSignature,
  computeCursorWindowIndex,
  type MaskedPolarizationView,
  type MaskedWelchView,
} from "../utils/masking";
import { CursorProvider, useCursor } from "./CursorContext";
import { loadRecentPairs, saveRecentPairs } from "../utils/recentPairs";
import {
  INITIAL_ANALYSIS_STATE,
  analysisReducer,
  getStationBaselineWarning,
  type AnalysisAction,
  type AnalysisState,
} from "./analysisReducer";

interface AuthStatusState {
  checked: boolean;
  authenticated: boolean;
  requireAuth: boolean;
  error: string | null;
}

export interface AnalysisContextValue {
  state: AnalysisState;
  dispatch: React.Dispatch<AnalysisAction>;
  auth: AuthStatusState;
  health: HealthResponse | null;
  catalog: CatalogResponse | null;
  activeEpisode: EpisodeInfo | null;
  episodeBoundsMs: [number, number] | null;
  stationMetaMap: Map<string, StationInfo>;
  stationAvailabilityMap: Map<string, { days: number; validPct: number; qualities: string[] }>;

  geomag: GeomagTimeseriesResponse | null;
  rawSpaceWeather: SpaceWeatherResponse | null;
  shiftedSolar1: ShiftedSolar1Output | null;
  polarizationRaw: PolarizationResponse | null;
  maskedPolarization: MaskedPolarizationView | null;
  welchPairRaw: WelchPairResponse | null;
  welchPairSig?: string | null;
  maskedWelch: MaskedWelchView | null;
  flrProfile: MeridionalFLRResponse | null;
  wtcResult: WTCResultResponse | null;

  loadingCatalog: boolean;
  loadingTimeseries: boolean;
  loadingPolarization: boolean;
  loadingWelch: boolean;
  loadingFlr: boolean;
  loadingWtcResult: boolean;

  globalError: string | null;
  polarizationError: string | null;
  welchError: string | null;
  flrError: string | null;
  wtcError: string | null;
  baselineWarning: string | null;

  authenticateWithToken: (token: string) => Promise<void>;
  refreshJobsAndCache: () => Promise<void>;
  launchWtcJob: (overrides?: Partial<WTCJobRequest>) => Promise<JobStatusResponse | null>;
  cancelWtcJob: (jobId: string) => Promise<void>;
  openWtcJobResult: (jobId: string, stride?: number) => Promise<void>;
  loadFlrAtCursor: (windowIndexOverride?: number) => Promise<void>;
  setMapFlrChainActive?: (active: boolean) => void;
  triggerShutdown: () => Promise<void>;
}

function buildFlrRequestSignature(state: AnalysisState, windowIndex: number): string {
  return JSON.stringify([
    state.episodeId,
    state.meridionalChain,
    state.crossComponent,
    state.coordFrame,
    state.band,
    state.tMaxPc6Sec,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.spectrumType,
    windowIndex,
  ]);
}

const FlrAutoSync: React.FC<{
  isNeeded: boolean;
  ready: boolean;
  episodeBoundsMs: [number, number] | null;
  state: AnalysisState;
  lastFetchedSigRef: React.MutableRefObject<string | null>;
  lastFailedSigRef: React.MutableRefObject<string | null>;
  executeFlrFetch: (windowIndexOverride?: number, autoSig?: string) => Promise<void>;
  abortAutoFlr: () => void;
}> = ({
  isNeeded,
  ready,
  episodeBoundsMs,
  state,
  lastFetchedSigRef,
  lastFailedSigRef,
  executeFlrFetch,
  abortAutoFlr,
}) => {
  const { cursorUtcMs } = useCursor();

  const effectiveCursorMs = useMemo(() => {
    if (cursorUtcMs !== null && Number.isFinite(cursorUtcMs)) return cursorUtcMs;
    if (state.cursorUtcMs !== null && Number.isFinite(state.cursorUtcMs)) return state.cursorUtcMs;
    if (episodeBoundsMs) return Math.round(0.5 * (episodeBoundsMs[0] + episodeBoundsMs[1]));
    return null;
  }, [cursorUtcMs, state.cursorUtcMs, episodeBoundsMs]);

  const windowIndex = useMemo(
    () =>
      computeCursorWindowIndex(
        effectiveCursorMs,
        episodeBoundsMs,
        state.windowSamples,
        state.stepSamples,
      ),
    [effectiveCursorMs, episodeBoundsMs, state.windowSamples, state.stepSamples],
  );

  const requestSig = useMemo(() => {
    if (!ready || !isNeeded || !episodeBoundsMs || state.meridionalChain.length < 2) {
      return null;
    }
    return buildFlrRequestSignature(state, windowIndex);
  }, [
    ready,
    isNeeded,
    episodeBoundsMs,
    state.episodeId,
    state.meridionalChain,
    state.crossComponent,
    state.coordFrame,
    state.band,
    state.tMaxPc6Sec,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.spectrumType,
    windowIndex,
  ]);

  useEffect(() => {
    if (!requestSig) return;
    if (
      lastFetchedSigRef.current === requestSig ||
      lastFailedSigRef.current === requestSig
    ) {
      return;
    }

    const timer = setTimeout(() => {
      void executeFlrFetch(windowIndex, requestSig);
    }, 220);

    return () => {
      clearTimeout(timer);
      abortAutoFlr();
    };
  }, [requestSig, windowIndex, executeFlrFetch, abortAutoFlr, lastFetchedSigRef, lastFailedSigRef]);

  return null;
};

export const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(
    analysisReducer,
    INITIAL_ANALYSIS_STATE,
    (init) => ({ ...init, recentPairs: loadRecentPairs() }),
  );
  const didMountRecentRef = useRef(false);
  useEffect(() => {
    if (!didMountRecentRef.current) {
      didMountRecentRef.current = true;
      return;
    }
    saveRecentPairs(state.recentPairs);
  }, [state.recentPairs]);

  const [auth, setAuth] = useState<AuthStatusState>({
    checked: false,
    authenticated: false,
    requireAuth: true,
    error: null,
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);

  const [geomag, setGeomag] = useState<GeomagTimeseriesResponse | null>(null);
  const [rawSpaceWeather, setRawSpaceWeather] = useState<SpaceWeatherResponse | null>(null);
  const [polarizationRaw, setPolarizationRaw] = useState<PolarizationResponse | null>(null);
  const [welchPairRaw, setWelchPairRaw] = useState<WelchPairResponse | null>(null);
  const [welchPairSig, setWelchPairSig] = useState<string | null>(null);
  const [flrProfileRaw, setFlrProfileRaw] = useState<MeridionalFLRResponse | null>(null);
  const [wtcResult, setWtcResult] = useState<WTCResultResponse | null>(null);

  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [loadingTimeseries, setLoadingTimeseries] = useState(false);
  const [loadingPolarization, setLoadingPolarization] = useState(false);
  const [loadingWelch, setLoadingWelch] = useState(false);
  const [loadingFlr, setLoadingFlr] = useState(false);
  const [loadingWtcResult, setLoadingWtcResult] = useState(false);
  const [mapFlrChainActive, setMapFlrChainActive] = useState(false);

  const [globalError, setGlobalError] = useState<string | null>(null);
  const [polarizationError, setPolarizationError] = useState<string | null>(null);
  const [welchError, setWelchError] = useState<string | null>(null);
  const [flrError, setFlrError] = useState<string | null>(null);
  const [wtcError, setWtcError] = useState<string | null>(null);

  const wsSubsRef = useRef<Map<string, () => void>>(new Map());
  const cursorRef = useRef<number | null>(state.cursorUtcMs);
  const jobsRef = useRef(state.jobs);
  jobsRef.current = state.jobs;
  const wtcAbortRef = useRef<AbortController | null>(null);
  const flrAbortRef = useRef<AbortController | null>(null);
  const flrAutoInFlightRef = useRef(false);
  const lastFetchedFlrSigRef = useRef<string | null>(null);
  const lastFailedFlrSigRef = useRef<string | null>(null);

  const refreshJobsAndCache = useCallback(async () => {
    try {
      const [jobsList, cacheList] = await Promise.all([
        listJobs(50),
        listCacheEntries(undefined, 100),
      ]);
      dispatch({ type: "SET_JOBS", jobs: jobsList });
      dispatch({ type: "SET_CACHE_ENTRIES", entries: cacheList });
    } catch {
      // silent during transient reload
    }
  }, []);

  // 1. Auth bootstrap on mount
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await initAuthSession();
        if (!active) return;
        setAuth({
          checked: true,
          authenticated: res.authenticated,
          requireAuth: res.requireAuth,
          error: null,
        });
      } catch (err) {
        if (!active) return;
        setAuth({
          checked: true,
          authenticated: false,
          requireAuth: true,
          error: err instanceof Error ? err.message : "Session initialization failed",
        });
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const authenticateWithToken = useCallback(async (token: string) => {
    setAuth((prev) => ({ ...prev, error: null }));
    try {
      const res = await submitBootstrapToken(token);
      setAuth({
        checked: true,
        authenticated: res.authenticated,
        requireAuth: true,
        error: null,
      });
    } catch (err) {
      setAuth((prev) => ({
        ...prev,
        authenticated: false,
        error: err instanceof ApiError ? err.detail : "Invalid or expired access token",
      }));
    }
  }, []);

  // 2. Load catalog, health, initial jobs/cache once authenticated
  useEffect(() => {
    if (!auth.checked || !auth.authenticated) return;
    const ctrl = new AbortController();
    setLoadingCatalog(true);
    setGlobalError(null);

    Promise.all([
      getHealth(ctrl.signal),
      getCatalog(ctrl.signal),
      listJobs(50, ctrl.signal),
      listCacheEntries(undefined, 100, ctrl.signal),
    ])
      .then(([h, cat, jobs, cache]) => {
        setHealth(h);
        setCatalog(cat);
        dispatch({ type: "SET_JOBS", jobs });
        dispatch({ type: "SET_CACHE_ENTRIES", entries: cache });
        const curEp =
          cat.episodes.find((ep) => ep.episode_id === state.episodeId) ?? cat.episodes[0];
        if (curEp && !state.viewRangeMs) {
          dispatch({ type: "SET_EPISODE", episodeId: curEp.episode_id, episode: curEp });
        }
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") {
          setGlobalError(err instanceof Error ? err.message : "Failed to load station catalog");
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) {
          setLoadingCatalog(false);
        }
      });

    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.checked, auth.authenticated]);

  const activeEpisode = useMemo(
    () => catalog?.episodes.find((e) => e.episode_id === state.episodeId) ?? null,
    [catalog, state.episodeId],
  );

  const episodeBoundsMs = useMemo<[number, number] | null>(() => {
    if (!activeEpisode) return null;
    const t0 = Date.parse(activeEpisode.start_utc);
    const t1 = Date.parse(activeEpisode.end_utc);
    return Number.isFinite(t0) && Number.isFinite(t1) ? [t0, t1] : null;
  }, [activeEpisode]);

  const stationMetaMap = useMemo(() => {
    const map = new Map<string, StationInfo>();
    if (catalog) {
      for (const st of catalog.stations) {
        map.set(st.iaga_code, st);
      }
    }
    return map;
  }, [catalog]);

  const stationAvailabilityMap = useMemo(() => {
    const map = new Map<string, { days: number; validPct: number; qualities: string[] }>();
    if (!catalog) return map;
    const grouped = new Map<string, { sumPct: number; days: number; q: Set<string> }>();
    for (const row of catalog.availability) {
      if (row.episode_id !== state.episodeId) continue;
      let g = grouped.get(row.station_iaga);
      if (!g) {
        g = { sumPct: 0, days: 0, q: new Set() };
        grouped.set(row.station_iaga, g);
      }
      g.sumPct += row.valid_vector_pct;
      g.days += 1;
      g.q.add(row.quality_code);
    }
    for (const [code, g] of grouped.entries()) {
      map.set(code, {
        days: g.days,
        validPct: g.days > 0 ? g.sumPct / g.days : 0,
        qualities: Array.from(g.q).sort(),
      });
    }
    return map;
  }, [catalog, state.episodeId]);

  const baselineWarning = useMemo(
    () =>
      getStationBaselineWarning(
        stationMetaMap.get(state.stationA),
        state.coordFrame,
        state.externalD0Deg,
      ),
    [stationMetaMap, state.stationA, state.coordFrame, state.externalD0Deg],
  );

  // 3. Fetch raw space weather once per episode (SOLAR-1 shift applied client-side!)
  useEffect(() => {
    if (!auth.checked || !auth.authenticated || !catalog) return;
    const ctrl = new AbortController();
    setLoadingTimeseries(true);

    getSpaceWeatherTimeseries(
      {
        episode_id: state.episodeId,
        solar1_shift: "l1_raw",
        include_omni: true,
        include_oulu: true,
      },
      ctrl.signal,
    )
      .then((sw) => {
        setRawSpaceWeather(sw);
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") {
          setGlobalError(err instanceof Error ? err.message : "Failed to load space weather time series");
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) {
          setLoadingTimeseries(false);
        }
      });

    return () => ctrl.abort();
  }, [auth.checked, auth.authenticated, catalog, state.episodeId]);

  // 4. Client-side ballistic/manual shift of SOLAR-1 (< 5 ms)
  const shiftedSolar1 = useMemo<ShiftedSolar1Output | null>(() => {
    if (!rawSpaceWeather) return null;
    return applySolar1Shift(rawSpaceWeather.solar1, state.solar1ShiftMode, {
      manualLagMin: state.manualLagMin,
      refDistanceKm: state.refDistanceKm,
      smoothWindowMin: state.smoothWindowMin,
    });
  }, [
    rawSpaceWeather,
    state.solar1ShiftMode,
    state.manualLagMin,
    state.refDistanceKm,
    state.smoothWindowMin,
  ]);

  // 5. Fetch magnetometer series + polarization (NOT dependent on alpha!)
  useEffect(() => {
    if (!auth.checked || !auth.authenticated || !catalog) return;
    if (baselineWarning) {
      setGeomag(null);
      setPolarizationRaw(null);
      setPolarizationError(baselineWarning);
      return;
    }

    const ctrl = new AbortController();
    setLoadingPolarization(true);
    setPolarizationError(null);
    setGlobalError(null);

    const geomagReq = getGeomagTimeseries(
      {
        episode_id: state.episodeId,
        station_iaga: state.stationA,
        coord_frame: state.coordFrame,
        external_d0_deg: state.externalD0Deg,
      },
      ctrl.signal,
    );

    const polReq = computePolarization(
      {
        episode_id: state.episodeId,
        station_iaga: state.stationA,
        method: state.method,
        band: state.band,
        t_max_pc6_sec: state.tMaxPc6Sec,
        coord_frame: state.coordFrame,
        external_d0_deg: state.externalD0Deg,
        alpha: 0.05,
        spectrum_type: state.spectrumType,
        include_3d: true,
        include_dynamic: true,
        include_filtered: state.method.startsWith("variant_b"),
        include_spectrogram: true,
        L: state.L,
        M: state.M,
        window_samples: state.windowSamples,
        step_samples: state.stepSamples,
        nw: state.nw,
        order_n: state.orderN,
        oversample_m: state.oversampleM,
        oversample_method: state.oversampleMethod,
      },
      ctrl.signal,
    );

    Promise.all([geomagReq, polReq])
      .then(([geo, pol]) => {
        setGeomag(geo);
        setPolarizationRaw(pol);
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") {
          setGeomag(null);
          setPolarizationRaw(null);
          setPolarizationError(
            err instanceof ApiError ? err.detail : (err as Error).message,
          );
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) {
          setLoadingPolarization(false);
        }
      });

    return () => ctrl.abort();
  }, [
    auth.checked,
    auth.authenticated,
    catalog,
    baselineWarning,
    state.episodeId,
    state.stationA,
    state.band,
    state.tMaxPc6Sec,
    state.method,
    state.coordFrame,
    state.externalD0Deg,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.nw,
    state.spectrumType,
    state.oversampleM,
    state.oversampleMethod,
    state.orderN,
  ]);

  // 6. Client-side dynamic threshold & noise masking (< 5 ms when alpha changes!)
  const maskedPolarization = useMemo<MaskedPolarizationView | null>(() => {
    if (!polarizationRaw) return null;
    return applyPolarizationAlphaMask(polarizationRaw, state.alpha);
  }, [polarizationRaw, state.alpha]);

  const currentWelchSig = useMemo(() => buildWelchPairRequestSignature(state), [
    state.episodeId,
    state.stationA,
    state.stationB,
    state.crossComponent,
    state.coordFrame,
    state.externalD0Deg,
    state.externalD0DegB,
    state.band,
    state.tMaxPc6Sec,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.spectrumType,
  ]);

  // 7. Fetch Welch pair when cross Welch tab is active (alpha NOT in deps!)
  useEffect(() => {
    if (!auth.checked || !auth.authenticated || !catalog) return;
    if (state.activeCrossTab !== "welch") return;

    const reqSig = currentWelchSig;
    const warnA = getStationBaselineWarning(
      stationMetaMap.get(state.stationA),
      state.coordFrame,
      state.externalD0Deg,
    );
    const warnB = getStationBaselineWarning(
      stationMetaMap.get(state.stationB),
      state.coordFrame,
      state.externalD0DegB,
    );
    if (warnA || warnB) {
      setWelchPairRaw(null);
      setWelchPairSig(null);
      setWelchError(warnA ?? warnB);
      return;
    }

    const ctrl = new AbortController();
    setLoadingWelch(true);
    setWelchError(null);

    computeWelchPair(
      {
        episode_id: state.episodeId,
        station_i: state.stationA,
        station_j: state.stationB,
        component: state.crossComponent,
        coord_frame: state.coordFrame,
        external_d0_deg_i: state.externalD0Deg,
        external_d0_deg_j: state.externalD0DegB,
        band: state.band,
        t_max_pc6_sec: state.tMaxPc6Sec,
        L: state.L,
        M: state.M,
        window_samples: state.windowSamples,
        step_samples: state.stepSamples,
        alpha: 0.05,
        spectrum_type: state.spectrumType,
      },
      ctrl.signal,
    )
      .then((res) => {
        setWelchPairRaw(res);
        setWelchPairSig(reqSig);
        void listCacheEntries(undefined, 100).then((entries) => {
          dispatch({ type: "SET_CACHE_ENTRIES", entries });
        });
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") {
          setWelchPairRaw(null);
          setWelchPairSig(null);
          setWelchError(err instanceof ApiError ? err.detail : (err as Error).message);
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) {
          setLoadingWelch(false);
        }
      });

    return () => ctrl.abort();
  }, [
    auth.checked,
    auth.authenticated,
    catalog,
    stationMetaMap,
    state.activeCrossTab,
    currentWelchSig,
    state.episodeId,
    state.stationA,
    state.stationB,
    state.crossComponent,
    state.coordFrame,
    state.externalD0Deg,
    state.externalD0DegB,
    state.band,
    state.tMaxPc6Sec,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.spectrumType,
  ]);

  const maskedWelch = useMemo<MaskedWelchView | null>(() => {
    if (!welchPairRaw) return null;
    return applyWelchAlphaMask(welchPairRaw, state.alpha, 0.05);
  }, [welchPairRaw, state.alpha]);

  // 8. Live WS / SSE stream subscriptions for active jobs
  const jobSubsSignature = useMemo(
    () =>
      state.jobs
        .filter((j) => !TERMINAL_JOB_STATUSES.has(j.status))
        .map((j) => j.job_id)
        .sort()
        .join(","),
    [state.jobs],
  );

  useEffect(() => {
    const activeIds = new Set(jobSubsSignature ? jobSubsSignature.split(",") : []);

    for (const [id, unsub] of wsSubsRef.current.entries()) {
      if (!activeIds.has(id)) {
        unsub();
        wsSubsRef.current.delete(id);
      }
    }

    for (const id of activeIds) {
      if (wsSubsRef.current.has(id)) continue;
      const unsub = subscribeJobProgress(id, {
        onTransport: (transport) => {
          dispatch({ type: "SET_JOB_TRANSPORT", jobId: id, transport });
        },
        onUpdate: (job) => {
          dispatch({ type: "UPSERT_JOB", job });
          if (TERMINAL_JOB_STATUSES.has(job.status)) {
            void listCacheEntries(undefined, 100).then((entries) => {
              dispatch({ type: "SET_CACHE_ENTRIES", entries });
            });
          }
        },
        onError: (msg) => {
          setWtcError(msg);
        },
        onClose: () => {
          wsSubsRef.current.delete(id);
        },
      });
      wsSubsRef.current.set(id, unsub);
    }
  }, [jobSubsSignature]);

  useEffect(() => {
    const subs = wsSubsRef.current;
    return () => {
      for (const unsub of subs.values()) {
        unsub();
      }
      subs.clear();
      wtcAbortRef.current?.abort();
      flrAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    setFlrProfileRaw(null);
    setFlrError(null);
    lastFetchedFlrSigRef.current = null;
    lastFailedFlrSigRef.current = null;
  }, [state.episodeId]);

  useEffect(() => {
    if (state.meridionalChain.length < 2) {
      setFlrProfileRaw(null);
      lastFetchedFlrSigRef.current = null;
    }
  }, [state.meridionalChain]);

  const flrProfile = useMemo<MeridionalFLRResponse | null>(() => {
    if (!flrProfileRaw) return null;
    return applyFlrAlphaMask(flrProfileRaw, state.alpha, 0.05);
  }, [flrProfileRaw, state.alpha]);

  const openWtcJobResult = useCallback(
    async (jobId: string, strideOverride?: number) => {
      const effStride = strideOverride ?? state.wtcStride;
      if (strideOverride !== undefined && strideOverride !== state.wtcStride) {
        dispatch({ type: "SET_WTC_STRIDE", stride: strideOverride });
      }
      wtcAbortRef.current?.abort();
      const ctrl = new AbortController();
      wtcAbortRef.current = ctrl;

      setLoadingWtcResult(true);
      setWtcError(null);
      try {
        const res = await getWtcJobResult(jobId, effStride, ctrl.signal);
        if (ctrl.signal.aborted) return;

        const ep = catalog?.episodes.find((e) => e.episode_id === res.episode_id);
        let bounds: [number, number] | null = null;
        if (ep) {
          bounds = [Date.parse(ep.start_utc), Date.parse(ep.end_utc)];
        } else if (res.timestamps_utc.length >= 2) {
          bounds = [
            Date.parse(res.timestamps_utc[0]),
            Date.parse(res.timestamps_utc[res.timestamps_utc.length - 1]),
          ];
        }
        const coordFrame: CoordFrame = res.coord_frame === "XmYm" ? "XmYm" : "XYZ";

        const jobObj = jobsRef.current.find((j) => j.job_id === jobId);
        let restoredD0A: number | null | undefined;
        let restoredD0B: number | null | undefined;
        if (coordFrame === "XmYm" && jobObj?.params) {
          const p = jobObj.params as Record<string, unknown>;
          const extA = p.external_d0_deg_i;
          const extB = p.external_d0_deg_j;
          const stA = stationMetaMap.get(res.station_i.toUpperCase());
          const stB = stationMetaMap.get(res.station_j.toUpperCase());
          if (typeof extA === "number" && Number.isFinite(extA)) {
            restoredD0A = extA;
          } else if (stA && !stA.has_baseline && typeof p.d0_deg_i === "number") {
            restoredD0A = p.d0_deg_i;
          } else {
            restoredD0A = null;
          }
          if (typeof extB === "number" && Number.isFinite(extB)) {
            restoredD0B = extB;
          } else if (stB && !stB.has_baseline && typeof p.d0_deg_j === "number") {
            restoredD0B = p.d0_deg_j;
          } else {
            restoredD0B = null;
          }
        }

        dispatch({
          type: "OPEN_WTC_JOB",
          jobId,
          episodeId: res.episode_id,
          stationA: res.station_i,
          stationB: res.station_j,
          component: res.component as CrossComponent,
          coordFrame,
          externalD0Deg: restoredD0A,
          externalD0DegB: restoredD0B,
          boundsMs: bounds,
        });
        setWtcResult(res);
      } catch (err) {
        if ((err as Error).name === "AbortError" || ctrl.signal.aborted) {
          return;
        }
        setWtcResult(null);
        setWtcError(err instanceof ApiError ? err.detail : (err as Error).message);
      } finally {
        if (wtcAbortRef.current === ctrl) {
          setLoadingWtcResult(false);
        }
      }
    },
    [state.wtcStride, catalog, stationMetaMap],
  );

  // Auto-load active WTC result once its job completes/caches
  const activeJobObj = useMemo(
    () => state.jobs.find((j) => j.job_id === state.activeWtcJobId) ?? null,
    [state.jobs, state.activeWtcJobId],
  );

  useEffect(() => {
    if (state.activeWtcJobId === null && wtcResult !== null) {
      setWtcResult(null);
    }
  }, [state.activeWtcJobId, wtcResult]);

  useEffect(() => {
    if (!activeJobObj) return;
    if (
      (activeJobObj.status === "completed" || activeJobObj.status === "cached") &&
      wtcResult?.job_id !== activeJobObj.job_id
    ) {
      void openWtcJobResult(activeJobObj.job_id, state.wtcStride);
    }
  }, [activeJobObj, wtcResult?.job_id, openWtcJobResult, state.wtcStride]);

  const launchWtcJob = useCallback(
    async (overrides: Partial<WTCJobRequest> = {}) => {
      setWtcError(null);
      setWtcResult(null);
      const req: WTCJobRequest = {
        episode_id: state.episodeId,
        station_i: state.stationA,
        station_j: state.stationB,
        component: state.crossComponent,
        coord_frame: state.coordFrame,
        external_d0_deg_i: state.externalD0Deg,
        external_d0_deg_j: state.externalD0DegB,
        period_min_sec: 120,
        period_max_sec: 3600,
        dj: 1 / 12,
        omega0: 6,
        gap_policy: "subsegments",
        min_subsegment_samples: 120,
        n_surrogates: 300,
        alpha: state.alpha,
        seed: 42,
        ...overrides,
      };
      try {
        const job = await submitWtcJob(req);
        dispatch({ type: "UPSERT_JOB", job });
        dispatch({ type: "SET_ACTIVE_WTC_JOB", jobId: job.job_id });
        dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "wtc" });
        return job;
      } catch (err) {
        setWtcError(err instanceof ApiError ? err.detail : (err as Error).message);
        return null;
      }
    },
    [
      state.episodeId,
      state.stationA,
      state.stationB,
      state.crossComponent,
      state.coordFrame,
      state.externalD0Deg,
      state.externalD0DegB,
      state.alpha,
    ],
  );

  const cancelWtcJob = useCallback(async (jobId: string) => {
    try {
      const updated = await cancelJob(jobId);
      dispatch({ type: "UPSERT_JOB", job: updated });
    } catch (err) {
      setWtcError(err instanceof ApiError ? err.detail : (err as Error).message);
    }
  }, []);

  const abortAutoFlr = useCallback(() => {
    if (flrAutoInFlightRef.current && flrAbortRef.current) {
      flrAbortRef.current.abort();
      flrAbortRef.current = null;
      flrAutoInFlightRef.current = false;
      setLoadingFlr(false);
    }
  }, []);

  const flrFetchContextRef = useRef({
    state,
    episodeBoundsMs,
    welchPairRaw,
    welchPairSig,
    currentWelchSig,
  });
  flrFetchContextRef.current = {
    state,
    episodeBoundsMs,
    welchPairRaw,
    welchPairSig,
    currentWelchSig,
  };

  const executeFlrFetch = useCallback(
    async (windowIndexOverride?: number, autoSig?: string) => {
      const {
        state: curState,
        episodeBoundsMs: curBounds,
        welchPairRaw: curWelchRaw,
        welchPairSig: curWelchRawSig,
        currentWelchSig: curWelchSig,
      } = flrFetchContextRef.current;

      flrAbortRef.current?.abort();
      const ctrl = new AbortController();
      flrAbortRef.current = ctrl;
      flrAutoInFlightRef.current = autoSig !== undefined;

      setLoadingFlr(true);
      setFlrError(null);
      let winIdx = windowIndexOverride ?? 0;
      if (windowIndexOverride === undefined) {
        const targetMs = cursorRef.current ?? curState.cursorUtcMs;
        if (
          curWelchRaw &&
          curWelchRaw.window_timestamps_utc.length > 0 &&
          curWelchRawSig === curWelchSig
        ) {
          const refMs = targetMs ?? Date.parse(curWelchRaw.window_timestamps_utc[0]);
          let bestIdx = 0;
          let bestDiff = Number.POSITIVE_INFINITY;
          curWelchRaw.window_timestamps_utc.forEach((iso, idx) => {
            const diff = Math.abs(Date.parse(iso) - refMs);
            if (diff < bestDiff) {
              bestDiff = diff;
              bestIdx = idx;
            }
          });
          winIdx = bestIdx;
        } else {
          winIdx = computeCursorWindowIndex(
            targetMs,
            curBounds,
            curState.windowSamples,
            curState.stepSamples,
          );
        }
      }

      const resolvedSig = autoSig ?? buildFlrRequestSignature(curState, winIdx);

      try {
        const res = await computeWelchFlr(
          {
            episode_id: curState.episodeId,
            station_chain: curState.meridionalChain,
            component: curState.crossComponent,
            coord_frame: curState.coordFrame,
            band: curState.band,
            t_max_pc6_sec: curState.tMaxPc6Sec,
            L: curState.L,
            M: curState.M,
            window_samples: curState.windowSamples,
            step_samples: curState.stepSamples,
            window_index: winIdx,
            alpha: 0.05,
            spectrum_type: curState.spectrumType,
          },
          ctrl.signal,
        );
        if (!ctrl.signal.aborted) {
          setFlrProfileRaw(res);
          lastFetchedFlrSigRef.current = resolvedSig;
          lastFailedFlrSigRef.current = null;
        }
      } catch (err) {
        if ((err as Error).name === "AbortError" || ctrl.signal.aborted) {
          return;
        }
        setFlrProfileRaw(null);
        setFlrError(err instanceof ApiError ? err.detail : (err as Error).message);
        lastFetchedFlrSigRef.current = null;
        lastFailedFlrSigRef.current = resolvedSig;
      } finally {
        if (flrAbortRef.current === ctrl) {
          flrAbortRef.current = null;
          flrAutoInFlightRef.current = false;
          setLoadingFlr(false);
        }
      }
    },
    [],
  );

  const loadFlrAtCursor = useCallback(
    async (windowIndexOverride?: number) => {
      await executeFlrFetch(windowIndexOverride);
    },
    [executeFlrFetch],
  );

  const triggerShutdown = useCallback(async () => {
    dispatch({ type: "SET_SERVER_STATUS", status: "shutting_down" });
    try {
      await shutdownServer();
      dispatch({ type: "SET_SERVER_STATUS", status: "stopped" });
    } catch (err) {
      dispatch({ type: "SET_SERVER_STATUS", status: "running" });
      setGlobalError(err instanceof ApiError ? err.detail : (err as Error).message);
    }
  }, []);

  const value = useMemo<AnalysisContextValue>(
    () => ({
      state,
      dispatch,
      auth,
      health,
      catalog,
      activeEpisode,
      episodeBoundsMs,
      stationMetaMap,
      stationAvailabilityMap,
      geomag,
      rawSpaceWeather,
      shiftedSolar1,
      polarizationRaw,
      maskedPolarization,
      welchPairRaw,
      welchPairSig,
      maskedWelch,
      flrProfile,
      wtcResult,
      loadingCatalog,
      loadingTimeseries,
      loadingPolarization,
      loadingWelch,
      loadingFlr,
      loadingWtcResult,
      globalError,
      polarizationError,
      welchError,
      flrError,
      wtcError,
      baselineWarning,
      authenticateWithToken,
      refreshJobsAndCache,
      launchWtcJob,
      cancelWtcJob,
      openWtcJobResult,
      loadFlrAtCursor,
      setMapFlrChainActive,
      triggerShutdown,
    }),
    [
      state,
      auth,
      health,
      catalog,
      activeEpisode,
      episodeBoundsMs,
      stationMetaMap,
      stationAvailabilityMap,
      geomag,
      rawSpaceWeather,
      shiftedSolar1,
      polarizationRaw,
      maskedPolarization,
      welchPairRaw,
      welchPairSig,
      maskedWelch,
      flrProfile,
      wtcResult,
      loadingCatalog,
      loadingTimeseries,
      loadingPolarization,
      loadingWelch,
      loadingFlr,
      loadingWtcResult,
      globalError,
      polarizationError,
      welchError,
      flrError,
      wtcError,
      baselineWarning,
      authenticateWithToken,
      refreshJobsAndCache,
      launchWtcJob,
      cancelWtcJob,
      openWtcJobResult,
      loadFlrAtCursor,
      triggerShutdown,
    ],
  );

  const isFlrAutoNeeded = state.activeDeckTab === "welch" || mapFlrChainActive;
  const isFlrReady = auth.checked && auth.authenticated && Boolean(catalog);

  return (
    <AnalysisContext.Provider value={value}>
      <CursorProvider resetValue={state.cursorUtcMs} cursorRef={cursorRef}>
        <FlrAutoSync
          isNeeded={isFlrAutoNeeded}
          ready={isFlrReady}
          episodeBoundsMs={episodeBoundsMs}
          state={state}
          lastFetchedSigRef={lastFetchedFlrSigRef}
          lastFailedSigRef={lastFailedFlrSigRef}
          executeFlrFetch={executeFlrFetch}
          abortAutoFlr={abortAutoFlr}
        />
        {children}
      </CursorProvider>
    </AnalysisContext.Provider>
  );
};

export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) {
    throw new Error("useAnalysis must be used inside AnalysisProvider");
  }
  return ctx;
}
