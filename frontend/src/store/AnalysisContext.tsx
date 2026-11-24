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
  applyPolarizationAlphaMask,
  applyWelchAlphaMask,
  type MaskedPolarizationView,
  type MaskedWelchView,
} from "../utils/masking";
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

interface AnalysisContextValue {
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
  triggerShutdown: () => Promise<void>;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(analysisReducer, INITIAL_ANALYSIS_STATE);

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
  const [flrProfile, setFlrProfile] = useState<MeridionalFLRResponse | null>(null);
  const [wtcResult, setWtcResult] = useState<WTCResultResponse | null>(null);
  const [wtcStride, setWtcStride] = useState<number>(2);

  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [loadingTimeseries, setLoadingTimeseries] = useState(false);
  const [loadingPolarization, setLoadingPolarization] = useState(false);
  const [loadingWelch, setLoadingWelch] = useState(false);
  const [loadingFlr, setLoadingFlr] = useState(false);
  const [loadingWtcResult, setLoadingWtcResult] = useState(false);

  const [globalError, setGlobalError] = useState<string | null>(null);
  const [polarizationError, setPolarizationError] = useState<string | null>(null);
  const [welchError, setWelchError] = useState<string | null>(null);
  const [flrError, setFlrError] = useState<string | null>(null);
  const [wtcError, setWtcError] = useState<string | null>(null);

  const wsSubsRef = useRef<Map<string, () => void>>(new Map());

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
          error: err instanceof Error ? err.message : "Ошибка инициализации сессии",
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
        error: err instanceof ApiError ? err.detail : "Неверный или просроченный токен",
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
          setGlobalError(err instanceof Error ? err.message : "Ошибка загрузки каталога");
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
          setGlobalError(err instanceof Error ? err.message : "Ошибка загрузки космической погоды");
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

  // 7. Fetch Welch pair when cross Welch tab is active (alpha NOT in deps!)
  useEffect(() => {
    if (!auth.checked || !auth.authenticated || !catalog) return;
    if (state.activeCrossTab !== "welch") return;

    const warnA = getStationBaselineWarning(
      stationMetaMap.get(state.stationA),
      state.coordFrame,
      state.externalD0Deg,
    );
    const warnB = getStationBaselineWarning(
      stationMetaMap.get(state.stationB),
      state.coordFrame,
      null,
    );
    if (warnA || warnB) {
      setWelchPairRaw(null);
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
        void listCacheEntries(undefined, 100).then((entries) => {
          dispatch({ type: "SET_CACHE_ENTRIES", entries });
        });
      })
      .catch((err) => {
        if ((err as Error).name !== "AbortError") {
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
    state.episodeId,
    state.stationA,
    state.stationB,
    state.crossComponent,
    state.coordFrame,
    state.externalD0Deg,
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
    };
  }, []);

  const openWtcJobResult = useCallback(
    async (jobId: string, stride = wtcStride) => {
      setWtcStride(stride);
      setLoadingWtcResult(true);
      setWtcError(null);
      try {
        const res = await getWtcJobResult(jobId, stride);
        setWtcResult(res);
        dispatch({ type: "SET_ACTIVE_WTC_JOB", jobId });
        dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "wtc" });
      } catch (err) {
        setWtcError(err instanceof ApiError ? err.detail : (err as Error).message);
      } finally {
        setLoadingWtcResult(false);
      }
    },
    [wtcStride],
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
      void openWtcJobResult(activeJobObj.job_id, wtcStride);
    }
  }, [activeJobObj, wtcResult?.job_id, openWtcJobResult, wtcStride]);

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

  const loadFlrAtCursor = useCallback(
    async (windowIndexOverride?: number) => {
      setLoadingFlr(true);
      setFlrError(null);
      let winIdx = windowIndexOverride ?? 0;
      if (windowIndexOverride === undefined && welchPairRaw && welchPairRaw.window_timestamps_utc.length > 0) {
        const targetMs = state.cursorUtcMs ?? Date.parse(welchPairRaw.window_timestamps_utc[0]);
        let bestIdx = 0;
        let bestDiff = Number.POSITIVE_INFINITY;
        welchPairRaw.window_timestamps_utc.forEach((iso, idx) => {
          const diff = Math.abs(Date.parse(iso) - targetMs);
          if (diff < bestDiff) {
            bestDiff = diff;
            bestIdx = idx;
          }
        });
        winIdx = bestIdx;
      }

      try {
        const res = await computeWelchFlr({
          episode_id: state.episodeId,
          station_chain: state.meridionalChain,
          component: state.crossComponent,
          coord_frame: state.coordFrame,
          band: state.band,
          t_max_pc6_sec: state.tMaxPc6Sec,
          L: state.L,
          M: state.M,
          window_samples: state.windowSamples,
          step_samples: state.stepSamples,
          window_index: winIdx,
          alpha: state.alpha,
        });
        setFlrProfile(res);
      } catch (err) {
        setFlrError(err instanceof ApiError ? err.detail : (err as Error).message);
      } finally {
        setLoadingFlr(false);
      }
    },
    [
      welchPairRaw,
      state.cursorUtcMs,
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
      state.alpha,
    ],
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

  const value: AnalysisContextValue = {
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
  };

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
};

export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) {
    throw new Error("useAnalysis must be used inside AnalysisProvider");
  }
  return ctx;
}
