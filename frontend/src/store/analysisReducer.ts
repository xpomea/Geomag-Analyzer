import type {
  BandType,
  CacheEntryOut,
  CoordFrame,
  CrossComponent,
  EpisodeInfo,
  JobStatusResponse,
  OversampleMethod,
  PolarizationMethod,
  Solar1ShiftMode,
  SpectrumType,
  StationInfo,
} from "../api/types";

export const DEFAULT_MERIDIONAL_CHAIN = [
  "HRN",
  "ABK",
  "SOD",
  "LYC",
  "NUR",
  "HLP",
  "BEL",
  "LVV",
  "PAG",
  "PEG",
];

export const MIN_ZOOM_SPAN_MS = 30 * 60_000;

export type CrossPanelTab = "spectrogram" | "welch" | "wtc";

export interface AnalysisState {
  episodeId: string;
  stationA: string;
  stationB: string;
  meridionalChain: string[];

  band: BandType;
  tMaxPc6Sec: number;
  method: PolarizationMethod;
  coordFrame: CoordFrame;
  externalD0Deg: number | null;

  L: number;
  M: number;
  windowSamples: number;
  stepSamples: number;
  nw: number;
  spectrumType: SpectrumType;
  oversampleM: number;
  oversampleMethod: OversampleMethod;
  orderN: number;

  // Strictly client-side instant state (never triggers backend refetch)
  alpha: number;
  solar1ShiftMode: Solar1ShiftMode;
  manualLagMin: number;
  refDistanceKm: number;
  smoothWindowMin: number;
  viewRangeMs: [number, number] | null;
  cursorUtcMs: number | null;
  showFilteredMagnetogram: boolean;
  showNoiseGhost: boolean;

  crossComponent: CrossComponent;
  activeCrossTab: CrossPanelTab;

  jobs: JobStatusResponse[];
  activeWtcJobId: string | null;
  jobTransport: Record<string, "ws" | "sse">;
  cacheEntries: CacheEntryOut[];
  jobDrawerOpen: boolean;
  serverStatus: "running" | "shutting_down" | "stopped";
}

export const INITIAL_ANALYSIS_STATE: AnalysisState = {
  episodeId: "storm_20261009_11",
  stationA: "LVV",
  stationB: "BEL",
  meridionalChain: DEFAULT_MERIDIONAL_CHAIN,

  band: "Pc5",
  tMaxPc6Sec: 1800,
  method: "welch",
  coordFrame: "XYZ",
  externalD0Deg: null,

  L: 128,
  M: 64,
  windowSamples: 360,
  stepSamples: 30,
  nw: 2.5,
  spectrumType: "flat",
  oversampleM: 4,
  oversampleMethod: "fourier",
  orderN: 3,

  alpha: 0.05,
  solar1ShiftMode: "ballistic_auto",
  manualLagMin: 45,
  refDistanceKm: 1_400_000,
  smoothWindowMin: 31,
  viewRangeMs: null,
  cursorUtcMs: null,
  showFilteredMagnetogram: false,
  showNoiseGhost: false,

  crossComponent: "X",
  activeCrossTab: "spectrogram",

  jobs: [],
  activeWtcJobId: null,
  jobTransport: {},
  cacheEntries: [],
  jobDrawerOpen: false,
  serverStatus: "running",
};

export type AnalysisAction =
  | { type: "SET_EPISODE"; episodeId: string; episode?: EpisodeInfo | null }
  | { type: "SET_STATION_A"; stationIaga: string }
  | { type: "SET_STATION_B"; stationIaga: string }
  | { type: "SET_PAIR"; stationA: string; stationB: string }
  | { type: "SET_MERIDIONAL_CHAIN"; chain: string[] }
  | { type: "SET_BAND"; band: BandType }
  | { type: "SET_T_MAX_PC6"; sec: number }
  | { type: "SET_METHOD"; method: PolarizationMethod }
  | { type: "SET_COORD_FRAME"; coordFrame: CoordFrame }
  | { type: "SET_EXTERNAL_D0"; d0Deg: number | null }
  | { type: "SET_WELCH_PARAMS"; L?: number; M?: number; windowSamples?: number; stepSamples?: number }
  | { type: "SET_NW"; nw: number }
  | { type: "SET_SPECTRUM_TYPE"; spectrumType: SpectrumType }
  | { type: "SET_VARIANT_B_PARAMS"; oversampleM?: number; oversampleMethod?: OversampleMethod; orderN?: number }
  | { type: "SET_ALPHA"; alpha: number }
  | { type: "SET_SOLAR1_SHIFT"; mode: Solar1ShiftMode }
  | { type: "SET_MANUAL_LAG"; lagMin: number }
  | { type: "SET_REF_DISTANCE"; km: number }
  | { type: "SET_SMOOTH_WINDOW"; min: number }
  | { type: "SET_VIEW_RANGE"; rangeMs: [number, number] | null; boundsMs?: [number, number] | null }
  | { type: "SET_CURSOR_UTC"; utcMs: number | null }
  | { type: "SET_SHOW_FILTERED"; show: boolean }
  | { type: "SET_SHOW_NOISE_GHOST"; show: boolean }
  | { type: "SET_CROSS_COMPONENT"; component: CrossComponent }
  | { type: "SET_ACTIVE_CROSS_TAB"; tab: CrossPanelTab }
  | { type: "SET_JOBS"; jobs: JobStatusResponse[] }
  | { type: "UPSERT_JOB"; job: JobStatusResponse }
  | { type: "SET_ACTIVE_WTC_JOB"; jobId: string | null }
  | { type: "SET_JOB_TRANSPORT"; jobId: string; transport: "ws" | "sse" }
  | { type: "SET_CACHE_ENTRIES"; entries: CacheEntryOut[] }
  | { type: "SET_JOB_DRAWER_OPEN"; open: boolean }
  | { type: "SET_SERVER_STATUS"; status: "running" | "shutting_down" | "stopped" };

export function clampViewRange(
  rangeMs: [number, number],
  boundsMs?: [number, number] | null,
): [number, number] {
  let start = Math.min(rangeMs[0], rangeMs[1]);
  let end = Math.max(rangeMs[0], rangeMs[1]);

  if (boundsMs) {
    const bStart = Math.min(boundsMs[0], boundsMs[1]);
    const bEnd = Math.max(boundsMs[0], boundsMs[1]);
    const totalSpan = Math.max(1, bEnd - bStart);
    const minSpan = Math.min(MIN_ZOOM_SPAN_MS, totalSpan);

    start = Math.max(bStart, start);
    end = Math.min(bEnd, end);
    if (end - start < minSpan) {
      const center = 0.5 * (start + end);
      start = Math.max(bStart, center - 0.5 * minSpan);
      end = Math.min(bEnd, start + minSpan);
      if (end - start < minSpan) {
        start = Math.max(bStart, end - minSpan);
      }
    }
    return [Math.round(start), Math.round(end)];
  }

  if (end - start < MIN_ZOOM_SPAN_MS) {
    const center = 0.5 * (start + end);
    start = center - 0.5 * MIN_ZOOM_SPAN_MS;
    end = center + 0.5 * MIN_ZOOM_SPAN_MS;
  }
  return [Math.round(start), Math.round(end)];
}

function harmonizeCrossComponent(
  prev: CrossComponent,
  coordFrame: CoordFrame,
): CrossComponent {
  if (prev === "Z") return "Z";
  if (coordFrame === "XmYm") {
    return prev === "Y" || prev === "Ym" ? "Ym" : "Xm";
  }
  return prev === "Y" || prev === "Ym" ? "Y" : "X";
}

export function analysisReducer(
  state: AnalysisState,
  action: AnalysisAction,
): AnalysisState {
  switch (action.type) {
    case "SET_EPISODE": {
      const ep = action.episode;
      let nextRange: [number, number] | null = null;
      let nextCursor: number | null = null;
      if (ep) {
        const t0 = Date.parse(ep.start_utc);
        const t1 = Date.parse(ep.end_utc);
        if (Number.isFinite(t0) && Number.isFinite(t1)) {
          nextRange = [t0, t1];
          nextCursor = Math.round(0.5 * (t0 + t1));
        }
      }
      return {
        ...state,
        episodeId: action.episodeId,
        viewRangeMs: nextRange,
        cursorUtcMs: nextCursor,
        activeWtcJobId: null,
      };
    }

    case "SET_STATION_A":
      return { ...state, stationA: action.stationIaga.toUpperCase() };

    case "SET_STATION_B":
      return { ...state, stationB: action.stationIaga.toUpperCase() };

    case "SET_PAIR":
      return {
        ...state,
        stationA: action.stationA.toUpperCase(),
        stationB: action.stationB.toUpperCase(),
      };

    case "SET_MERIDIONAL_CHAIN":
      return { ...state, meridionalChain: action.chain };

    case "SET_BAND":
      return { ...state, band: action.band };

    case "SET_T_MAX_PC6":
      return { ...state, tMaxPc6Sec: Math.max(660, action.sec) };

    case "SET_METHOD": {
      const isVarB = action.method.startsWith("variant_b");
      return {
        ...state,
        method: action.method,
        showFilteredMagnetogram: isVarB ? state.showFilteredMagnetogram : false,
      };
    }

    case "SET_COORD_FRAME":
      return {
        ...state,
        coordFrame: action.coordFrame,
        crossComponent: harmonizeCrossComponent(state.crossComponent, action.coordFrame),
      };

    case "SET_EXTERNAL_D0":
      return { ...state, externalD0Deg: action.d0Deg };

    case "SET_WELCH_PARAMS":
      return {
        ...state,
        L: action.L ?? state.L,
        M: action.M ?? state.M,
        windowSamples: action.windowSamples ?? state.windowSamples,
        stepSamples: action.stepSamples ?? state.stepSamples,
      };

    case "SET_NW":
      return { ...state, nw: Math.max(1.0, action.nw) };

    case "SET_SPECTRUM_TYPE":
      return { ...state, spectrumType: action.spectrumType };

    case "SET_VARIANT_B_PARAMS":
      return {
        ...state,
        oversampleM: action.oversampleM ?? state.oversampleM,
        oversampleMethod: action.oversampleMethod ?? state.oversampleMethod,
        orderN: action.orderN ?? state.orderN,
      };

    case "SET_ALPHA": {
      if (!(action.alpha > 0 && action.alpha < 1)) return state;
      return { ...state, alpha: action.alpha };
    }

    case "SET_SOLAR1_SHIFT":
      return { ...state, solar1ShiftMode: action.mode };

    case "SET_MANUAL_LAG":
      return { ...state, manualLagMin: action.lagMin };

    case "SET_REF_DISTANCE":
      return { ...state, refDistanceKm: Math.max(100_000, action.km) };

    case "SET_SMOOTH_WINDOW":
      return { ...state, smoothWindowMin: Math.max(1, Math.round(action.min)) };

    case "SET_VIEW_RANGE":
      if (action.rangeMs === null) {
        return { ...state, viewRangeMs: null };
      }
      return {
        ...state,
        viewRangeMs: clampViewRange(action.rangeMs, action.boundsMs),
      };

    case "SET_CURSOR_UTC":
      return { ...state, cursorUtcMs: action.utcMs };

    case "SET_SHOW_FILTERED":
      return { ...state, showFilteredMagnetogram: action.show };

    case "SET_SHOW_NOISE_GHOST":
      return { ...state, showNoiseGhost: action.show };

    case "SET_CROSS_COMPONENT":
      return { ...state, crossComponent: action.component };

    case "SET_ACTIVE_CROSS_TAB":
      return { ...state, activeCrossTab: action.tab };

    case "SET_JOBS":
      return { ...state, jobs: action.jobs };

    case "UPSERT_JOB": {
      const idx = state.jobs.findIndex((j) => j.job_id === action.job.job_id);
      if (idx === -1) {
        return { ...state, jobs: [action.job, ...state.jobs] };
      }
      const next = state.jobs.slice();
      next[idx] = action.job;
      return { ...state, jobs: next };
    }

    case "SET_ACTIVE_WTC_JOB":
      return { ...state, activeWtcJobId: action.jobId };

    case "SET_JOB_TRANSPORT":
      return {
        ...state,
        jobTransport: {
          ...state.jobTransport,
          [action.jobId]: action.transport,
        },
      };

    case "SET_CACHE_ENTRIES":
      return { ...state, cacheEntries: action.entries };

    case "SET_JOB_DRAWER_OPEN":
      return { ...state, jobDrawerOpen: action.open };

    case "SET_SERVER_STATUS":
      return { ...state, serverStatus: action.status };

    default:
      return state;
  }
}

/**
 * Pure selector of server-fetched analysis parameters.
 * Excludes strictly client-side parameters (alpha, solar1ShiftMode, viewRangeMs, cursorUtcMs)
 * so toggling them never causes a backend roundtrip.
 */
export function selectBackendFetchSignature(state: AnalysisState): string {
  return JSON.stringify({
    episodeId: state.episodeId,
    stationA: state.stationA,
    band: state.band,
    tMaxPc6Sec: state.tMaxPc6Sec,
    method: state.method,
    coordFrame: state.coordFrame,
    externalD0Deg: state.externalD0Deg,
    L: state.L,
    M: state.M,
    windowSamples: state.windowSamples,
    stepSamples: state.stepSamples,
    nw: state.nw,
    spectrumType: state.spectrumType,
    oversampleM: state.oversampleM,
    oversampleMethod: state.oversampleMethod,
    orderN: state.orderN,
  });
}

export function getStationBaselineWarning(
  station: StationInfo | undefined,
  coordFrame: CoordFrame,
  externalD0Deg: number | null,
): string | null {
  if (!station) return null;
  if (!station.has_baseline && coordFrame === "XmYm" && externalD0Deg === null) {
    return (
      `Станция ${station.iaga_code} не имеет абсолютной базовой линии (variation-only). ` +
      `Поворот в локальную СК XmYm заблокирован без явного ввода D0.`
    );
  }
  return null;
}
