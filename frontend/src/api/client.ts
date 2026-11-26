import type {
  CacheEntryOut,
  CatalogResponse,
  GeomagTimeseriesResponse,
  HealthResponse,
  JobStatusResponse,
  MeridionalFLRRequest,
  MeridionalFLRResponse,
  NetworkSnapshotRequest,
  NetworkSnapshotResponse,
  PolarizationRequest,
  PolarizationResponse,
  ShutdownResponse,
  SpaceWeatherResponse,
  WTCJobRequest,
  WTCResultResponse,
  WelchMatrixRequest,
  WelchMatrixResponse,
  WelchPairRequest,
  WelchPairResponse,
} from "./types";

export const TERMINAL_JOB_STATUSES = new Set([
  "completed",
  "cached",
  "cancelled",
  "failed",
]);

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function buildQuery(params: Record<string, string | number | boolean | null | undefined>): string {
  const sp = new URLSearchParams();
  for (const [key, val] of Object.entries(params)) {
    if (val !== undefined && val !== null && val !== "") {
      sp.set(key, String(val));
    }
  }
  const qs = sp.toString();
  return qs ? `?${qs}` : "";
}

export async function requestJson<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  const headers = new Headers(options.headers);
  if (
    options.body !== undefined &&
    typeof options.body === "string" &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  const resp = await fetch(url, {
    ...options,
    method,
    headers,
    credentials: "same-origin",
  });

  if (!resp.ok) {
    let detail = resp.statusText || "Request failed";
    try {
      const payload = (await resp.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail !== undefined) {
        detail = JSON.stringify(payload.detail);
      }
    } catch {
      // keep statusText
    }
    throw new ApiError(resp.status, detail);
  }

  return (await resp.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health", { signal });
}

export function getCatalog(signal?: AbortSignal): Promise<CatalogResponse> {
  return requestJson<CatalogResponse>("/api/catalog", { signal });
}

export function getGeomagTimeseries(
  params: {
    episode_id: string;
    station_iaga: string;
    coord_frame?: string;
    external_d0_deg?: number | null;
    start_utc?: string | null;
    end_utc?: string | null;
  },
  signal?: AbortSignal,
): Promise<GeomagTimeseriesResponse> {
  return requestJson<GeomagTimeseriesResponse>(
    `/api/timeseries/geomag${buildQuery(params)}`,
    { signal },
  );
}

export function getSpaceWeatherTimeseries(
  params: {
    episode_id: string;
    solar1_shift?: string;
    manual_lag_min?: number | null;
    ref_distance_km?: number;
    smooth_window_min?: number;
    include_omni?: boolean;
    include_oulu?: boolean;
    start_utc?: string | null;
    end_utc?: string | null;
  },
  signal?: AbortSignal,
): Promise<SpaceWeatherResponse> {
  return requestJson<SpaceWeatherResponse>(
    `/api/timeseries/space-weather${buildQuery(params)}`,
    { signal },
  );
}

export function computePolarization(
  req: PolarizationRequest,
  signal?: AbortSignal,
): Promise<PolarizationResponse> {
  return requestJson<PolarizationResponse>("/api/polarization", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function computeNetworkSnapshot(
  req: NetworkSnapshotRequest,
  signal?: AbortSignal,
): Promise<NetworkSnapshotResponse> {
  return requestJson<NetworkSnapshotResponse>("/api/polarization/network-snapshot", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function computeWelchPair(
  req: WelchPairRequest,
  signal?: AbortSignal,
): Promise<WelchPairResponse> {
  return requestJson<WelchPairResponse>("/api/cross/welch/pair", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function computeWelchFlr(
  req: MeridionalFLRRequest,
  signal?: AbortSignal,
): Promise<MeridionalFLRResponse> {
  return requestJson<MeridionalFLRResponse>("/api/cross/welch/flr", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function computeWelchMatrix(
  req: WelchMatrixRequest,
  signal?: AbortSignal,
): Promise<WelchMatrixResponse> {
  return requestJson<WelchMatrixResponse>("/api/cross/welch/matrix", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function submitWtcJob(
  req: WTCJobRequest,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>("/api/jobs/wtc", {
    method: "POST",
    body: JSON.stringify(req),
    signal,
  });
}

export function listJobs(limit = 50, signal?: AbortSignal): Promise<JobStatusResponse[]> {
  return requestJson<JobStatusResponse[]>(`/api/jobs${buildQuery({ limit })}`, { signal });
}

export function getJobStatus(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export function cancelJob(jobId: string, signal?: AbortSignal): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
    signal,
  });
}

export function getWtcJobResult(
  jobId: string,
  stride = 2,
  signal?: AbortSignal,
): Promise<WTCResultResponse> {
  return requestJson<WTCResultResponse>(
    `/api/jobs/${encodeURIComponent(jobId)}/result${buildQuery({ stride })}`,
    { signal },
  );
}

export function listCacheEntries(
  category?: string,
  limit = 100,
  signal?: AbortSignal,
): Promise<CacheEntryOut[]> {
  return requestJson<CacheEntryOut[]>(
    `/api/cache${buildQuery({ category, limit })}`,
    { signal },
  );
}

export function shutdownServer(): Promise<ShutdownResponse> {
  return requestJson<ShutdownResponse>("/api/shutdown", { method: "POST" });
}

export interface JobStreamHandlers {
  onUpdate: (job: JobStatusResponse) => void;
  onError?: (err: string) => void;
  onClose?: () => void;
}

export function subscribeJobProgress(
  jobId: string,
  handlers: JobStreamHandlers,
): () => void {
  let closed = false;
  let sawTerminal = false;
  let ws: WebSocket | null = null;
  let pollTimer: number | null = null;
  let pollAbort: AbortController | null = null;

  const cleanup = () => {
    closed = true;
    if (ws) {
      try {
        ws.close();
      } catch {
        // ignore
      }
      ws = null;
    }
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    if (pollAbort) {
      pollAbort.abort();
      pollAbort = null;
    }
  };

  const applyJob = (job: JobStatusResponse) => {
    if (TERMINAL_JOB_STATUSES.has(job.status)) {
      sawTerminal = true;
    }
    handlers.onUpdate(job);
    if (sawTerminal) {
      cleanup();
      handlers.onClose?.();
    }
  };

  const handleMessage = (rawData: string) => {
    try {
      applyJob(JSON.parse(rawData) as JobStatusResponse);
    } catch {
      handlers.onError?.("Malformed progress payload");
    }
  };

  const startPollFallback = () => {
    if (closed || sawTerminal || pollTimer !== null) {
      return;
    }
    let consecutiveFailures = 0;
    const pollOnce = async () => {
      if (closed || sawTerminal) return;
      const ctrl = new AbortController();
      pollAbort = ctrl;
      try {
        const job = await getJobStatus(jobId, ctrl.signal);
        consecutiveFailures = 0;
        if (!closed) {
          applyJob(job);
        }
      } catch (err) {
        if (closed || (err as Error).name === "AbortError") return;
        consecutiveFailures += 1;
        if (consecutiveFailures >= 5) {
          cleanup();
          handlers.onError?.("Job status polling failed repeatedly");
          handlers.onClose?.();
        }
      } finally {
        if (pollAbort === ctrl) {
          pollAbort = null;
        }
      }
    };
    void pollOnce();
    pollTimer = window.setInterval(() => {
      void pollOnce();
    }, 1000);
  };

  try {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/ws/jobs/${encodeURIComponent(jobId)}`;
    ws = new WebSocket(wsUrl);
    ws.onmessage = (ev) => {
      if (!closed && typeof ev.data === "string") {
        handleMessage(ev.data);
      }
    };
    ws.onerror = () => {
      if (!closed && !sawTerminal) {
        try {
          ws?.close();
        } catch {
          // ignore
        }
        ws = null;
        startPollFallback();
      }
    };
    ws.onclose = () => {
      ws = null;
      if (!closed && !sawTerminal) {
        startPollFallback();
      }
    };
  } catch {
    startPollFallback();
  }

  return cleanup;
}
