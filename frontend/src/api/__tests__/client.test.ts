import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, requestJson, subscribeJobProgress } from "../client";
import type { JobStatusResponse } from "../types";

describe("API client requestJson and WebSocket + polling progress", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("parses JSON error details cleanly into ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Baseline required for GCK" }), {
          status: 422,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(requestJson("/api/polarization")).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      detail: "Baseline required for GCK",
    } satisfies Partial<ApiError>);
  });

  it("streams WebSocket updates until terminal status without invoking polling", () => {
    const sockets: Array<{
      onmessage: ((ev: { data: string }) => void) | null;
      onerror: (() => void) | null;
      onclose: (() => void) | null;
      close: ReturnType<typeof vi.fn>;
    }> = [];

    class FakeWebSocket {
      onmessage: ((ev: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      constructor(_url: string) {
        sockets.push(this);
      }
    }

    const fetchMock = vi.fn();
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("fetch", fetchMock);

    const updates: JobStatusResponse[] = [];
    const onClose = vi.fn();
    subscribeJobProgress("job-ws-1", {
      onUpdate: (j) => updates.push(j),
      onClose,
    });

    expect(sockets).toHaveLength(1);
    const baseJob: JobStatusResponse = {
      job_id: "job-ws-1",
      job_type: "cross_wtc",
      cache_key: "0".repeat(64),
      status: "running",
      stage: "monte_carlo",
      progress_pct: 50,
      completed_steps: 150,
      total_steps: 300,
      eta_sec: 4.5,
      cache_hit: false,
      result_path: null,
      error_message: null,
      params: {},
      created_at: "2026-10-10T00:00:00Z",
      updated_at: "2026-10-10T00:00:05Z",
    };

    sockets[0].onmessage?.({ data: JSON.stringify(baseJob) });
    expect(updates).toHaveLength(1);
    expect(updates[0].progress_pct).toBe(50);

    sockets[0].onmessage?.({
      data: JSON.stringify({ ...baseJob, status: "completed", progress_pct: 100 }),
    });
    expect(updates).toHaveLength(2);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(sockets[0].close).toHaveBeenCalledTimes(1);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("falls back to GET /api/jobs/{id} polling if WebSocket closes before terminal state", async () => {
    const sockets: Array<{
      onerror: (() => void) | null;
      onclose: (() => void) | null;
      close: ReturnType<typeof vi.fn>;
    }> = [];

    class FailingWebSocket {
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      constructor(_url: string) {
        sockets.push(this);
      }
    }

    const doneJob: JobStatusResponse = {
      job_id: "job-poll-1",
      job_type: "cross_wtc",
      cache_key: "1".repeat(64),
      status: "completed",
      stage: "done",
      progress_pct: 100,
      completed_steps: 20,
      total_steps: 20,
      eta_sec: 0,
      cache_hit: false,
      result_path: "/tmp/wtc.npz",
      error_message: null,
      params: {},
      created_at: "2026-10-10T00:00:00Z",
      updated_at: "2026-10-10T00:00:05Z",
    };

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(doneJob), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("WebSocket", FailingWebSocket);
    vi.stubGlobal("fetch", fetchMock);

    const updates: JobStatusResponse[] = [];
    const onClose = vi.fn();
    subscribeJobProgress("job-poll-1", {
      onUpdate: (j) => updates.push(j),
      onClose,
    });

    sockets[0].onclose?.();
    await vi.advanceTimersByTimeAsync(10);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/jobs/job-poll-1",
      expect.objectContaining({ method: "GET" }),
    );
    expect(updates).toHaveLength(1);
    expect(updates[0].status).toBe("completed");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
