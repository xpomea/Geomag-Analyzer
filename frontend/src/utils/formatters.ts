export function parseTimestampsMs(isoList: string[]): number[] {
  return isoList.map((iso) => Date.parse(iso));
}

export function formatUtcShort(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  const d = new Date(ms);
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}

export function formatUtcFull(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  const d = new Date(ms);
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${mi} UTC`;
}

export function formatEta(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec) || sec < 0) return "—";
  if (sec < 60) return `${Math.round(sec)} s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m} min ${String(s).padStart(2, "0")} s`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatNum(val: number | null | undefined, digits = 3): string {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  return val.toFixed(digits);
}

export function formatSolar1TimeFrame(
  shiftMode: string,
  meanDelayMin?: number | null,
): string {
  if (shiftMode === "ballistic_auto") {
    return meanDelayMin != null && Number.isFinite(meanDelayMin)
      ? `BSN (Ballistic Vx, ~${meanDelayMin.toFixed(0)} min)`
      : "BSN (Ballistic Vx)";
  }
  if (shiftMode === "manual") {
    return meanDelayMin != null && Number.isFinite(meanDelayMin)
      ? `BSN (Manual +${meanDelayMin.toFixed(0)} min)`
      : "BSN (Manual)";
  }
  return "L1 Raw";
}

const JOB_STATUS_LABELS: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  cancelling: "Cancelling",
  completed: "Completed",
  cached: "Cached",
  cancelled: "Cancelled",
  failed: "Failed",
};

export function formatJobStatus(status: string): string {
  return JOB_STATUS_LABELS[status] ?? status;
}
