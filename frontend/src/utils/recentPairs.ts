export interface RecentPair {
  a: string;
  b: string;
}

export const RECENT_PAIRS_STORAGE_KEY = "geomag.recentPairs.v1";
export const MAX_RECENT_PAIRS = 6;

export function isSameUnorderedPair(p1: RecentPair, p2: RecentPair): boolean {
  return (p1.a === p2.a && p1.b === p2.b) || (p1.a === p2.b && p1.b === p2.a);
}

/**
 * Insert or promote a station pair to the front (MRU) of the recent pairs list.
 * Ignores empty or degenerate (A === B) pairs and deduplicates A-B / B-A.
 */
export function recordRecentPair(
  prev: RecentPair[],
  stationA: string,
  stationB: string,
  maxItems = MAX_RECENT_PAIRS,
): RecentPair[] {
  const a = stationA.trim().toUpperCase();
  const b = stationB.trim().toUpperCase();
  if (!a || !b || a === b) return prev;
  const nextEntry: RecentPair = { a, b };
  const filtered = prev.filter((item) => !isSameUnorderedPair(item, nextEntry));
  return [nextEntry, ...filtered].slice(0, maxItems);
}

/**
 * Safely load persisted recent pairs from localStorage (returns [] in SSR or restricted environments).
 */
export function loadRecentPairs(): RecentPair[] {
  try {
    const storage = typeof window !== "undefined" ? window.localStorage : undefined;
    if (!storage) return [];
    const raw = storage.getItem(RECENT_PAIRS_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const out: RecentPair[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const rec = item as { a?: unknown; b?: unknown };
      if (typeof rec.a !== "string" || typeof rec.b !== "string") continue;
      const a = rec.a.trim().toUpperCase();
      const b = rec.b.trim().toUpperCase();
      if (!a || !b || a === b) continue;
      const candidate: RecentPair = { a, b };
      if (!out.some((x) => isSameUnorderedPair(x, candidate))) {
        out.push(candidate);
      }
      if (out.length >= MAX_RECENT_PAIRS) break;
    }
    return out;
  } catch {
    return [];
  }
}

/**
 * Safely persist recent pairs to localStorage.
 */
export function saveRecentPairs(pairs: RecentPair[]): void {
  try {
    const storage = typeof window !== "undefined" ? window.localStorage : undefined;
    if (!storage) return;
    storage.setItem(
      RECENT_PAIRS_STORAGE_KEY,
      JSON.stringify(pairs.slice(0, MAX_RECENT_PAIRS)),
    );
  } catch {
    // ignore storage quota, SSR, or disabled localStorage
  }
}
