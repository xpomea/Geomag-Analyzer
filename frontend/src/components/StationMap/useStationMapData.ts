import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  computeNetworkSnapshot,
  computeWelchMatrix,
  computeWelchPair,
} from "../../api/client";
import type {
  NetworkSnapshotResponse,
  WelchMatrixResponse,
  WelchPairResponse,
} from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { useCursor } from "../../store/CursorContext";
import {
  applyWelchAlphaMask,
  buildWelchPairRequestSignature,
  computeCursorWindowIndex,
} from "../../utils/masking";
import {
  applyNetworkSnapshotAlphaMask,
  computeSnapshotWindowUtc,
} from "../../utils/stokesEllipse";
import { scaleGamma2CritForAlpha } from "../../utils/thresholds";

export interface MatrixArcEdge {
  stationI: string;
  stationJ: string;
  gamma2: number | null;
  gamma2Crit: number | null;
  phaseDeg: number | null;
  isSignificant: boolean;
}

export interface UseStationMapDataResult {
  effectiveCursorMs: number | null;
  snapshotWindow: { startIso: string; endIso: string; centerMs: number } | null;
  snapshotMasked: NetworkSnapshotResponse | null;
  loadingSnapshot: boolean;
  snapshotError: string | null;
  pairMetricsAtCursor: {
    gamma2: number | null;
    gamma2Crit: number | null;
    phaseDeg: number | null;
    phaseDegMasked: number | null;
    azimuthalM: number | null;
    mAlias: number | null;
    isSignificant: boolean;
    isMeridionalGuarded: boolean;
    deltaLonDeg: number | null;
  } | null;
  loadingPair: boolean;
  matrixArcsFromA: MatrixArcEdge[];
  loadingMatrix: boolean;
  matrixWindowIndex: number;
}

export function useStationMapData(options: {
  showArcs: boolean;
  showMatrixArcs: boolean;
}): UseStationMapDataResult {
  const {
    state,
    catalog,
    episodeBoundsMs,
    stationMetaMap,
    welchPairRaw,
    welchPairSig,
    maskedWelch,
    loadingWelch,
  } = useAnalysis();
  const { cursorUtcMs } = useCursor();

  const [snapshotState, setSnapshotState] = useState<{
    paramSig: string;
    data: NetworkSnapshotResponse;
  } | null>(null);
  const [loadingSnapshot, setLoadingSnapshot] = useState(false);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);

  const [localPairState, setLocalPairState] = useState<{
    sig: string;
    data: WelchPairResponse;
  } | null>(null);
  const [loadingPair, setLoadingPair] = useState(false);

  const [matrixState, setMatrixState] = useState<{
    paramSig: string;
    data: WelchMatrixResponse;
  } | null>(null);
  const [loadingMatrix, setLoadingMatrix] = useState(false);

  const effectiveCursorMs = useMemo(() => {
    if (cursorUtcMs !== null && Number.isFinite(cursorUtcMs)) return cursorUtcMs;
    if (state.cursorUtcMs !== null && Number.isFinite(state.cursorUtcMs)) return state.cursorUtcMs;
    if (episodeBoundsMs) return Math.round(0.5 * (episodeBoundsMs[0] + episodeBoundsMs[1]));
    return null;
  }, [cursorUtcMs, state.cursorUtcMs, episodeBoundsMs]);

  const snapshotWindow = useMemo(() => {
    if (effectiveCursorMs === null) return null;
    return computeSnapshotWindowUtc(effectiveCursorMs, state.windowSamples, episodeBoundsMs);
  }, [effectiveCursorMs, state.windowSamples, episodeBoundsMs]);

  const snapParamSig = useMemo(
    () =>
      JSON.stringify([
        state.episodeId,
        state.band,
        state.tMaxPc6Sec,
        state.coordFrame,
        state.L,
        state.M,
        state.windowSamples,
        state.spectrumType,
      ]),
    [
      state.episodeId,
      state.band,
      state.tMaxPc6Sec,
      state.coordFrame,
      state.L,
      state.M,
      state.windowSamples,
      state.spectrumType,
    ],
  );
  const lastSnapSigRef = useRef<string>("");

  // 1. Debounced 49-station network snapshot (NOT dependent on alpha!)
  useEffect(() => {
    if (!catalog || !snapshotWindow || catalog.stations.length === 0) {
      setSnapshotState((prev) =>
        !snapshotWindow || (prev && prev.paramSig !== snapParamSig) ? null : prev,
      );
      setLoadingSnapshot(false);
      return;
    }

    const sigChanged = lastSnapSigRef.current !== snapParamSig;
    const delayMs =
      sigChanged || snapshotState === null || snapshotState.paramSig !== snapParamSig ? 15 : 220;
    const ctrl = new AbortController();
    const fetchParamSig = snapParamSig;

    setSnapshotState((prev) => (prev && prev.paramSig === snapParamSig ? prev : null));

    const timer = setTimeout(() => {
      lastSnapSigRef.current = fetchParamSig;
      setLoadingSnapshot(true);
      setSnapshotError(null);
      computeNetworkSnapshot(
        {
          episode_id: state.episodeId,
          band: state.band,
          t_max_pc6_sec: state.tMaxPc6Sec,
          coord_frame: state.coordFrame,
          alpha: 0.05,
          spectrum_type: state.spectrumType,
          L: state.L,
          M: state.M,
          start_utc: snapshotWindow.startIso,
          end_utc: snapshotWindow.endIso,
        },
        ctrl.signal,
      )
        .then((res) => {
          if (!ctrl.signal.aborted) {
            setSnapshotState({ paramSig: fetchParamSig, data: res });
          }
        })
        .catch((err) => {
          if ((err as Error).name !== "AbortError" && !ctrl.signal.aborted) {
            setSnapshotState(null);
            setSnapshotError(err instanceof ApiError ? err.detail : (err as Error).message);
          }
        })
        .finally(() => {
          if (!ctrl.signal.aborted) {
            setLoadingSnapshot(false);
          }
        });
    }, delayMs);

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    catalog,
    snapParamSig,
    snapshotWindow?.startIso,
    snapshotWindow?.endIso,
  ]);

  // Instant client-side alpha re-masking for all 49 stations
  const snapshotMasked = useMemo(() => {
    if (!snapshotState || snapshotState.paramSig !== snapParamSig) return null;
    return applyNetworkSnapshotAlphaMask(snapshotState.data, state.alpha);
  }, [snapshotState, snapParamSig, state.alpha]);

  // 2. Pair coherence data for active pair A <-> B (reuse AnalysisContext welchPairRaw if matching)
  const currentPairSig = useMemo(
    () => buildWelchPairRequestSignature(state),
    [
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
    ],
  );

  const contextWelchOwnsFetch = state.activeDeckTab === "welch";

  const pairBaselessBlocked = useMemo(() => {
    if (state.coordFrame !== "XmYm") return false;
    const stA = stationMetaMap.get(state.stationA);
    const stB = stationMetaMap.get(state.stationB);
    return Boolean(
      (stA && !stA.has_baseline && state.externalD0Deg === null) ||
        (stB && !stB.has_baseline && state.externalD0DegB === null),
    );
  }, [stationMetaMap, state.stationA, state.stationB, state.coordFrame, state.externalD0Deg, state.externalD0DegB]);

  const contextWelchMatches = useMemo(() => {
    if (!welchPairRaw || pairBaselessBlocked) return false;
    if (welchPairSig !== currentPairSig) return false;
    return (
      welchPairRaw.episode_id === state.episodeId &&
      welchPairRaw.station_i === state.stationA &&
      welchPairRaw.station_j === state.stationB &&
      welchPairRaw.component === state.crossComponent &&
      welchPairRaw.coord_frame === state.coordFrame &&
      welchPairRaw.band === state.band
    );
  }, [
    welchPairRaw,
    welchPairSig,
    pairBaselessBlocked,
    currentPairSig,
    state.episodeId,
    state.stationA,
    state.stationB,
    state.crossComponent,
    state.coordFrame,
    state.band,
  ]);

  useEffect(() => {
    if (!options.showArcs || state.stationA === state.stationB || pairBaselessBlocked) {
      setLocalPairState(null);
      setLoadingPair(false);
      return;
    }
    if (contextWelchMatches || contextWelchOwnsFetch) {
      setLocalPairState(null);
      setLoadingPair(false);
      return;
    }
    if (!catalog || catalog.stations.length === 0) {
      return;
    }

    const ctrl = new AbortController();
    const fetchSig = currentPairSig;
    setLocalPairState((prev) => (prev && prev.sig === fetchSig ? prev : null));
    setLoadingPair(true);
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
        if (!ctrl.signal.aborted) {
          setLocalPairState({ sig: fetchSig, data: res });
        }
      })
      .catch(() => {
        if (!ctrl.signal.aborted) {
          setLocalPairState(null);
        }
      })
      .finally(() => {
        if (!ctrl.signal.aborted) {
          setLoadingPair(false);
        }
      });

    return () => ctrl.abort();
  }, [
    options.showArcs,
    contextWelchMatches,
    contextWelchOwnsFetch,
    pairBaselessBlocked,
    currentPairSig,
    catalog,
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

  const effectivePairRaw = useMemo(() => {
    if (!options.showArcs || state.stationA === state.stationB || pairBaselessBlocked) {
      return null;
    }
    if (contextWelchMatches) return welchPairRaw;
    if (localPairState && localPairState.sig === currentPairSig) {
      return localPairState.data;
    }
    return null;
  }, [
    options.showArcs,
    state.stationA,
    state.stationB,
    pairBaselessBlocked,
    contextWelchMatches,
    welchPairRaw,
    localPairState,
    currentPairSig,
  ]);

  const effectiveMaskedWelch = useMemo(() => {
    if (!effectivePairRaw) return null;
    if (contextWelchMatches && effectivePairRaw === welchPairRaw && maskedWelch) {
      return maskedWelch;
    }
    return applyWelchAlphaMask(effectivePairRaw, state.alpha, 0.05);
  }, [effectivePairRaw, contextWelchMatches, welchPairRaw, maskedWelch, state.alpha]);

  const pairMetricsAtCursor = useMemo(() => {
    if (!effectivePairRaw || !effectivePairRaw.available || !effectiveMaskedWelch) return null;
    const tsList = effectivePairRaw.window_timestamps_utc;
    if (tsList.length === 0) return null;

    const targetMs = effectiveCursorMs ?? Date.parse(tsList[0]);
    let bestIdx = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < tsList.length; i++) {
      const diff = Math.abs(Date.parse(tsList[i]) - targetMs);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIdx = i;
      }
    }

    return {
      gamma2: effectivePairRaw.gamma2[bestIdx] ?? null,
      gamma2Crit: effectiveMaskedWelch.gamma2Crit[bestIdx] ?? null,
      phaseDeg: effectivePairRaw.phase_deg[bestIdx] ?? null,
      phaseDegMasked: effectiveMaskedWelch.phaseDegMasked[bestIdx] ?? null,
      azimuthalM: effectiveMaskedWelch.azimuthalMMasked[bestIdx] ?? null,
      mAlias: effectivePairRaw.azimuthal?.m_alias ?? null,
      isSignificant: Boolean(effectiveMaskedWelch.isSignificant[bestIdx]),
      isMeridionalGuarded: Boolean(effectivePairRaw.azimuthal?.is_meridional_guarded),
      deltaLonDeg: effectivePairRaw.azimuthal?.delta_lon_deg ?? null,
    };
  }, [effectivePairRaw, effectiveMaskedWelch, effectiveCursorMs]);

  // 3. Optional all-network matrix slice from stationA at cursor window
  const matrixWindowIndex = useMemo(
    () =>
      computeCursorWindowIndex(
        effectiveCursorMs,
        episodeBoundsMs,
        state.windowSamples,
        state.stepSamples,
      ),
    [effectiveCursorMs, episodeBoundsMs, state.windowSamples, state.stepSamples],
  );

  const matrixParamSig = useMemo(
    () =>
      JSON.stringify([
        state.episodeId,
        state.crossComponent,
        state.coordFrame,
        state.band,
        state.tMaxPc6Sec,
        state.L,
        state.M,
        state.windowSamples,
        state.stepSamples,
        state.spectrumType,
      ]),
    [
      state.episodeId,
      state.crossComponent,
      state.coordFrame,
      state.band,
      state.tMaxPc6Sec,
      state.L,
      state.M,
      state.windowSamples,
      state.stepSamples,
      state.spectrumType,
    ],
  );

  useEffect(() => {
    if (!options.showMatrixArcs || !options.showArcs) {
      setMatrixState(null);
      setLoadingMatrix(false);
      return;
    }
    if (!catalog || catalog.stations.length === 0) {
      setMatrixState((prev) => (prev && prev.paramSig !== matrixParamSig ? null : prev));
      return;
    }

    const ctrl = new AbortController();
    const fetchParamSig = matrixParamSig;
    setMatrixState((prev) => (prev && prev.paramSig === matrixParamSig ? prev : null));

    const timer = setTimeout(() => {
      setLoadingMatrix(true);
      computeWelchMatrix(
        {
          episode_id: state.episodeId,
          component: state.crossComponent,
          coord_frame: state.coordFrame,
          band: state.band,
          t_max_pc6_sec: state.tMaxPc6Sec,
          L: state.L,
          M: state.M,
          window_samples: state.windowSamples,
          step_samples: state.stepSamples,
          alpha: 0.05,
          spectrum_type: state.spectrumType,
          window_index: matrixWindowIndex,
        },
        ctrl.signal,
      )
        .then((res) => {
          if (!ctrl.signal.aborted) {
            setMatrixState({ paramSig: fetchParamSig, data: res });
          }
        })
        .catch(() => {
          if (!ctrl.signal.aborted) {
            setMatrixState(null);
          }
        })
        .finally(() => {
          if (!ctrl.signal.aborted) {
            setLoadingMatrix(false);
          }
        });
    }, 240);

    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [
    options.showMatrixArcs,
    options.showArcs,
    catalog,
    matrixParamSig,
    state.episodeId,
    state.crossComponent,
    state.coordFrame,
    state.band,
    state.tMaxPc6Sec,
    state.L,
    state.M,
    state.windowSamples,
    state.stepSamples,
    state.spectrumType,
    matrixWindowIndex,
  ]);

  const matrixArcsFromA = useMemo<MatrixArcEdge[]>(() => {
    if (
      !options.showMatrixArcs ||
      !matrixState ||
      matrixState.paramSig !== matrixParamSig ||
      matrixState.data.gamma2.length === 0
    ) {
      return [];
    }
    const matrixRaw = matrixState.data;
    const idxA = matrixRaw.station_codes.indexOf(state.stationA);
    if (idxA === -1) return [];
    const g2Row = matrixRaw.gamma2[0]?.[idxA];
    const gcRow = matrixRaw.gamma2_crit[0]?.[idxA];
    const phRow = matrixRaw.phase_deg?.[0]?.[idxA] ?? matrixRaw.phase_deg_masked[0]?.[idxA];
    if (!g2Row || !gcRow) return [];

    const edges: MatrixArcEdge[] = [];
    for (let j = 0; j < matrixRaw.station_codes.length; j++) {
      if (j === idxA) continue;
      const codeJ = matrixRaw.station_codes[j];
      if (codeJ === state.stationB) continue;
      const g2 = g2Row[j] ?? null;
      const gc0 = gcRow[j] ?? null;
      if (g2 === null || gc0 === null || !Number.isFinite(g2) || !Number.isFinite(gc0)) {
        continue;
      }
      const gc = scaleGamma2CritForAlpha(gc0, 0.05, state.alpha);
      const isSig = g2 > gc;
      if (!isSig) continue;
      edges.push({
        stationI: state.stationA,
        stationJ: codeJ,
        gamma2: g2,
        gamma2Crit: gc,
        phaseDeg: phRow?.[j] ?? null,
        isSignificant: true,
      });
    }
    return edges;
  }, [options.showMatrixArcs, matrixState, matrixParamSig, state.stationA, state.stationB, state.alpha]);

  return {
    effectiveCursorMs,
    snapshotWindow,
    snapshotMasked,
    loadingSnapshot,
    snapshotError,
    pairMetricsAtCursor,
    loadingPair: contextWelchOwnsFetch ? loadingWelch : loadingPair,
    matrixArcsFromA,
    loadingMatrix,
    matrixWindowIndex,
  };
}
