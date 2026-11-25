import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Compass,
  GitBranch,
  Globe2,
  Maximize2,
  Minimize2,
  Network,
  RotateCcw,
  Sun,
  Tag,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { StationInfo } from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { DEFAULT_MERIDIONAL_CHAIN } from "../../store/analysisReducer";
import {
  IGRF14_NORTH_POLE,
  buildMltMeridianLine,
  buildNightHemispherePolygon,
  buildTerminatorLine,
  computeMLT,
  computeSubsolarPoint,
  formatMltHours,
  geoToMagneticDipole,
} from "../../utils/astronomy";
import {
  buildExportFilename,
  buildStationMapSvg,
  formatCsvWithMetadata,
  type MapSvgArcItem,
  type MapSvgStationItem,
  type StationMapSvgScene,
} from "../../utils/exportUtils";
import { formatNum, formatUtcFull, formatUtcShort } from "../../utils/formatters";
import {
  MAP_REGION_PRESETS,
  createFittedMapProjection,
  createMapPathGenerator,
  getWorldGeoData,
  resolveLabelVisibility,
  scalePanOffsetForViewport,
  type LabelCandidate,
  type MapProjectionType,
  type MapRegionPreset,
} from "../../utils/mapProjection";
import { canonicalizeStationPair } from "../../utils/stationPair";
import {
  buildStokesEllipseGeometry,
  coherenceArcColor,
  coherenceArcStrokeWidth,
  computeArcCoherenceMetrics,
  computeLocalNorthBearingDeg,
  type ArcCoherenceMetrics,
  type StokesEllipseGeometry,
} from "../../utils/stokesEllipse";
import { ExportToolbar } from "../Common/ExportToolbar";
import { useStationMapData } from "./useStationMapData";

const REGIONAL_ANCHORS = new Set([
  "VIC",
  "YKC",
  "RES",
  "OTT",
  "STJ",
  "GUI",
  "VAL",
  "HRN",
  "SOD",
  "LVV",
  "BEL",
  "PEG",
  "IZN",
  "NUR",
]);

interface StationRenderNode {
  meta: StationInfo;
  x: number;
  y: number;
  mlatDeg: number;
  mltHours: number;
  d0Deg: number;
  available: boolean;
  isValid: boolean;
  stokes: ReturnType<typeof useStationMapData>["snapshotMasked"] extends infer _
    ? NonNullable<
        NonNullable<ReturnType<typeof useStationMapData>["snapshotMasked"]>["stations"][number]
      >["stokes_2d"]
    : never;
  warning: string | null;
  geometry: StokesEllipseGeometry;
  isStationA: boolean;
  isStationB: boolean;
  isPendingPick: boolean;
  inFlrChain: boolean;
}

interface ArcRenderNode {
  key: string;
  pathD: string;
  metrics: ArcCoherenceMetrics;
  isActivePair: boolean;
  labelX: number | null;
  labelY: number | null;
  labelText: string | null;
}

export const StationMapPanel: React.FC = () => {
  const {
    state,
    dispatch,
    catalog,
    stationMetaMap,
    rawSpaceWeather,
    flrProfile,
    flrError,
    loadFlrAtCursor,
    loadingFlr,
  } = useAnalysis();

  const [collapsed, setCollapsed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [regionPreset, setRegionPreset] = useState<MapRegionPreset>("all");
  const [projectionType, setProjectionType] = useState<MapProjectionType>("lcc");
  const [interactionMode, setInteractionMode] = useState<"pair" | "chain">("pair");
  const [pendingStart, setPendingStart] = useState<string | null>(null);

  const [showTerminator, setShowTerminator] = useState(true);
  const [showEllipses, setShowEllipses] = useState(true);
  const [showArcs, setShowArcs] = useState(true);
  const [showMatrixArcs, setShowMatrixArcs] = useState(false);
  const [showOulu, setShowOulu] = useState(true);
  const [showAllLabels, setShowAllLabels] = useState(false);
  const [ellipseSizeMode, setEllipseSizeMode] = useState<"log" | "sqrt">("log");

  const [zoomScale, setZoomScale] = useState(1);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const dragRef = useRef<{
    startX: number;
    startY: number;
    origPanX: number;
    origPanY: number;
    moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);

  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null);
  const [stageWidth, setStageWidth] = useState(880);
  const [hoveredStation, setHoveredStation] = useState<string | null>(null);
  const [hoveredOulu, setHoveredOulu] = useState(false);
  const [hoveredArcKey, setHoveredArcKey] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    if (!stageEl) return;
    const ro = new ResizeObserver((entries) => {
      const nextW = Math.floor(entries[0]?.contentRect.width ?? stageEl.clientWidth);
      if (nextW >= 240) {
        setStageWidth(nextW);
      }
    });
    ro.observe(stageEl);
    return () => ro.disconnect();
  }, [stageEl]);

  useEffect(() => {
    if (!stageEl) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.85 : 1.18;
      setZoomScale((prev) => Math.min(8, Math.max(0.7, Number((prev * factor).toFixed(3)))));
    };
    stageEl.addEventListener("wheel", onWheel, { passive: false });
    return () => stageEl.removeEventListener("wheel", onWheel);
  }, [stageEl]);

  const mapHeight = expanded ? 520 : regionPreset === "all" ? 390 : 410;

  const {
    effectiveCursorMs,
    snapshotWindow,
    snapshotMasked,
    loadingSnapshot,
    snapshotError,
    pairMetricsAtCursor,
    loadingPair,
    matrixArcsFromA,
    loadingMatrix,
    matrixWindowIndex,
  } = useStationMapData({ showArcs, showMatrixArcs });

  const magnetometers = useMemo(() => {
    if (!catalog) return [];
    return catalog.stations.filter((s) => s.station_type === "magnetometer");
  }, [catalog]);

  const ouluMeta = useMemo(
    () => catalog?.stations.find((s) => s.station_type === "neutron_monitor") ?? null,
    [catalog],
  );

  const worldGeo = useMemo(() => getWorldGeoData(), []);

  const buildSceneForDimensions = useCallback(
    (w: number, h: number, theme: "light" | "dark"): {
      projection: ReturnType<typeof createFittedMapProjection>;
      pathGen: ReturnType<typeof createMapPathGenerator>;
      stationNodes: StationRenderNode[];
      arcNodes: ArcRenderNode[];
      ouluPos: [number, number] | null;
      subsolarPos: [number, number] | null;
      nmpPos: [number, number] | null;
      graticulePathD: string;
      landPathsD: string[];
      bordersPathD: string;
      nightPolyPathD: string | null;
      terminatorPathD: string | null;
      mlt12PathD: string | null;
      mlt12LabelPos: [number, number] | null;
      mlt00PathD: string | null;
      labels: ReturnType<typeof resolveLabelVisibility>;
      svgScene: StationMapSvgScene;
    } => {
      const scaledPan = scalePanOffsetForViewport(panOffset, {
        projectionType,
        regionPreset,
        stations: magnetometers,
        fromWidth: stageWidth,
        fromHeight: mapHeight,
        toWidth: w,
        toHeight: h,
      });
      const proj = createFittedMapProjection({
        projectionType,
        regionPreset,
        width: w,
        height: h,
        stations: magnetometers,
        zoomScale,
        panX: scaledPan.x,
        panY: scaledPan.y,
      });
      const pathGen = createMapPathGenerator(proj);

      const snapshotMap = new Map(
        (snapshotMasked?.stations ?? []).map((st) => [st.station_iaga, st]),
      );

      let scaleMinI = Number.POSITIVE_INFINITY;
      let scaleMaxI = Number.NEGATIVE_INFINITY;
      for (const st of snapshotMasked?.stations ?? []) {
        const pwr = st.stokes_2d?.I;
        if (
          st.is_valid &&
          st.stokes_2d?.is_significant_p &&
          pwr !== null &&
          pwr !== undefined &&
          Number.isFinite(pwr) &&
          pwr > 0
        ) {
          if (pwr < scaleMinI) scaleMinI = pwr;
          if (pwr > scaleMaxI) scaleMaxI = pwr;
        }
      }
      if (!Number.isFinite(scaleMinI) || !Number.isFinite(scaleMaxI)) {
        scaleMinI = 1;
        scaleMaxI = 100;
      }

      const utcMs = effectiveCursorMs ?? Date.UTC(2026, 9, 10, 12, 0, 0);
      const chainSet = new Set(state.meridionalChain);

      const stationNodes: StationRenderNode[] = [];
      for (const meta of magnetometers) {
        const pt = proj([meta.lon_deg, meta.lat_deg]);
        if (!pt || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) continue;

        const snap = snapshotMap.get(meta.iaga_code);
        const available = snap ? snap.available : snapshotMasked === null;
        const isValid = snap ? snap.is_valid : false;
        const stokes = snap?.stokes_2d ?? null;
        const d0Deg = snap?.d0_deg ?? 0;
        const localNorthDeg = computeLocalNorthBearingDeg(proj, meta.lon_deg, meta.lat_deg);

        const geometry = buildStokesEllipseGeometry({
          stationIaga: meta.iaga_code,
          available,
          isValid,
          stokes,
          coordFrame: state.coordFrame,
          d0Deg,
          localNorthDeg,
          scaleMinI,
          scaleMaxI,
          sizeMode: ellipseSizeMode,
        });

        const magCoords = geoToMagneticDipole(meta.lat_deg, meta.lon_deg);
        const mltHours = computeMLT(meta.lat_deg, meta.lon_deg, utcMs);

        stationNodes.push({
          meta,
          x: pt[0],
          y: pt[1],
          mlatDeg: magCoords.mlatDeg,
          mltHours,
          d0Deg,
          available,
          isValid,
          stokes,
          warning: snap?.warning ?? null,
          geometry,
          isStationA: meta.iaga_code === state.stationA,
          isStationB: meta.iaga_code === state.stationB,
          isPendingPick: meta.iaga_code === pendingStart,
          inFlrChain: chainSet.has(meta.iaga_code),
        });
      }

      // Build coherence arcs
      const arcNodes: ArcRenderNode[] = [];
      const seenPairs = new Set<string>();
      const addArc = (
        codeI: string,
        codeJ: string,
        opts: {
          gamma2: number | null;
          gamma2Crit: number | null;
          phaseDeg: number | null;
          azimuthalM?: number | null;
          mAlias?: number | null;
          isMeridionalGuarded?: boolean;
          isActivePair: boolean;
        },
      ) => {
        if (codeI === codeJ) return;
        const canonicalPairKey = [codeI, codeJ].sort().join("--");
        if (!opts.isActivePair && seenPairs.has(canonicalPairKey)) return;
        seenPairs.add(canonicalPairKey);

        const metaI = stationMetaMap.get(codeI);
        const metaJ = stationMetaMap.get(codeJ);
        if (!metaI || !metaJ) return;

        const d =
          pathGen({
            type: "LineString",
            coordinates: [
              [metaI.lon_deg, metaI.lat_deg],
              [metaJ.lon_deg, metaJ.lat_deg],
            ],
          }) ?? "";
        if (!d) return;

        const metrics = computeArcCoherenceMetrics({
          stationI: codeI,
          latI: metaI.lat_deg,
          lonI: metaI.lon_deg,
          stationJ: codeJ,
          latJ: metaJ.lat_deg,
          lonJ: metaJ.lon_deg,
          gamma2: opts.gamma2,
          gamma2Crit: opts.gamma2Crit,
          phaseDeg: opts.phaseDeg,
          azimuthalMOverride: opts.azimuthalM,
          mAliasOverride: opts.mAlias,
          isMeridionalGuardedOverride: opts.isMeridionalGuarded,
        });

        let labelX: number | null = null;
        let labelY: number | null = null;
        let labelText: string | null = null;
        if (opts.isActivePair) {
          const mid = pathGen.centroid({
            type: "LineString",
            coordinates: [
              [metaI.lon_deg, metaI.lat_deg],
              [metaJ.lon_deg, metaJ.lat_deg],
            ],
          });
          if (mid && Number.isFinite(mid[0]) && Number.isFinite(mid[1])) {
            labelX = mid[0];
            labelY = mid[1];
          }
          if (metrics.gamma2 !== null) {
            const gStr = `γ²=${metrics.gamma2.toFixed(2)}`;
            const pStr =
              metrics.phaseDegMasked !== null
                ? ` Δφ=${metrics.phaseDegMasked.toFixed(0)}°`
                : "";
            const mStr =
              metrics.azimuthalM !== null
                ? ` m=${metrics.azimuthalM.toFixed(1)}`
                : metrics.isMeridionalGuarded
                  ? " (мерид.)"
                  : "";
            labelText = `${codeI}–${codeJ}: ${gStr}${pStr}${mStr}`;
          } else {
            labelText = `${codeI} ↔ ${codeJ}`;
          }
        }

        arcNodes.push({
          key: opts.isActivePair ? `active:${codeI}-${codeJ}` : `arc:${codeI}-${codeJ}`,
          pathD: d,
          metrics,
          isActivePair: opts.isActivePair,
          labelX,
          labelY,
          labelText,
        });
      };

      if (showArcs) {
        // 1. Active pair A <-> B
        if (state.stationA !== state.stationB) {
          addArc(state.stationA, state.stationB, {
            gamma2: pairMetricsAtCursor?.gamma2 ?? null,
            gamma2Crit: pairMetricsAtCursor?.gamma2Crit ?? null,
            phaseDeg: pairMetricsAtCursor?.phaseDeg ?? null,
            azimuthalM: pairMetricsAtCursor?.azimuthalM,
            mAlias: pairMetricsAtCursor?.mAlias,
            isMeridionalGuarded: pairMetricsAtCursor?.isMeridionalGuarded,
            isActivePair: true,
          });
        }

        // 2. FLR meridional chain arcs when loaded or in chain mode
        if (flrProfile && flrProfile.station_codes.length >= 2) {
          for (let i = 0; i < flrProfile.station_codes.length - 1; i++) {
            addArc(flrProfile.station_codes[i], flrProfile.station_codes[i + 1], {
              gamma2: flrProfile.pair_coherence[i] ?? null,
              gamma2Crit: flrProfile.pair_gamma2_crit[i] ?? null,
              phaseDeg: flrProfile.pair_phase_deg[i] ?? null,
              isActivePair: false,
            });
          }
        } else if (interactionMode === "chain" && state.meridionalChain.length >= 2) {
          for (let i = 0; i < state.meridionalChain.length - 1; i++) {
            addArc(state.meridionalChain[i], state.meridionalChain[i + 1], {
              gamma2: null,
              gamma2Crit: null,
              phaseDeg: null,
              isActivePair: false,
            });
          }
        }

        // 3. Optional matrix arcs from station A
        for (const edge of matrixArcsFromA) {
          addArc(edge.stationI, edge.stationJ, {
            gamma2: edge.gamma2,
            gamma2Crit: edge.gamma2Crit,
            phaseDeg: edge.phaseDeg,
            isActivePair: false,
          });
        }
      }

      // Astronomy layers
      let nightPolyPathD: string | null = null;
      let terminatorPathD: string | null = null;
      let mlt12PathD: string | null = null;
      let mlt12LabelPos: [number, number] | null = null;
      let mlt00PathD: string | null = null;
      let subsolarPos: [number, number] | null = null;

      if (showTerminator) {
        const nightPoly = buildNightHemispherePolygon(utcMs, 1);
        const termLine = buildTerminatorLine(utcMs, 2);
        const mlt12Line = buildMltMeridianLine(12, utcMs, 2);
        const mlt00Line = buildMltMeridianLine(0, utcMs, 2);

        nightPolyPathD = pathGen(nightPoly) ?? null;
        terminatorPathD = pathGen(termLine) ?? null;
        mlt12PathD = pathGen(mlt12Line) ?? null;
        mlt00PathD = pathGen(mlt00Line) ?? null;

        const sub = computeSubsolarPoint(utcMs);
        const subPt = proj([sub.lonDeg, sub.latDeg]);
        if (
          subPt &&
          subPt[0] >= 0 &&
          subPt[0] <= w &&
          subPt[1] >= 0 &&
          subPt[1] <= h
        ) {
          subsolarPos = [subPt[0], subPt[1]];
        }

        for (const coord of mlt12Line.coordinates) {
          if (coord[1] < 42 || coord[1] > 72) continue;
          const p = proj(coord as [number, number]);
          if (p && p[0] > 36 && p[0] < w - 36 && p[1] > 18 && p[1] < h - 18) {
            mlt12LabelPos = [p[0], p[1]];
            break;
          }
        }
      }

      const nmpRaw = proj([IGRF14_NORTH_POLE.lonDeg, IGRF14_NORTH_POLE.latDeg]);
      const nmpPos: [number, number] | null =
        nmpRaw && nmpRaw[0] >= 0 && nmpRaw[0] <= w && nmpRaw[1] >= 0 && nmpRaw[1] <= h
          ? [nmpRaw[0], nmpRaw[1]]
          : null;

      let ouluPos: [number, number] | null = null;
      if (showOulu && ouluMeta) {
        const op = proj([ouluMeta.lon_deg, ouluMeta.lat_deg]);
        if (op && Number.isFinite(op[0]) && Number.isFinite(op[1])) {
          ouluPos = [op[0], op[1]];
        }
      }

      const graticulePathD = pathGen(worldGeo.graticule) ?? "";
      const landPathsD = worldGeo.land.features.map((f) => pathGen(f) ?? "");
      const bordersPathD = pathGen(worldGeo.borders) ?? "";

      const labelCandidates: LabelCandidate[] = stationNodes.map((n) => {
        let priority = 100;
        if (n.isStationA) priority = 1000;
        else if (n.isStationB) priority = 960;
        else if (n.meta.iaga_code === hoveredStation) priority = 930;
        else if (n.isPendingPick) priority = 910;
        else if (n.inFlrChain) priority = 700;
        else if (n.geometry.status === "significant") priority = 420;
        if (REGIONAL_ANCHORS.has(n.meta.iaga_code)) priority += 120;

        return {
          code: n.meta.iaga_code,
          x: n.x,
          y: n.y,
          priority,
          radiusPx: Math.max(4, n.geometry.majorRadius),
        };
      });
      if (ouluPos) {
        labelCandidates.push({
          code: "OULU",
          x: ouluPos[0],
          y: ouluPos[1],
          priority: hoveredOulu ? 920 : 380,
          radiusPx: 6,
        });
      }

      const labels = resolveLabelVisibility(labelCandidates, w, h, showAllLabels);

      const svgStations: MapSvgStationItem[] = stationNodes.map((n) => ({
        iagaCode: n.meta.iaga_code,
        x: n.x,
        y: n.y,
        geometry: n.geometry,
        isStationA: n.isStationA,
        isStationB: n.isStationB,
        inFlrChain: n.inFlrChain,
      }));
      const svgArcs: MapSvgArcItem[] = arcNodes.map((a) => ({
        pathD: a.pathD,
        metrics: a.metrics,
        isActivePair: a.isActivePair,
        labelX: a.labelX,
        labelY: a.labelY,
        labelText: a.labelText,
      }));

      const bandDesc =
        state.band === "Pc6" ? `Pc6 (600-${state.tMaxPc6Sec} s)` : "Pc5 (150-600 s)";
      const caption = `${state.episodeId} | ${formatUtcFull(utcMs)} | ${bandDesc} | ${state.coordFrame} | alpha=${state.alpha} | proj=${projectionType.toUpperCase()}`;

      return {
        projection: proj,
        pathGen,
        stationNodes,
        arcNodes,
        ouluPos,
        subsolarPos,
        nmpPos,
        graticulePathD,
        landPathsD,
        bordersPathD,
        nightPolyPathD,
        terminatorPathD,
        mlt12PathD,
        mlt12LabelPos,
        mlt00PathD,
        labels,
        svgScene: {
          width: w,
          height: h,
          theme,
          title: `GeoMag Analyzer — 49-Station Polarization Map (${state.episodeId})`,
          caption,
          graticulePathD,
          landPathsD,
          bordersPathD,
          nightPolyPathD,
          terminatorPathD,
          mlt12PathD,
          mlt12LabelPos,
          mlt00PathD,
          subsolarPos,
          nmpPos,
          arcs: svgArcs,
          stations: svgStations,
          ouluPos,
          labels,
          showTerminator,
          showEllipses,
          showArcs,
        },
      };
    },
    [
      projectionType,
      regionPreset,
      magnetometers,
      zoomScale,
      panOffset.x,
      panOffset.y,
      stageWidth,
      mapHeight,
      snapshotMasked,
      effectiveCursorMs,
      state.meridionalChain,
      state.coordFrame,
      state.stationA,
      state.stationB,
      state.band,
      state.tMaxPc6Sec,
      state.alpha,
      state.episodeId,
      pendingStart,
      ellipseSizeMode,
      showArcs,
      stationMetaMap,
      pairMetricsAtCursor,
      flrProfile,
      interactionMode,
      matrixArcsFromA,
      showTerminator,
      showOulu,
      ouluMeta,
      worldGeo,
      hoveredStation,
      hoveredOulu,
      showAllLabels,
      showEllipses,
    ],
  );

  const liveScene = useMemo(
    () => buildSceneForDimensions(stageWidth, mapHeight, "dark"),
    [buildSceneForDimensions, stageWidth, mapHeight],
  );

  const handleStationClick = (code: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (interactionMode === "chain") {
      const exists = state.meridionalChain.includes(code);
      let nextChain: string[];
      if (exists) {
        if (state.meridionalChain.length <= 2) return;
        nextChain = state.meridionalChain.filter((c) => c !== code);
      } else {
        nextChain = [...state.meridionalChain, code].sort((a, b) => {
          const latA = stationMetaMap.get(a)?.lat_deg ?? 0;
          const latB = stationMetaMap.get(b)?.lat_deg ?? 0;
          return latB - latA;
        });
      }
      dispatch({ type: "SET_MERIDIONAL_CHAIN", chain: nextChain });
      return;
    }

    if (e.shiftKey) {
      setPendingStart(null);
      dispatch({ type: "SET_STATION_B", stationIaga: code });
      return;
    }

    if (pendingStart === null) {
      setPendingStart(code);
      dispatch({ type: "SET_STATION_A", stationIaga: code });
      return;
    }

    if (pendingStart === code) {
      setPendingStart(null);
      return;
    }

    const canon = canonicalizeStationPair(pendingStart, code, stationMetaMap);
    dispatch({ type: "SET_PAIR", stationA: canon.stationA, stationB: canon.stationB });
    setPendingStart(null);
  };

  const handleStageMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      origPanX: panOffset.x,
      origPanY: panOffset.y,
      moved: false,
    };
  };

  const handleStageMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (stageEl) {
      const rect = stageEl.getBoundingClientRect();
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
    if (dragRef.current) {
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        dragRef.current.moved = true;
      }
      setPanOffset({
        x: dragRef.current.origPanX + dx,
        y: dragRef.current.origPanY + dy,
      });
    }
  };

  const handleStageMouseUp = () => {
    if (dragRef.current?.moved) {
      suppressClickRef.current = true;
      setTimeout(() => {
        suppressClickRef.current = false;
      }, 0);
    }
    dragRef.current = null;
  };

  const hoveredStationNode = useMemo(
    () =>
      hoveredStation
        ? liveScene.stationNodes.find((n) => n.meta.iaga_code === hoveredStation) ?? null
        : null,
    [liveScene.stationNodes, hoveredStation],
  );

  const hoveredArcNode = useMemo(
    () =>
      hoveredArcKey
        ? liveScene.arcNodes.find((a) => a.key === hoveredArcKey) ?? null
        : null,
    [liveScene.arcNodes, hoveredArcKey],
  );

  const ouluValAtCursor = useMemo(() => {
    const oulu = rawSpaceWeather?.oulu;
    if (!oulu || !oulu.available || oulu.timestamps_utc.length === 0 || effectiveCursorMs === null) {
      return null;
    }
    let bestIdx = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < oulu.timestamps_utc.length; i++) {
      const d = Math.abs(Date.parse(oulu.timestamps_utc[i]) - effectiveCursorMs);
      if (d < bestDiff) {
        bestDiff = d;
        bestIdx = i;
      }
    }
    return oulu.delta_n_pct[bestIdx] ?? null;
  }, [rawSpaceWeather, effectiveCursorMs]);

  const exportBaseFilename = useMemo(
    () =>
      buildExportFilename(
        [
          "map",
          state.episodeId,
          regionPreset,
          state.band,
          state.coordFrame,
          snapshotWindow ? formatUtcShort(snapshotWindow.centerMs) : "all",
        ],
        "svg",
      ).replace(/\.svg$/, ""),
    [state.episodeId, regionPreset, state.band, state.coordFrame, snapshotWindow],
  );

  const getMapExportSvg = useCallback(() => {
    const expW = 960;
    const expH = regionPreset === "all" ? 440 : 460;
    const built = buildSceneForDimensions(expW, expH, state.exportTheme);
    return buildStationMapSvg(built.svgScene);
  }, [buildSceneForDimensions, regionPreset, state.exportTheme]);

  const getMapExportCsv = useCallback(() => {
    const metadata = {
      episode_id: state.episodeId,
      cursor_utc: effectiveCursorMs ? new Date(effectiveCursorMs).toISOString() : "",
      window_start_utc: snapshotWindow?.startIso ?? "",
      window_end_utc: snapshotWindow?.endIso ?? "",
      band: state.band,
      t_max_pc6_sec: state.band === "Pc6" ? state.tMaxPc6Sec : "",
      coord_frame: state.coordFrame,
      alpha: state.alpha,
      L: state.L,
      M: state.M,
      window_samples: state.windowSamples,
      projection: projectionType,
      region_preset: regionPreset,
      active_pair: `${state.stationA}-${state.stationB}`,
      pair_gamma2: pairMetricsAtCursor?.gamma2 ?? "",
      pair_phase_deg: pairMetricsAtCursor?.phaseDegMasked ?? "",
      pair_azimuthal_m: pairMetricsAtCursor?.azimuthalM ?? "",
    };
    const columns = [
      "iaga_code",
      "station_name",
      "region",
      "lat_deg",
      "lon_deg",
      "mlat_deg",
      "mlt_hours",
      "has_baseline",
      "available",
      "is_valid",
      "d0_deg",
      "I_nT2",
      "p",
      "p_crit",
      "is_significant_p",
      "psi_deg",
      "chi_deg",
      "epsilon",
      "sense",
      "gamma2",
      "gamma2_crit",
      "arg_sxy_deg",
      "n_eff",
    ];
    const rows = liveScene.stationNodes.map((n) => [
      n.meta.iaga_code,
      n.meta.station_name,
      n.meta.region,
      Number(n.meta.lat_deg.toFixed(4)),
      Number(n.meta.lon_deg.toFixed(4)),
      Number(n.mlatDeg.toFixed(3)),
      Number(n.mltHours.toFixed(4)),
      n.meta.has_baseline,
      n.available,
      n.isValid,
      Number(n.d0Deg.toFixed(3)),
      n.stokes?.I ?? null,
      n.stokes?.p ?? null,
      n.stokes?.p_crit ?? null,
      n.stokes?.is_significant_p ?? null,
      n.stokes?.psi_deg ?? null,
      n.stokes?.chi_deg ?? null,
      n.stokes?.epsilon ?? null,
      n.geometry.sense,
      n.stokes?.gamma2 ?? null,
      n.stokes?.gamma2_crit ?? null,
      n.stokes?.arg_sxy_deg ?? null,
      n.stokes?.n_eff ?? null,
    ]);
    return formatCsvWithMetadata(metadata, columns, rows);
  }, [
    state.episodeId,
    state.band,
    state.tMaxPc6Sec,
    state.coordFrame,
    state.alpha,
    state.L,
    state.M,
    state.windowSamples,
    state.stationA,
    state.stationB,
    effectiveCursorMs,
    snapshotWindow,
    projectionType,
    regionPreset,
    pairMetricsAtCursor,
    liveScene.stationNodes,
  ]);

  const sigCount = liveScene.stationNodes.filter((n) => n.geometry.status === "significant").length;
  const cwCount = liveScene.stationNodes.filter((n) => n.geometry.sense === "CW").length;
  const ccwCount = liveScene.stationNodes.filter((n) => n.geometry.sense === "CCW").length;

  return (
    <section className="panel-card" data-testid="station-map-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <Globe2 size={15} color="#38bdf8" />
          <span>0. Офлайн-карта сети INTERMAGNET (49 станций, D3-geo)</span>
          <span className="pill mono">
            {sigCount}/{liveScene.stationNodes.length} знач. (CW:{cwCount} · CCW:{ccwCount})
          </span>
          {snapshotWindow && (
            <span className="pill pill-accent mono">
              {formatUtcShort(Date.parse(snapshotWindow.startIso))}–
              {formatUtcShort(Date.parse(snapshotWindow.endIso))}
            </span>
          )}
          {(loadingSnapshot || loadingPair || loadingMatrix) && (
            <span className="pill">Обновление…</span>
          )}
        </div>

        <div className="panel-controls">
          <ExportToolbar
            baseFilename={exportBaseFilename}
            getSvg={getMapExportSvg}
            svgWidth={960}
            svgHeight={(regionPreset === "all" ? 440 : 460) + 44}
            getCsv={getMapExportCsv}
            disabled={liveScene.stationNodes.length === 0}
          />
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setExpanded((v) => !v)}
            title={expanded ? "Обычный размер карты" : "Увеличить высоту карты"}
          >
            {expanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setCollapsed((v) => !v)}
            title={collapsed ? "Развернуть карту" : "Свернуть карту"}
          >
            {collapsed ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="panel-body">
          <div className="map-toolbar">
            <div className="quick-chips">
              {(Object.keys(MAP_REGION_PRESETS) as MapRegionPreset[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`chip-btn ${regionPreset === key ? "active" : ""}`}
                  onClick={() => {
                    setRegionPreset(key);
                    setZoomScale(1);
                    setPanOffset({ x: 0, y: 0 });
                  }}
                >
                  {MAP_REGION_PRESETS[key].label}
                </button>
              ))}
            </div>

            <div className="segmented" style={{ width: 170 }}>
              <button
                type="button"
                className={projectionType === "lcc" ? "active" : ""}
                onClick={() => setProjectionType("lcc")}
                title="Равноугольная коническая проекция Ламберта"
              >
                Ламберт LCC
              </button>
              <button
                type="button"
                className={projectionType === "laea" ? "active" : ""}
                onClick={() => setProjectionType("laea")}
                title="Азимутальная равновеликая проекция Ламберта"
              >
                LAEA
              </button>
            </div>

            <div className="segmented" style={{ width: 196 }}>
              <button
                type="button"
                className={interactionMode === "pair" ? "active" : ""}
                onClick={() => {
                  setInteractionMode("pair");
                  setPendingStart(null);
                }}
                title="Клик по 2 станциям выбирает пару (Запад→Восток) для кросс-спектра"
              >
                Пара (A↔B)
              </button>
              <button
                type="button"
                className={interactionMode === "chain" ? "active" : ""}
                onClick={() => {
                  setInteractionMode("chain");
                  setPendingStart(null);
                }}
                title="Клик добавляет/убирает станции из меридиональной цепочки FLR"
              >
                Цепочка FLR
              </button>
            </div>

            {interactionMode === "chain" && (
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() =>
                    dispatch({
                      type: "SET_MERIDIONAL_CHAIN",
                      chain: DEFAULT_MERIDIONAL_CHAIN,
                    })
                  }
                >
                  <GitBranch size={12} />
                  <span>HRN..PEG</span>
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  disabled={loadingFlr}
                  onClick={() => void loadFlrAtCursor(matrixWindowIndex)}
                >
                  {loadingFlr ? "FLR…" : "Расчёт FLR"}
                </button>
              </div>
            )}

            <div className="quick-chips">
              <button
                type="button"
                className={`chip-btn ${showTerminator ? "active" : ""}`}
                onClick={() => setShowTerminator((v) => !v)}
                title="Показать терминатор день/ночь и меридиан MLT = 12"
              >
                <Sun size={11} style={{ marginRight: 3, verticalAlign: "-1px" }} />
                Терминатор + MLT=12
              </button>
              <button
                type="button"
                className={`chip-btn ${showEllipses ? "active" : ""}`}
                onClick={() => setShowEllipses((v) => !v)}
              >
                <Compass size={11} style={{ marginRight: 3, verticalAlign: "-1px" }} />
                Эллипсы
              </button>
              <button
                type="button"
                className={`chip-btn ${ellipseSizeMode === "log" ? "active" : ""}`}
                onClick={() => setEllipseSizeMode((m) => (m === "log" ? "sqrt" : "log"))}
                title="Масштаб размера эллипсов по мощности I"
              >
                R ∝ {ellipseSizeMode === "log" ? "log(I)" : "√I"}
              </button>
              <button
                type="button"
                className={`chip-btn ${showArcs ? "active" : ""}`}
                onClick={() => setShowArcs((v) => !v)}
              >
                Дуги γ²
              </button>
              <button
                type="button"
                className={`chip-btn ${showMatrixArcs ? "active" : ""}`}
                onClick={() => setShowMatrixArcs((v) => !v)}
                title="Показать все статистически значимые дуги от станции A"
              >
                <Network size={11} style={{ marginRight: 3, verticalAlign: "-1px" }} />
                Сеть от {state.stationA}
              </button>
              <button
                type="button"
                className={`chip-btn ${showOulu ? "active" : ""}`}
                onClick={() => setShowOulu((v) => !v)}
              >
                OULU NM
              </button>
              <button
                type="button"
                className={`chip-btn ${showAllLabels ? "active" : ""}`}
                onClick={() => setShowAllLabels((v) => !v)}
              >
                <Tag size={11} style={{ marginRight: 3, verticalAlign: "-1px" }} />
                Все подписи
              </button>
            </div>

            <div style={{ display: "flex", gap: 4, marginLeft: "auto" }}>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setZoomScale((z) => Math.min(8, z * 1.25))}
                title="Приблизить"
              >
                <ZoomIn size={12} />
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setZoomScale((z) => Math.max(0.7, z * 0.8))}
                title="Отдалить"
              >
                <ZoomOut size={12} />
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => {
                  setZoomScale(1);
                  setPanOffset({ x: 0, y: 0 });
                }}
                title="Сбросить масштаб"
              >
                <RotateCcw size={12} />
              </button>
            </div>
          </div>

          {pendingStart && (
            <div className="banner banner-info" style={{ padding: "5px 10px" }}>
              <span>
                Выбрана первая станция пары: <b className="mono">{pendingStart}</b>. Нажмите вторую
                станцию на карте (повторный клик на {pendingStart} отменит выбор).
              </span>
            </div>
          )}

          {snapshotError && (
            <div className="banner banner-danger" style={{ padding: "5px 10px" }}>
              {snapshotError}
            </div>
          )}

          {flrError && (
            <div className="banner banner-danger" data-testid="map-flr-error" style={{ padding: "5px 10px" }}>
              FLR: {flrError}
            </div>
          )}

          <div
            ref={setStageEl}
            className="map-stage"
            style={{ height: mapHeight }}
            onMouseDown={handleStageMouseDown}
            onMouseMove={handleStageMouseMove}
            onMouseUp={handleStageMouseUp}
            onMouseLeave={() => {
              dragRef.current = null;
              setHoveredStation(null);
              setHoveredOulu(false);
              setHoveredArcKey(null);
            }}
          >
            <svg
              className="map-svg"
              width={stageWidth}
              height={mapHeight}
              viewBox={`0 0 ${stageWidth} ${mapHeight}`}
              role="img"
              aria-label="Интерактивная карта 49 геомагнитных станций INTERMAGNET"
            >
              <defs>
                <clipPath id="live-map-clip">
                  <rect x={0} y={0} width={stageWidth} height={mapHeight} />
                </clipPath>
              </defs>
              <rect x={0} y={0} width={stageWidth} height={mapHeight} fill="#070d19" />

              <g clipPath="url(#live-map-clip)">
                {liveScene.graticulePathD && (
                  <path
                    d={liveScene.graticulePathD}
                    fill="none"
                    stroke="rgba(51, 65, 85, 0.42)"
                    strokeWidth={0.7}
                  />
                )}
                {liveScene.landPathsD.map((d, idx) =>
                  d ? (
                    <path
                      key={`land-${idx}`}
                      d={d}
                      fill="#162238"
                      stroke="#334155"
                      strokeWidth={0.8}
                    />
                  ) : null,
                )}
                {liveScene.bordersPathD && (
                  <path
                    d={liveScene.bordersPathD}
                    fill="none"
                    stroke="#1e293b"
                    strokeWidth={0.55}
                  />
                )}

                {showTerminator && (
                  <>
                    {liveScene.nightPolyPathD && (
                      <path
                        d={liveScene.nightPolyPathD}
                        fill="rgba(2, 6, 23, 0.44)"
                        stroke="none"
                        data-testid="map-night-hemisphere"
                      />
                    )}
                    {liveScene.terminatorPathD && (
                      <path
                        d={liveScene.terminatorPathD}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth={1.5}
                        strokeDasharray="6,4"
                        data-testid="map-terminator"
                      />
                    )}
                    {liveScene.mlt00PathD && (
                      <path
                        d={liveScene.mlt00PathD}
                        fill="none"
                        stroke="#64748b"
                        strokeWidth={1}
                        strokeDasharray="3,4"
                        opacity={0.7}
                      />
                    )}
                    {liveScene.mlt12PathD && (
                      <path
                        d={liveScene.mlt12PathD}
                        fill="none"
                        stroke="#fbbf24"
                        strokeWidth={2}
                        data-testid="map-mlt12"
                      />
                    )}
                    {liveScene.mlt12LabelPos && (
                      <g>
                        <rect
                          x={liveScene.mlt12LabelPos[0] - 28}
                          y={liveScene.mlt12LabelPos[1] - 9}
                          width={56}
                          height={16}
                          rx={3}
                          fill="rgba(15, 23, 42, 0.9)"
                          stroke="#fbbf24"
                          strokeWidth={0.9}
                        />
                        <text
                          x={liveScene.mlt12LabelPos[0]}
                          y={liveScene.mlt12LabelPos[1] + 2.5}
                          textAnchor="middle"
                          fill="#fbbf24"
                          fontFamily="var(--font-mono)"
                          fontSize={9.5}
                          fontWeight={700}
                        >
                          MLT = 12
                        </text>
                      </g>
                    )}
                    {liveScene.subsolarPos && (
                      <circle
                        cx={liveScene.subsolarPos[0]}
                        cy={liveScene.subsolarPos[1]}
                        r={4.5}
                        fill="#fbbf24"
                        stroke="#0f172a"
                        strokeWidth={1}
                      />
                    )}
                  </>
                )}

                {liveScene.nmpPos && (
                  <g>
                    <circle
                      cx={liveScene.nmpPos[0]}
                      cy={liveScene.nmpPos[1]}
                      r={3.5}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth={1.2}
                    />
                    <text
                      x={liveScene.nmpPos[0] + 6}
                      y={liveScene.nmpPos[1] + 3}
                      fill="#fbbf24"
                      fontFamily="var(--font-mono)"
                      fontSize={8.5}
                    >
                      NMP
                    </text>
                  </g>
                )}

                {/* Coherence arcs */}
                {showArcs &&
                  liveScene.arcNodes.map((arc) => {
                    const col = coherenceArcColor(
                      arc.metrics.gamma2,
                      arc.metrics.isSignificant,
                      "dark",
                    );
                    const sw = coherenceArcStrokeWidth(arc.metrics.gamma2, arc.isActivePair);
                    return (
                      <g
                        key={arc.key}
                        data-arc-key={arc.key}
                        onMouseEnter={() => {
                          setHoveredArcKey(arc.key);
                          setHoveredStation(null);
                        }}
                        onMouseLeave={() => setHoveredArcKey(null)}
                        style={{ cursor: "pointer" }}
                      >
                        <path
                          d={arc.pathD}
                          fill="none"
                          stroke="transparent"
                          strokeWidth={12}
                        />
                        <path
                          d={arc.pathD}
                          fill="none"
                          stroke={col}
                          strokeWidth={sw}
                          strokeDasharray={arc.metrics.isSignificant ? undefined : "5,4"}
                          opacity={arc.isActivePair ? 0.96 : 0.72}
                          data-testid={arc.isActivePair ? "active-coherence-arc" : `coherence-${arc.key}`}
                        />
                        {arc.labelText && arc.labelX !== null && arc.labelY !== null && (
                          <g pointerEvents="none">
                            <rect
                              x={arc.labelX - Math.max(88, arc.labelText.length * 6.2 + 12) * 0.5}
                              y={arc.labelY - 10}
                              width={Math.max(88, arc.labelText.length * 6.2 + 12)}
                              height={16}
                              rx={3}
                              fill="rgba(15, 23, 42, 0.92)"
                              stroke={col}
                              strokeWidth={0.9}
                            />
                            <text
                              x={arc.labelX}
                              y={arc.labelY + 1.5}
                              textAnchor="middle"
                              fill="#f1f5f9"
                              fontFamily="var(--font-mono)"
                              fontSize={9.5}
                              fontWeight={600}
                            >
                              {arc.labelText}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}

                {/* OULU Neutron Monitor */}
                {showOulu && liveScene.ouluPos && (
                  <g
                    onMouseEnter={() => {
                      setHoveredOulu(true);
                      setHoveredStation(null);
                    }}
                    onMouseLeave={() => setHoveredOulu(false)}
                    data-testid="station-marker-OULU"
                  >
                    <polygon
                      points={`${liveScene.ouluPos[0]},${liveScene.ouluPos[1] - 6} ${liveScene.ouluPos[0] + 6},${liveScene.ouluPos[1]} ${liveScene.ouluPos[0]},${liveScene.ouluPos[1] + 6} ${liveScene.ouluPos[0] - 6},${liveScene.ouluPos[1]}`}
                      fill="#2dd4bf"
                      stroke="#0f172a"
                      strokeWidth={1.1}
                    />
                  </g>
                )}

                {/* 49 Magnetometer stations & Stokes ellipses */}
                {liveScene.stationNodes.map((n) => {
                  const g = n.geometry;
                  const senseCol =
                    g.sense === "CW"
                      ? "#f43f5e"
                      : g.sense === "CCW"
                        ? "#38bdf8"
                        : "#a78bfa";
                  const senseFill =
                    g.sense === "CW"
                      ? "rgba(244, 63, 94, 0.24)"
                      : g.sense === "CCW"
                        ? "rgba(56, 189, 248, 0.24)"
                        : "rgba(167, 139, 250, 0.24)";

                  return (
                    <g
                      key={n.meta.iaga_code}
                      data-testid={`station-node-${n.meta.iaga_code}`}
                      data-status={g.status}
                      data-sense={g.sense}
                      style={{ cursor: "pointer" }}
                      onClick={(e) => handleStationClick(n.meta.iaga_code, e)}
                      onMouseEnter={() => {
                        setHoveredStation(n.meta.iaga_code);
                        setHoveredOulu(false);
                        setHoveredArcKey(null);
                      }}
                      onMouseLeave={() => setHoveredStation(null)}
                    >
                      <circle cx={n.x} cy={n.y} r={11} fill="transparent" />

                      {n.isPendingPick && (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={Math.max(12, g.majorRadius + 6)}
                          fill="none"
                          stroke="#2dd4bf"
                          strokeWidth={2}
                          strokeDasharray="3,2"
                        />
                      )}
                      {n.isStationA && (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={Math.max(10, g.majorRadius + 4)}
                          fill="none"
                          stroke="#38bdf8"
                          strokeWidth={1.8}
                        />
                      )}
                      {n.isStationB && (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={Math.max(10, g.majorRadius + 4)}
                          fill="none"
                          stroke="#fbbf24"
                          strokeWidth={1.8}
                        />
                      )}
                      {n.inFlrChain && !n.isStationA && !n.isStationB && (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={Math.max(8, g.majorRadius + 3)}
                          fill="none"
                          stroke="#a78bfa"
                          strokeWidth={1.2}
                          strokeDasharray="3,2"
                        />
                      )}

                      {showEllipses && g.status === "significant" ? (
                        <g transform={`translate(${n.x}, ${n.y}) rotate(${g.rotationDeg})`}>
                          <ellipse
                            cx={0}
                            cy={0}
                            rx={Math.max(0.8, g.minorRadius)}
                            ry={g.majorRadius}
                            fill={senseFill}
                            stroke={senseCol}
                            strokeWidth={1.7}
                          />
                          <line
                            x1={0}
                            y1={-g.majorRadius}
                            x2={0}
                            y2={g.majorRadius}
                            stroke={senseCol}
                            strokeWidth={0.9}
                            opacity={0.65}
                          />
                          {g.showArrow && (
                            <>
                              <polygon points={g.topArrowPoints} fill={senseCol} />
                              <polygon points={g.bottomArrowPoints} fill={senseCol} />
                            </>
                          )}
                          <circle cx={0} cy={0} r={1.8} fill={senseCol} />
                        </g>
                      ) : g.status === "unavailable" ? (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={3.8}
                          fill="none"
                          stroke="#64748b"
                          strokeWidth={1.2}
                          strokeDasharray="2,2"
                        />
                      ) : (
                        <circle
                          cx={n.x}
                          cy={n.y}
                          r={4}
                          fill="#64748b"
                          stroke="#0f172a"
                          strokeWidth={0.9}
                          opacity={0.88}
                        />
                      )}
                    </g>
                  );
                })}

                {/* Labels */}
                {liveScene.labels.map((lbl) => (
                  <text
                    key={`lbl-${lbl.code}`}
                    x={lbl.x}
                    y={lbl.y}
                    textAnchor={lbl.anchor}
                    fill="#e2e8f0"
                    fontFamily="var(--font-mono)"
                    fontSize={9.5}
                    fontWeight={600}
                    pointerEvents="none"
                    style={{ textShadow: "0 1px 3px #020617" }}
                  >
                    {lbl.code}
                  </text>
                ))}
              </g>
            </svg>

            {/* Hover Tooltip */}
            {hoveredStationNode && (
              <div
                className="map-tooltip mono"
                style={{
                  left: Math.min(stageWidth - 260, Math.max(8, mousePos.x + 14)),
                  top: Math.max(8, Math.min(mapHeight - 170, mousePos.y - 12)),
                }}
              >
                <div style={{ fontWeight: 700, color: "var(--accent-primary)" }}>
                  {hoveredStationNode.meta.iaga_code} — {hoveredStationNode.meta.station_name}
                </div>
                <div>
                  {hoveredStationNode.meta.region} · {hoveredStationNode.meta.lat_deg.toFixed(2)}°N,{" "}
                  {hoveredStationNode.meta.lon_deg.toFixed(2)}°E
                </div>
                <div>
                  MLat: {hoveredStationNode.mlatDeg.toFixed(1)}° · MLT:{" "}
                  {formatMltHours(hoveredStationNode.mltHours)} ·{" "}
                  {hoveredStationNode.meta.has_baseline ? "Полное поле" : "Variation (без базы)"}
                </div>
                {state.coordFrame === "XmYm" && hoveredStationNode.meta.has_baseline && (
                  <div>D₀ = {hoveredStationNode.d0Deg.toFixed(2)}°</div>
                )}
                {hoveredStationNode.stokes ? (
                  <div style={{ marginTop: 4, borderTop: "1px solid var(--border-subtle)", paddingTop: 4 }}>
                    <div>
                      I={formatNum(hoveredStationNode.stokes.I, 1)} нТл² · p=
                      {formatNum(hoveredStationNode.stokes.p)} (p_crit=
                      {formatNum(hoveredStationNode.stokes.p_crit)})
                    </div>
                    <div>
                      ψ={formatNum(hoveredStationNode.stokes.psi_deg, 1)}° · χ=
                      {formatNum(hoveredStationNode.stokes.chi_deg, 1)}° · ε=
                      {formatNum(hoveredStationNode.stokes.epsilon)} ({hoveredStationNode.geometry.sense})
                    </div>
                    <div>γ²_xy={formatNum(hoveredStationNode.stokes.gamma2)}</div>
                  </div>
                ) : (
                  <div style={{ marginTop: 4, color: "var(--warn)" }}>
                    {hoveredStationNode.warning ??
                      (hoveredStationNode.available
                        ? "Недостаточно валидных сегментов"
                        : "Нет данных в этом эпизоде")}
                  </div>
                )}
              </div>
            )}

            {hoveredOulu && ouluMeta && (
              <div
                className="map-tooltip mono"
                style={{
                  left: Math.min(stageWidth - 230, Math.max(8, mousePos.x + 14)),
                  top: Math.max(8, mousePos.y - 12),
                }}
              >
                <div style={{ fontWeight: 700, color: "#2dd4bf" }}>
                  OULU — {ouluMeta.station_name} (NM)
                </div>
                <div>
                  {ouluMeta.lat_deg.toFixed(2)}°N, {ouluMeta.lon_deg.toFixed(2)}°E · Rc=
                  {ouluMeta.cutoff_rigidity_gv ?? "0.8"} GV
                </div>
                {ouluValAtCursor !== null && (
                  <div>ΔN/N₀ у курсора: {ouluValAtCursor.toFixed(2)}%</div>
                )}
              </div>
            )}

            {hoveredArcNode && (
              <div
                className="map-tooltip mono"
                style={{
                  left: Math.min(stageWidth - 260, Math.max(8, mousePos.x + 14)),
                  top: Math.max(8, mousePos.y - 12),
                }}
              >
                <div style={{ fontWeight: 700, color: "var(--accent-primary)" }}>
                  Дуга {hoveredArcNode.metrics.stationI} ↔ {hoveredArcNode.metrics.stationJ}
                </div>
                <div>
                  Дистанция: {hoveredArcNode.metrics.distanceKm.toFixed(0)} км · Δλ=
                  {hoveredArcNode.metrics.deltaLonDeg.toFixed(2)}°
                </div>
                <div>
                  γ²={formatNum(hoveredArcNode.metrics.gamma2)} (γ²_crit=
                  {formatNum(hoveredArcNode.metrics.gamma2Crit)})
                </div>
                <div>
                  Δφ=
                  {hoveredArcNode.metrics.phaseDegMasked !== null
                    ? `${hoveredArcNode.metrics.phaseDegMasked.toFixed(1)}°`
                    : "маск."}
                  {" · "}
                  {hoveredArcNode.metrics.isMeridionalGuarded
                    ? "меридиональная защита m (|Δλ|<3°)"
                    : `m=${formatNum(hoveredArcNode.metrics.azimuthalM, 2)} (|m_alias|=${formatNum(hoveredArcNode.metrics.mAlias, 1)})`}
                </div>
              </div>
            )}
          </div>

          <div className="map-legend">
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#f43f5e" }} />
              CW (V &gt; 0, ε &gt; 0)
            </span>
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#38bdf8" }} />
              CCW (V &lt; 0, ε &lt; 0)
            </span>
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#a78bfa" }} />
              Линейная (|ε| &lt; 0.05)
            </span>
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#64748b" }} />
              p ≤ p_crit / нет данных
            </span>
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#fbbf24" }} />
              MLT = 12 (диполь IGRF-14)
            </span>
            <span className="map-legend-item">
              <span className="legend-dot" style={{ background: "#2dd4bf" }} />
              OULU NM
            </span>
            <span className="hint-text" style={{ marginLeft: "auto" }}>
              Клик 1 → Клик 2: выбор пары (Запад→Восток) · Колесо/перетаскивание: зум/панорама
            </span>
          </div>
        </div>
      )}
    </section>
  );
};
