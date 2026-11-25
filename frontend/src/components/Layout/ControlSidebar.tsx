import React, { useMemo } from "react";
import {
  AlertTriangle,
  ChevronRight,
  Compass,
  Info,
  Layers,
  Sliders,
  Sun,
} from "lucide-react";
import type {
  BandType,
  CoordFrame,
  OversampleMethod,
  PolarizationMethod,
  Solar1ShiftMode,
  SpectrumType,
} from "../../api/types";
import { useAnalysis } from "../../store/AnalysisContext";
import { DraftNullableNumberInput, DraftNumberInput } from "./DraftNumberInput";

const QUICK_PAIRS: { label: string; a: string; b: string }[] = [
  { label: "LVV–BEL", a: "LVV", b: "BEL" },
  { label: "HRN–LVV", a: "HRN", b: "LVV" },
  { label: "LVV–OTT", a: "LVV", b: "OTT" },
];

const METHOD_SHORT_LABELS: Record<PolarizationMethod, string> = {
  welch: "Welch",
  multitaper: "DPSS",
  variant_b_fft: "FFT+Hilbert",
  variant_b_sos: "Butterworth",
};

export const ControlSidebar: React.FC = React.memo(() => {
  const {
    state,
    dispatch,
    catalog,
    activeEpisode,
    stationMetaMap,
    stationAvailabilityMap,
    shiftedSolar1,
    baselineWarning,
  } = useAnalysis();

  const magnetometers = useMemo(() => {
    if (!catalog) return [];
    return catalog.stations.filter((s) => s.station_type === "magnetometer");
  }, [catalog]);

  const europeMags = useMemo(
    () => magnetometers.filter((s) => s.region.toLowerCase() === "europe"),
    [magnetometers],
  );
  const canadaMags = useMemo(
    () => magnetometers.filter((s) => s.region.toLowerCase() === "canada"),
    [magnetometers],
  );

  const renderStationOption = (code: string, name: string, hasBaseline: boolean) => {
    const avail = stationAvailabilityMap.get(code);
    const availText = avail ? `${avail.days}d (${avail.qualities.join("/")})` : "absent";
    const baseTag = hasBaseline ? "" : " [no baseline]";
    return (
      <option key={code} value={code}>
        {code} — {name} · {availText}
        {baseTag}
      </option>
    );
  };

  const stationAMeta = stationMetaMap.get(state.stationA);
  const stationBMeta = stationMetaMap.get(state.stationB);
  const isVariantB = state.method.startsWith("variant_b");

  return (
    <aside className="sidebar">
      <section className="sidebar-section">
        <div className="sidebar-heading">
          <span>Episode &amp; Stations</span>
          <Compass size={13} />
        </div>

        <label className="field-label">
          <span>Episode</span>
          <select
            value={state.episodeId}
            onChange={(e) => {
              const ep = catalog?.episodes.find((x) => x.episode_id === e.target.value) ?? null;
              dispatch({ type: "SET_EPISODE", episodeId: e.target.value, episode: ep });
            }}
          >
            {(catalog?.episodes ?? []).map((ep) => (
              <option key={ep.episode_id} value={ep.episode_id}>
                {ep.label} · {ep.n_days}d {ep.omni_available ? "[OMNI]" : "[no OMNI]"}
              </option>
            ))}
          </select>
        </label>

        {activeEpisode && !activeEpisode.omni_available && (
          <div className="hint-text" style={{ color: "var(--warn)" }}>
            OMNI data unavailable for this episode.
          </div>
        )}

        <label className="field-label">
          <span>Station A (Primary)</span>
          <select
            value={state.stationA}
            onChange={(e) => dispatch({ type: "SET_STATION_A", stationIaga: e.target.value })}
          >
            <optgroup label="Europe">
              {europeMags.map((s) => renderStationOption(s.iaga_code, s.station_name, s.has_baseline))}
            </optgroup>
            <optgroup label="Canada">
              {canadaMags.map((s) => renderStationOption(s.iaga_code, s.station_name, s.has_baseline))}
            </optgroup>
          </select>
        </label>

        <label className="field-label">
          <span>Station B (Cross-Spectral)</span>
          <select
            value={state.stationB}
            onChange={(e) => dispatch({ type: "SET_STATION_B", stationIaga: e.target.value })}
          >
            <optgroup label="Europe">
              {europeMags.map((s) => renderStationOption(s.iaga_code, s.station_name, s.has_baseline))}
            </optgroup>
            <optgroup label="Canada">
              {canadaMags.map((s) => renderStationOption(s.iaga_code, s.station_name, s.has_baseline))}
            </optgroup>
          </select>
        </label>

        <div className="quick-chips">
          {QUICK_PAIRS.map((p) => (
            <button
              key={p.label}
              type="button"
              className={`chip-btn ${state.stationA === p.a && state.stationB === p.b ? "active" : ""}`}
              onClick={() => dispatch({ type: "SET_PAIR", stationA: p.a, stationB: p.b })}
            >
              {p.label}
            </button>
          ))}
        </div>

        {(stationAMeta && !stationAMeta.has_baseline) || (stationBMeta && !stationBMeta.has_baseline) ? (
          <div className="hint-text" style={{ display: "flex", gap: 5, alignItems: "flex-start" }}>
            <AlertTriangle size={13} color="var(--warn)" style={{ flexShrink: 0, marginTop: 2 }} />
            <span>
              No absolute baseline: enter D0 manually for XmYm.
            </span>
          </div>
        ) : null}
      </section>

      <section className="sidebar-section">
        <div className="sidebar-heading">
          <span className="sidebar-heading-left">
            <span>Band &amp; Coordinates</span>
            <span
              className="info-tip-badge"
              title="INTERMAGNET 1-min Gaussian filter: |H| = 0.805 at T = 150 s."
            >
              <Info size={12} />
            </span>
          </span>
          <Layers size={13} />
        </div>

        <div className="segmented">
          {(["Pc5", "Pc6"] as BandType[]).map((b) => (
            <button
              key={b}
              type="button"
              className={state.band === b ? "active" : ""}
              onClick={() => dispatch({ type: "SET_BAND", band: b })}
            >
              {b === "Pc5" ? "Pc5 (150–600 s)" : `Pc6 (600–${state.tMaxPc6Sec} s)`}
            </button>
          ))}
        </div>

        {state.band === "Pc6" && (
          <label className="field-label">
            <span>T_max for Pc6 [s]</span>
            <select
              value={state.tMaxPc6Sec}
              onChange={(e) => dispatch({ type: "SET_T_MAX_PC6", sec: Number(e.target.value) })}
            >
              <option value={900}>900 s (15 min)</option>
              <option value={1800}>1800 s (30 min)</option>
              <option value={2400}>2400 s (40 min)</option>
              <option value={3600}>3600 s (60 min)</option>
            </select>
          </label>
        )}

        <div className="field-label">
          <span>Coordinate Frame</span>
          <div className="segmented">
            {(["XYZ", "XmYm"] as CoordFrame[]).map((cf) => (
              <button
                key={cf}
                type="button"
                className={state.coordFrame === cf ? "active" : ""}
                onClick={() => dispatch({ type: "SET_COORD_FRAME", coordFrame: cf })}
              >
                {cf}
              </button>
            ))}
          </div>
        </div>

        {state.coordFrame === "XmYm" && (
          <div className="field-row">
            <label className="field-label">
              <span>D0 St. A [°]</span>
              <DraftNullableNumberInput
                min={-180}
                max={180}
                step={0.1}
                placeholder="Auto"
                value={state.externalD0Deg}
                ariaLabel="D0 Station A"
                onCommit={(val) => dispatch({ type: "SET_EXTERNAL_D0", d0Deg: val })}
              />
            </label>
            <label className="field-label">
              <span>D0 St. B [°]</span>
              <DraftNullableNumberInput
                min={-180}
                max={180}
                step={0.1}
                placeholder="Auto"
                value={state.externalD0DegB}
                ariaLabel="D0 Station B"
                onCommit={(val) => dispatch({ type: "SET_EXTERNAL_D0_B", d0Deg: val })}
              />
            </label>
          </div>
        )}

        {baselineWarning && (
          <div className="hint-text" style={{ color: "var(--danger)" }}>
            {baselineWarning}
          </div>
        )}

        {state.band === "Pc6" && state.L * 60 < 4 * state.tMaxPc6Sec && (
          <div
            className="hint-text"
            style={{
              color: "var(--warn)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 6,
            }}
          >
            <span>
              L·Δt ({state.L * 60} s) &lt; 4·T_max ({4 * state.tMaxPc6Sec} s)
              {state.method !== "welch" ? " (Welch / spec.)" : ""}
            </span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => dispatch({ type: "SET_WELCH_PARAMS", L: 256 })}
            >
              L = 256
            </button>
          </div>
        )}

        {state.method === "variant_b_sos" && state.band === "Pc5" && state.oversampleM < 4 && (
          <div
            className="hint-text"
            style={{
              color: "var(--warn)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 6,
            }}
          >
            <span>Pc5 Butterworth SOS requires m ≥ 4</span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => dispatch({ type: "SET_VARIANT_B_PARAMS", oversampleM: 4 })}
            >
              m = 4
            </button>
          </div>
        )}
      </section>

      <section className="sidebar-section">
        <div className="sidebar-heading">
          <span className="sidebar-heading-left">
            <span>
              Significance <span className="keep-case">α</span>
            </span>
            <span
              className="info-tip-badge"
              title="Statistical significance level for p_crit, γ²_crit, and Monte Carlo contours. Masks non-significant map ellipses, coherence arcs, Stokes angles (ψ, χ, ε), and cross-spectral phases (Δφ, m)."
            >
              <Info size={12} />
            </span>
          </span>
          <Sliders size={13} />
        </div>
        <div className="quick-chips">
          {[0.05, 0.01, 0.1].map((a) => (
            <button
              key={a}
              type="button"
              className={`chip-btn ${Math.abs(state.alpha - a) < 1e-6 ? "active" : ""}`}
              onClick={() => dispatch({ type: "SET_ALPHA", alpha: a })}
            >
              α = {a === 0.1 ? "0.10" : a} ({Math.round((1 - a) * 100)}%)
            </button>
          ))}
        </div>
      </section>

      <section className="sidebar-section">
        <div className="sidebar-heading">
          <span className="sidebar-heading-left">
            <span>SOLAR-1 Shift (L1 → BSN)</span>
          </span>
          <Sun size={13} />
        </div>
        <div className="segmented">
          {(
            [
              { id: "ballistic_auto", label: "Auto Vx" },
              { id: "l1_raw", label: "L1 Raw" },
              { id: "manual", label: "Manual" },
            ] as { id: Solar1ShiftMode; label: string }[]
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              className={state.solar1ShiftMode === item.id ? "active" : ""}
              onClick={() => dispatch({ type: "SET_SOLAR1_SHIFT", mode: item.id })}
            >
              {item.label}
            </button>
          ))}
        </div>

        {state.solar1ShiftMode === "manual" && (
          <label className="field-label">
            <span>Lag [min]: {state.manualLagMin}</span>
            <input
              type="range"
              min={0}
              max={120}
              step={1}
              value={state.manualLagMin}
              onChange={(e) =>
                dispatch({ type: "SET_MANUAL_LAG", lagMin: Number(e.target.value) })
              }
            />
          </label>
        )}

        {shiftedSolar1 && (
          <div className="hint-text mono">
            {state.solar1ShiftMode === "l1_raw"
              ? "No shift (L1 time)"
              : `Delay: ${shiftedSolar1.meanDelayMin.toFixed(1)} min [${shiftedSolar1.minDelayMin.toFixed(0)}..${shiftedSolar1.maxDelayMin.toFixed(0)}]`}
          </div>
        )}
      </section>

      <details className="sidebar-section sidebar-details">
        <summary className="sidebar-heading">
          <span className="sidebar-heading-left" style={{ whiteSpace: "nowrap" }}>
            <ChevronRight size={13} className="sidebar-details-chevron" />
            <span>Method &amp; Spectral</span>
          </span>
          <span
            className="mono keep-case"
            style={{ fontSize: 10, color: "var(--accent-primary)", whiteSpace: "nowrap" }}
          >
            {METHOD_SHORT_LABELS[state.method]} · {state.windowSamples}/{state.stepSamples}m
          </span>
        </summary>

        <div className="sidebar-details-body">
          <label className="field-label">
            <span>Estimator</span>
            <select
              value={state.method}
              onChange={(e) =>
                dispatch({ type: "SET_METHOD", method: e.target.value as PolarizationMethod })
              }
            >
              <option value="welch">Welch (Hann)</option>
              <option value="multitaper">Multitaper (DPSS)</option>
              <option value="variant_b_fft">FFT Bandpass + Hilbert</option>
              <option value="variant_b_sos">Butterworth SOS + Hilbert</option>
            </select>
          </label>

          <label className="field-label">
            <span>Background Spectrum</span>
            <select
              value={state.spectrumType}
              onChange={(e) =>
                dispatch({ type: "SET_SPECTRUM_TYPE", spectrumType: e.target.value as SpectrumType })
              }
            >
              <option value="flat">Flat (white)</option>
              <option value="red">Red (f⁻²)</option>
            </select>
          </label>

          <div className="field-row">
            <label className="field-label">
              <span>Window [min]</span>
              <DraftNumberInput
                min={Math.max(60, state.L)}
                max={1440}
                step={30}
                integer
                value={state.windowSamples}
                onCommit={(val) =>
                  dispatch({
                    type: "SET_WELCH_PARAMS",
                    windowSamples: val,
                  })
                }
              />
            </label>
            <label className="field-label">
              <span>Step [min]</span>
              <DraftNumberInput
                min={10}
                max={240}
                step={10}
                integer
                value={state.stepSamples}
                onCommit={(val) =>
                  dispatch({
                    type: "SET_WELCH_PARAMS",
                    stepSamples: val,
                  })
                }
              />
            </label>
          </div>

          {state.method === "welch" && (
            <div className="field-row">
              <label className="field-label">
                <span>Segment L</span>
                <select
                  value={state.L}
                  onChange={(e) =>
                    dispatch({ type: "SET_WELCH_PARAMS", L: Number(e.target.value) })
                  }
                >
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                </select>
              </label>
              <label className="field-label">
                <span>Overlap Shift M</span>
                <select
                  value={state.M}
                  onChange={(e) =>
                    dispatch({ type: "SET_WELCH_PARAMS", M: Number(e.target.value) })
                  }
                >
                  <option value={64}>64</option>
                  {state.L > 128 && <option value={128}>128</option>}
                </select>
              </label>
            </div>
          )}

          {state.method === "multitaper" && (
            <label className="field-label">
              <span>Time-Bandwidth NW</span>
              <DraftNumberInput
                min={1.5}
                max={6.0}
                step={0.5}
                value={state.nw}
                onCommit={(val) => dispatch({ type: "SET_NW", nw: val })}
              />
            </label>
          )}

          {isVariantB && (
            <>
              <div className="field-row">
                <label className="field-label">
                  <span>Oversample m</span>
                  <select
                    value={state.oversampleM}
                    onChange={(e) =>
                      dispatch({
                        type: "SET_VARIANT_B_PARAMS",
                        oversampleM: Number(e.target.value),
                      })
                    }
                  >
                    <option value={1}>m = 1</option>
                    <option value={4}>m = 4</option>
                    <option value={8}>m = 8</option>
                  </select>
                </label>
                <label className="field-label">
                  <span>OS Method</span>
                  <select
                    value={state.oversampleMethod}
                    onChange={(e) =>
                      dispatch({
                        type: "SET_VARIANT_B_PARAMS",
                        oversampleMethod: e.target.value as OversampleMethod,
                      })
                    }
                  >
                    <option value="fourier">Fourier</option>
                    <option value="kaiser">Kaiser FIR</option>
                  </select>
                </label>
              </div>

              {state.method === "variant_b_sos" && (
                <label className="field-label">
                  <span>Butterworth Order n</span>
                  <select
                    value={state.orderN}
                    onChange={(e) =>
                      dispatch({
                        type: "SET_VARIANT_B_PARAMS",
                        orderN: Number(e.target.value),
                      })
                    }
                  >
                    <option value={2}>n = 2</option>
                    <option value={3}>n = 3</option>
                    <option value={4}>n = 4</option>
                  </select>
                </label>
              )}
            </>
          )}
        </div>
      </details>
    </aside>
  );
});
