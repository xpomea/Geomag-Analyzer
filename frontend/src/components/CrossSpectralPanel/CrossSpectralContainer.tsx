import React from "react";
import { RadioReceiver } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import { SpectrogramPanel } from "./SpectrogramPanel";
import { WelchCrossPanel } from "./WelchCrossPanel";
import { WtcPanel } from "./WtcPanel";

export const CrossSpectralContainer: React.FC = () => {
  const { state, dispatch } = useAnalysis();

  return (
    <section className="panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <RadioReceiver size={15} color="#2dd4bf" />
          <span>Dynamic Spectrogram &amp; Cross-Spectral Analysis</span>
        </div>
      </div>

      <div className="tabs-bar">
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "spectrogram" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "spectrogram" })}
        >
          Spectrogram {state.stationA}
        </button>
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "welch" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "welch" })}
        >
          Cross-Welch γ² / Δφ / m / FLR ({state.stationA}↔{state.stationB})
        </button>
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "wtc" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "wtc" })}
        >
          Wavelet XWT / WTC + Monte Carlo
        </button>
      </div>

      <div className="panel-body">
        {state.activeCrossTab === "spectrogram" && <SpectrogramPanel />}
        {state.activeCrossTab === "welch" && <WelchCrossPanel />}
        {state.activeCrossTab === "wtc" && <WtcPanel />}
      </div>
    </section>
  );
};
