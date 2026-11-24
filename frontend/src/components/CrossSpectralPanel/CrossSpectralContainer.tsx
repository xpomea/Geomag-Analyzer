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
          <span>4. Динамическая спектрограмма и кросс-спектральный анализ (Tier 2 / Tier 3)</span>
        </div>
      </div>

      <div className="tabs-bar">
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "spectrogram" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "spectrogram" })}
        >
          Спектрограмма {state.stationA}
        </button>
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "welch" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "welch" })}
        >
          Кросс-Уэлч γ² / Δφ / m / FLR ({state.stationA}↔{state.stationB})
        </button>
        <button
          type="button"
          className={`tab-btn ${state.activeCrossTab === "wtc" ? "active" : ""}`}
          onClick={() => dispatch({ type: "SET_ACTIVE_CROSS_TAB", tab: "wtc" })}
        >
          Вейвлет XWT / WTC + Монте-Карло (Tier 3)
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
