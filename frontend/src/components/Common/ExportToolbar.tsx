import React, { useState } from "react";
import { Download, FileSpreadsheet, Image as ImageIcon, Moon, Sun } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";
import {
  downloadBlob,
  downloadTextFile,
  rasterizeSvgToPng300Dpi,
} from "../../utils/exportUtils";

export interface ExportToolbarProps {
  baseFilename: string;
  getSvg?: () => string | null;
  svgWidth?: number;
  svgHeight?: number;
  getCsv?: () => string | null;
  disabled?: boolean;
  compact?: boolean;
}

export const ExportToolbar: React.FC<ExportToolbarProps> = ({
  baseFilename,
  getSvg,
  svgWidth = 900,
  svgHeight = 400,
  getCsv,
  disabled = false,
  compact = true,
}) => {
  const { state, dispatch } = useAnalysis();
  const [busy, setBusy] = useState<"svg" | "png" | "csv" | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  const handleSvg = () => {
    if (!getSvg) return;
    setErrMsg(null);
    setBusy("svg");
    try {
      const svg = getSvg();
      if (!svg) return;
      downloadTextFile(`${baseFilename}.svg`, svg, "image/svg+xml;charset=utf-8");
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Ошибка SVG экспорта");
    } finally {
      setBusy(null);
    }
  };

  const handlePng = async () => {
    if (!getSvg) return;
    setErrMsg(null);
    setBusy("png");
    try {
      const svg = getSvg();
      if (!svg) return;
      const blob = await rasterizeSvgToPng300Dpi(svg, svgWidth, svgHeight, 300);
      downloadBlob(`${baseFilename}_300dpi.png`, blob);
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Ошибка PNG экспорта");
    } finally {
      setBusy(null);
    }
  };

  const handleCsv = () => {
    if (!getCsv) return;
    setErrMsg(null);
    setBusy("csv");
    try {
      const csv = getCsv();
      if (!csv) return;
      downloadTextFile(`${baseFilename}.csv`, csv, "text/csv;charset=utf-8");
    } catch (err) {
      setErrMsg(err instanceof Error ? err.message : "Ошибка CSV экспорта");
    } finally {
      setBusy(null);
    }
  };

  const isLight = state.exportTheme === "light";

  return (
    <div className="export-toolbar" title={errMsg ?? undefined}>
      {getSvg && (
        <button
          type="button"
          className="btn btn-sm"
          disabled={disabled}
          onClick={() =>
            dispatch({
              type: "SET_EXPORT_THEME",
              theme: isLight ? "dark" : "light",
            })
          }
          title={
            isLight
              ? "Тема экспорта: Светлая для диссертации/печати (нажмите для Тёмной)"
              : "Тема экспорта: Тёмная экранная (нажмите для Светлой)"
          }
        >
          {isLight ? <Sun size={12} color="#f59e0b" /> : <Moon size={12} />}
          {!compact && <span>{isLight ? "Печать" : "Экран"}</span>}
        </button>
      )}

      {getSvg && (
        <button
          type="button"
          className="btn btn-sm"
          disabled={disabled || busy !== null}
          onClick={handleSvg}
          title="Экспорт векторной графики SVG (для диссертации / Word / Inkscape)"
        >
          <Download size={12} />
          <span>SVG</span>
        </button>
      )}

      {getSvg && (
        <button
          type="button"
          className="btn btn-sm"
          disabled={disabled || busy !== null}
          onClick={() => void handlePng()}
          title="Экспорт растрового рисунка PNG 300 DPI"
        >
          <ImageIcon size={12} />
          <span>{busy === "png" ? "…" : "PNG 300"}</span>
        </button>
      )}

      {getCsv && (
        <button
          type="button"
          className="btn btn-sm"
          disabled={disabled || busy !== null}
          onClick={handleCsv}
          title="Экспорт числовых данных и метаданных расчёта в CSV"
        >
          <FileSpreadsheet size={12} />
          <span>CSV</span>
        </button>
      )}
    </div>
  );
};
