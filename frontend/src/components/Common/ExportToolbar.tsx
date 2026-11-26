import React, { useState } from "react";
import { Download, FileSpreadsheet, Image as ImageIcon } from "lucide-react";
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
}

export const ExportToolbar: React.FC<ExportToolbarProps> = ({
  baseFilename,
  getSvg,
  svgWidth = 900,
  svgHeight = 400,
  getCsv,
  disabled = false,
}) => {
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
      setErrMsg(err instanceof Error ? err.message : "SVG export failed");
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
      setErrMsg(err instanceof Error ? err.message : "PNG export failed");
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
      setErrMsg(err instanceof Error ? err.message : "CSV export failed");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="export-toolbar" title={errMsg ?? undefined}>
      {getSvg && (
        <button
          type="button"
          className="btn btn-sm"
          disabled={disabled || busy !== null}
          onClick={handleSvg}
          title="Export vector SVG"
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
          title="Export PNG (300 DPI)"
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
          title="Export CSV"
        >
          <FileSpreadsheet size={12} />
          <span>CSV</span>
        </button>
      )}
    </div>
  );
};
