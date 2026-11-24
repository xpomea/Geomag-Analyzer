import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { AnalysisContext } from "./AnalysisContext";

export interface CursorContextValue {
  cursorUtcMs: number | null;
  setCursorUtcMs: (utcMs: number | null) => void;
}

export const CursorContext = createContext<CursorContextValue | null>(null);

export const CursorProvider: React.FC<{
  resetValue: number | null;
  cursorRef?: React.MutableRefObject<number | null>;
  children: React.ReactNode;
}> = ({ resetValue, cursorRef, children }) => {
  const [cursorUtcMs, setCursorState] = useState<number | null>(resetValue);

  useEffect(() => {
    setCursorState(resetValue);
    if (cursorRef) {
      cursorRef.current = resetValue;
    }
  }, [resetValue, cursorRef]);

  const setCursorUtcMs = useCallback(
    (utcMs: number | null) => {
      if (cursorRef) {
        cursorRef.current = utcMs;
      }
      setCursorState(utcMs);
    },
    [cursorRef],
  );

  const value = useMemo(
    () => ({ cursorUtcMs, setCursorUtcMs }),
    [cursorUtcMs, setCursorUtcMs],
  );

  return <CursorContext.Provider value={value}>{children}</CursorContext.Provider>;
};

export function useCursor(): CursorContextValue {
  const cursorCtx = useContext(CursorContext);
  const analysisCtx = useContext(AnalysisContext);
  if (cursorCtx) {
    return cursorCtx;
  }
  if (analysisCtx) {
    return {
      cursorUtcMs: analysisCtx.state.cursorUtcMs,
      setCursorUtcMs: (utcMs: number | null) =>
        analysisCtx.dispatch({ type: "SET_CURSOR_UTC", utcMs }),
    };
  }
  throw new Error("useCursor must be used inside CursorProvider or AnalysisProvider");
}
