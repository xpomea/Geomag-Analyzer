import React, { useEffect, useState } from "react";

export interface DraftNumberInputProps {
  value: number;
  onCommit: (nextVal: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  ariaLabel?: string;
}

export const DraftNumberInput: React.FC<DraftNumberInputProps> = ({
  value,
  onCommit,
  min,
  max,
  step,
  integer = false,
  ariaLabel,
}) => {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = () => {
    const parsed = Number(draft.trim());
    if (!Number.isFinite(parsed)) {
      setDraft(String(value));
      return;
    }
    let clamped = integer ? Math.round(parsed) : parsed;
    if (min !== undefined) clamped = Math.max(min, clamped);
    if (max !== undefined) clamped = Math.min(max, clamped);
    setDraft(String(clamped));
    if (clamped !== value) {
      onCommit(clamped);
    }
  };

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft}
      aria-label={ariaLabel}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
        }
      }}
    />
  );
};

export interface DraftNullableNumberInputProps {
  value: number | null;
  onCommit: (nextVal: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  ariaLabel?: string;
}

export const DraftNullableNumberInput: React.FC<DraftNullableNumberInputProps> = ({
  value,
  onCommit,
  min,
  max,
  step,
  placeholder,
  ariaLabel,
}) => {
  const [draft, setDraft] = useState(value === null ? "" : String(value));

  useEffect(() => {
    setDraft(value === null ? "" : String(value));
  }, [value]);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed === "") {
      setDraft("");
      if (value !== null) {
        onCommit(null);
      }
      return;
    }
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) {
      setDraft(value === null ? "" : String(value));
      return;
    }
    let clamped = parsed;
    if (min !== undefined) clamped = Math.max(min, clamped);
    if (max !== undefined) clamped = Math.min(max, clamped);
    setDraft(String(clamped));
    if (clamped !== value) {
      onCommit(clamped);
    }
  };

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
        }
      }}
    />
  );
};
