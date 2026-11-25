import React, { useState } from "react";
import { KeyRound, ShieldAlert } from "lucide-react";
import { useAnalysis } from "../../store/AnalysisContext";

export const AuthGate: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { auth, authenticateWithToken } = useAnalysis();
  const [tokenInput, setTokenInput] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!auth.checked) {
    return (
      <div className="modal-backdrop">
        <div className="modal-card">
          <div style={{ fontWeight: 600 }}>Starting GeoMag Analyzer…</div>
          <div className="hint-text">Loading workspace…</div>
        </div>
      </div>
    );
  }

  if (auth.authenticated || !auth.requireAuth) {
    return <>{children}</>;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tokenInput.trim()) return;
    setSubmitting(true);
    try {
      await authenticateWithToken(tokenInput);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <form className="modal-card" onSubmit={handleSubmit}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 15 }}>
          <KeyRound size={18} color="var(--accent-primary)" />
          <span>Session Authentication</span>
        </div>
        <div className="hint-text">
          Enter the access token provided on launch.
        </div>
        {auth.error && (
          <div className="banner banner-danger">
            <ShieldAlert size={14} />
            <span>{auth.error}</span>
          </div>
        )}
        <label className="field-label">
          <span>Access Token</span>
          <input
            type="password"
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            placeholder="Paste access token…"
            autoFocus
          />
        </label>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button type="submit" className="btn btn-primary" disabled={submitting || !tokenInput.trim()}>
            {submitting ? "Verifying…" : "Connect"}
          </button>
        </div>
      </form>
    </div>
  );
};
