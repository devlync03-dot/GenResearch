"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { supabase } from "../../lib/supabase";

export default function ResetPasswordPage() {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (active && event === "PASSWORD_RECOVERY" && session) {
        setReady(true);
        setLoading(false);
      }
    });

    void supabase.auth.getSession().then(({ data: { session }, error: sessionError }) => {
      if (!active) return;
      if (sessionError) setError(sessionError.message);
      else if (session) setReady(true);
      else setError("This reset link is invalid or has expired. Request a new one from the sign-in page.");
      setLoading(false);
    });

    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, []);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setNotice("");
    if (password.length < 8) {
      setError("Your password must be at least 8 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("The passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const { error: updateError } = await supabase.auth.updateUser({ password });
      if (updateError) setError(updateError.message);
      else setNotice("Your password has been updated. You can now return to GenResearch.");
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Unable to update your password. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{
      minHeight: "100vh",
      display: "grid",
      placeItems: "center",
      padding: 24,
      background: "#f5f0e8",
      color: "#2c1f0e",
      fontFamily: "'Crimson Pro', Georgia, serif",
    }}>
      <section style={{
        width: "100%",
        maxWidth: 420,
        padding: 36,
        background: "#faf8f2",
        border: "1px solid rgba(180,160,120,0.22)",
        borderRadius: 4,
      }}>
        <p style={{
          color: "#8b6914",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
        }}>GenResearch</p>
        <h1 style={{
          margin: "12px 0 24px",
          fontFamily: "'Playfair Display', Georgia, serif",
          fontSize: 30,
        }}>Reset your password</h1>

        {loading && <p role="status">Verifying your reset link…</p>}
        {!loading && error && <p role="alert" style={{ color: "#a0522d", marginBottom: 20 }}>{error}</p>}
        {!loading && notice && <p role="status" style={{ color: "#5a8a3c", marginBottom: 20 }}>{notice}</p>}

        {!loading && ready && !notice && (
          <form onSubmit={handleSubmit}>
            <label htmlFor="new-password" style={{ display: "block", marginBottom: 6 }}>New password</label>
            <input
              id="new-password"
              type="password"
              autoComplete="new-password"
              minLength={8}
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              style={{ width: "100%", padding: 12, marginBottom: 16 }}
            />
            <label htmlFor="confirm-password" style={{ display: "block", marginBottom: 6 }}>Confirm password</label>
            <input
              id="confirm-password"
              type="password"
              autoComplete="new-password"
              minLength={8}
              required
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              style={{ width: "100%", padding: 12, marginBottom: 20 }}
            />
            <button type="submit" disabled={loading} style={{
              width: "100%",
              padding: 14,
              border: 0,
              borderRadius: 3,
              background: "#2c1f0e",
              color: "#f5f0e8",
              cursor: "pointer",
            }}>Update password</button>
          </form>
        )}

        {!loading && !ready && (
          <Link href="/" style={{ color: "#8b6914" }}>Return to sign in</Link>
        )}
        {notice && <Link href="/" style={{ color: "#8b6914" }}>Return to GenResearch</Link>}
      </section>
    </main>
  );
}
