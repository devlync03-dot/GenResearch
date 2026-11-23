"use client";

import { useState, useEffect } from "react";
import { supabase } from "../lib/supabase";
import Home from "./Components/Home";
import Dashboard from "./Components/Dashboard";
import Login from "./Components/Login";
import Register from "./Components/Register";

type View = "home" | "login" | "register" | "dashboard" | "loading";
const SESSION_LOAD_TIMEOUT_MS = 8_000;

export default function Page() {
  const [view, setView] = useState<View>("loading");

  useEffect(() => {
    let mounted = true;
    let authStateChanged = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (_event === "INITIAL_SESSION") return;
      authStateChanged = true;
      if (!mounted) return;

      if (session) {
        setView("dashboard");
      } else {
        setView("home");
      }
    });

    const checkSession = async () => {
      try {
        const result = await Promise.race([
          supabase.auth.getSession(),
          new Promise<null>((resolve) => {
            timeoutId = setTimeout(() => resolve(null), SESSION_LOAD_TIMEOUT_MS);
          }),
        ]);

        if (!mounted || authStateChanged) return;
        if (result === null) {
          console.error("Timed out while restoring the Supabase session.");
          setView("home");
          return;
        }
        if (result.error) throw result.error;
        setView(result.data.session ? "dashboard" : "home");
      } catch (error) {
        if (!mounted || authStateChanged) return;
        console.error("Failed to restore the Supabase session:", error);
        setView("home");
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
      }
    };

    void checkSession();

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  if (view === "loading") {
    return (
      <div style={{ 
        height: "100vh", 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "center",
        background: "#f5f0e8",
        fontFamily: "'Crimson Pro', Georgia, serif"
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ 
            width: 40, 
            height: 40, 
            border: "3px solid rgba(139,105,20,0.1)", 
            borderTop: "3px solid #8b6914", 
            borderRadius: "50%", 
            animation: "spin 1s linear infinite",
            margin: "0 auto 16px" 
          }} />
          <div style={{ color: "#8b6914", fontSize: 14, letterSpacing: "0.1em", textTransform: "uppercase" }}>Loading GenResearch...</div>
        </div>
        <style>{`
          @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        `}</style>
      </div>
    );
  }

  return (
    <>
      {view === "home" && (
        <Home
          onNavigate={() => setView("register")}
          onNavigateLogin={() => setView("login")}
        />
      )}
      {view === "login" && (
        <Login
          onLoginSuccess={() => setView("dashboard")}
          onNavigateRegister={() => setView("register")}
          onNavigateHome={() => setView("home")}
        />
      )}
      {view === "register" && (
        <Register
          onRegisterSuccess={() => setView("login")}
          onNavigateLogin={() => setView("login")}
          onNavigateHome={() => setView("home")}
        />
      )}
      {view === "dashboard" && (
        <Dashboard onNavigateHome={() => setView("home")} />
      )}
    </>
  );
}
