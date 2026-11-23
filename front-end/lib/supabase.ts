import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!;

const rememberSessionKey = "genresearch-remember-session";

function getSessionStorage() {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(rememberSessionKey) === "true"
    ? window.sessionStorage
    : window.localStorage;
}

export function setRememberSession(remember: boolean) {
  if (typeof window === "undefined") return;

  if (remember) {
    window.sessionStorage.removeItem(rememberSessionKey);
  } else {
    window.sessionStorage.setItem(rememberSessionKey, "true");
  }
}

export const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    storage: {
      getItem: (key) => getSessionStorage()?.getItem(key) ?? null,
      setItem: (key, value) => getSessionStorage()?.setItem(key, value),
      removeItem: (key) => {
        if (typeof window === "undefined") return;
        window.localStorage.removeItem(key);
        window.sessionStorage.removeItem(key);
      },
    },
  },
});
