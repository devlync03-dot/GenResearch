const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

export const API_BASE = (configuredApiUrl || "http://localhost:8000").replace(/\/+$/, "");
