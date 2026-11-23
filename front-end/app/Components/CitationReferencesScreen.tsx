"use client";

import { useEffect, useState } from "react";
import { C, bodyText, cardBase, headingStyle, sectionLabel } from "./theme";
import { getCurrentUserId, getUserPapers, type Paper } from "../../lib/db";
import { API_BASE } from "../../lib/api";
import { exportToDOCX, exportToPDF } from "../../lib/exportUtils";

const STYLES = ["apa", "mla", "ieee", "chicago"] as const;
type CitationStyle = (typeof STYLES)[number];

interface ExtractedReference {
  id: string;
  raw: string;
  formatted: string;
  title?: string | null;
  authors?: string | null;
  year?: number | null;
  venue?: string | null;
  doi?: string | null;
  verification_status?: string;
}

interface ExtractionResult {
  paper_id: string;
  title: string;
  style: string;
  citations_text: string;
  count: number;
  warning?: string;
  references: ExtractedReference[];
}

export default function CitationReferencesScreen() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [style, setStyle] = useState<CitationStyle>("apa");
  const [results, setResults] = useState<ExtractionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    getCurrentUserId().then(uid => {
      if (!uid) return;
      getUserPapers(uid).then(({ data }) => {
        if (active) setPapers(data.filter(paper => paper.status === "indexed"));
      });
    });
    return () => {
      active = false;
    };
  }, []);

  const togglePaper = (paperId: string) => {
    setSelected(previous => {
      const next = new Set(previous);
      if (next.has(paperId)) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
  };

  const extractReferences = async () => {
    const userId = await getCurrentUserId();
    if (!userId || selected.size === 0) return;
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const response = await fetch(`${API_BASE}/agents/citations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          paper_ids: [...selected],
          style,
        }),
      });
      const payload = await response.json() as {
        results?: ExtractionResult[];
        detail?: string;
      };
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      setResults(payload.results || []);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Reference extraction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1050, margin: "0 auto" }}>
      <header className="fade-1" style={{ marginBottom: 24 }}>
        <div style={{ ...sectionLabel, marginBottom: 6 }}>Reference Management</div>
        <h1 style={{ ...headingStyle, fontSize: "clamp(24px, 3vw, 34px)" }}>
          References <em style={{ color: C.gold }}>and Bibliographies</em>
        </h1>
        <p style={{ ...bodyText, fontSize: 14, marginTop: 6 }}>
          Extract and format references locally from your papers. Metadata and retraction status are not verified here.
        </p>
      </header>

      <section style={{ ...cardBase, padding: 22, marginBottom: 22 }}>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 14 }}>
          <strong style={sectionLabel}>Choose papers ({selected.size} selected)</strong>
          <button
            type="button"
            className="btn-outline"
            onClick={() => setSelected(selected.size === papers.length ? new Set() : new Set(papers.map(paper => paper.id)))}
          >
            {selected.size === papers.length ? "Deselect all" : "Select all"}
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 8 }}>
          {papers.map(paper => (
            <label key={paper.id} style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 10,
              padding: 12,
              border: `1px solid ${selected.has(paper.id) ? C.borderGold : C.border}`,
              background: selected.has(paper.id) ? C.goldFaint : "transparent",
              borderRadius: 4,
              cursor: "pointer",
            }}>
              <input type="checkbox" checked={selected.has(paper.id)} onChange={() => togglePaper(paper.id)} />
              <span style={{ ...bodyText, color: C.inkDark, fontSize: 13 }}>{paper.title}</span>
            </label>
          ))}
          {papers.length === 0 && <p style={bodyText}>No indexed papers are available.</p>}
        </div>
        <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 12, marginTop: 18 }}>
          <label style={{ ...bodyText, fontSize: 13 }}>
            Citation style{" "}
            <select value={style} onChange={event => setStyle(event.target.value as CitationStyle)}>
              {STYLES.map(item => <option key={item} value={item}>{item.toUpperCase()}</option>)}
            </select>
          </label>
          <button type="button" className="btn-gold" disabled={loading || selected.size === 0} onClick={() => void extractReferences()}>
            {loading ? "Extracting locally..." : "Extract references"}
          </button>
        </div>
      </section>

      {error && <div role="alert" style={{ ...cardBase, padding: 14, marginBottom: 16, color: C.sienna }}>{error}</div>}

      {results.map(result => (
        <article key={result.paper_id} className="fade-1" style={{
          maxWidth: 820,
          margin: "28px auto",
          padding: "clamp(24px, 5vw, 64px)",
          background: "#fffefa",
          border: `1px solid ${C.border}`,
          boxShadow: `0 8px 30px ${C.shadow}`,
          color: C.inkDark,
        }}>
          <div style={{ borderBottom: `1px solid ${C.border}`, paddingBottom: 16, marginBottom: 22 }}>
            <div style={{ ...sectionLabel, marginBottom: 6 }}>Extracted bibliography · {result.style}</div>
            <h2 style={{ ...headingStyle, fontSize: 23 }}>{result.title}</h2>
            <p style={{ ...bodyText, fontSize: 12, marginTop: 6 }}>
              {result.count} reference{result.count === 1 ? "" : "s"} · local extraction · metadata not checked
            </p>
          </div>
          {result.warning && <p role="status" style={{ ...bodyText, color: C.sienna, marginBottom: 16 }}>{result.warning}</p>}
          {result.references?.length ? result.references.map((reference, index) => (
            <div key={reference.id} style={{
              display: "grid",
              gridTemplateColumns: "32px minmax(0, 1fr)",
              gap: 10,
              padding: "12px 0",
              borderBottom: `1px solid ${C.border}`,
              lineHeight: 1.65,
              fontFamily: "'Crimson Pro', Georgia, serif",
              fontSize: 15,
            }}>
              <span style={{ color: C.gold, fontWeight: 700 }}>{index + 1}.</span>
              <span>{reference.formatted || reference.raw}</span>
            </div>
          )) : (
            <p style={bodyText}>No references were found in this paper.</p>
          )}
          {result.references?.some(reference => reference.verification_status !== "verified") && (
            <p style={{ ...bodyText, fontSize: 12, marginTop: 16, color: C.umber }}>
              Extracted text is shown as parsed. Check citation details against the original source before use.
            </p>
          )}
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 20 }}>
            <button type="button" className="btn-outline" onClick={() => exportToPDF(`References_${result.title}`, result.citations_text)}>
              Export PDF
            </button>
            <button type="button" className="btn-outline" onClick={() => void exportToDOCX(`References_${result.title}`, result.citations_text)}>
              Export DOCX
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}
