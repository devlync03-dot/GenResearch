"use client";
import { useState, useEffect } from "react";
import { C, sectionLabel, headingStyle, bodyText, cardBase } from "./theme";
import { getCurrentUserId, getUserPapers, type Paper } from "../../lib/db";
import { exportToPDF, exportToDOCX } from "../../lib/exportUtils";
import { API_BASE } from "../../lib/api";

type ComparisonRow = {
  paper_id: string;
  paper: string;
  paper_type?: string;
  method?: string;
  data_sample?: string;
  key_findings?: string[];
  stated_limitations?: string[];
};

type LiteratureReviewResult = {
  error?: string;
  review?: string;
  papers_analyzed?: number;
  chunks_used?: number;
  focus_topic?: string;
  scope_warning?: string;
  analysis_error?: string | null;
  comparison_matrix?: ComparisonRow[];
  evidence_quality?: {
    cards_ok?: number;
    quotes_verified?: number;
    quotes_checked?: number;
    card_errors?: Array<{ paper_id: string; error: string }>;
  };
};

function renderInlineMarkdown(text: string, keyPrefix: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*]+\*|_[^_]+_)/g);
  return parts.map((part, index) => {
    if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("__") && part.endsWith("__"))) {
      return <strong key={`${keyPrefix}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${keyPrefix}-${index}`}>{part.slice(1, -1)}</code>;
    }
    if ((part.startsWith("*") && part.endsWith("*")) || (part.startsWith("_") && part.endsWith("_"))) {
      return <em key={`${keyPrefix}-${index}`}>{part.slice(1, -1)}</em>;
    }
    return part;
  });
}

function FormattedReview({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks: React.ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index++;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      const style = level === 1
        ? { fontSize: 18, margin: "8px 0 16px", paddingBottom: 8, borderBottom: "1px solid #bbb" }
        : level === 2
          ? { fontSize: 16, margin: "20px 0 8px", paddingBottom: 5, borderBottom: "1px solid #bbb" }
          : { fontSize: level === 3 ? 14 : 12, margin: "14px 0 6px" };
      blocks.push(
        <h2
          key={`heading-${index}`}
          style={{ ...headingStyle, ...style, color: "#000", fontFamily: "Arial, Helvetica, sans-serif", lineHeight: 1.5 }}
        >
          {renderInlineMarkdown(heading[2], `heading-${index}`)}
        </h2>
      );
      index++;
      continue;
    }

    const listItem = line.match(/^([-*+]|\d+[.)])\s+(.+)$/);
    if (listItem) {
      const ordered = /^\d/.test(listItem[1]);
      const items: string[] = [];
      while (index < lines.length) {
        const item = lines[index].trim().match(/^([-*+]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(item[2]);
        index++;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(
        <List key={`list-${index}`} style={{ paddingLeft: 26, margin: "8px 0 14px", ...bodyText, color: "#111", fontSize: 12, lineHeight: 1.5, textAlign: "justify" }}>
          {items.map((item, itemIndex) => (
            <li key={`list-${index}-${itemIndex}`} style={{ paddingLeft: 4, marginBottom: 6 }}>
              {renderInlineMarkdown(item, `list-${index}-${itemIndex}`)}
            </li>
          ))}
        </List>
      );
      continue;
    }

    const paragraph = [line];
    index++;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4})\s+|([-*+]|\d+[.)])\s+/.test(lines[index].trim())
    ) {
      paragraph.push(lines[index].trim());
      index++;
    }
    blocks.push(
      <p key={`paragraph-${index}`} style={{ ...bodyText, color: "#111", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, lineHeight: 1.5, textAlign: "justify", marginBottom: 10 }}>
        {renderInlineMarkdown(paragraph.join(" "), `paragraph-${index}`)}
      </p>
    );
  }

  return <div>{blocks}</div>;
}

export default function LiteratureReviewScreen({ onBack }: { onBack: () => void }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [focusTopic, setFocusTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LiteratureReviewResult | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);

  useEffect(() => {
    getCurrentUserId().then(uid => {
      setUserId(uid);
      if (uid) getUserPapers(uid).then(({ data }) => setPapers(data.filter(p => p.status === "indexed")));
    });
  }, []);

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const run = async () => {
    if (!userId || selected.size === 0) return;
    setLoading(true); setResult(null);
    setProgress(["Connecting to the literature review agent..."]);
    try {
      const res = await fetch(`${API_BASE}/agents/literature-review/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, paper_ids: [...selected], focus_topic: focusTopic }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}`);
      if (!res.body) throw new Error("The server did not provide a progress stream.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let receivedResult = false;
      const handleLine = (line: string) => {
        if (!line.trim()) return;
        const event: unknown = JSON.parse(line);
        if (!event || typeof event !== "object") {
          throw new Error("The literature-review stream returned an invalid event.");
        }
        const payload = event as {
          type?: string;
          message?: string;
          result?: LiteratureReviewResult;
        };
        if (payload.type === "progress" && payload.message) {
          const message = payload.message;
          setProgress(current => [...current.slice(-29), message]);
        } else if (payload.type === "done" && payload.result) {
          setResult(payload.result);
          receivedResult = true;
        } else if (payload.type === "error") {
          throw new Error(payload.message || "Literature review failed.");
        } else {
          throw new Error("The literature-review stream returned an unknown event.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        lines.forEach(handleLine);
        if (done) break;
      }
      if (buffer.trim()) handleLine(buffer);
      if (!receivedResult) {
        throw new Error("The literature-review stream ended before returning results.");
      }
    } catch (error: unknown) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    }
    finally { setLoading(false); }
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <div className="fade-1" style={{ marginBottom: 28 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", ...bodyText, fontSize: 13, color: C.gold, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>← Back to Agents</button>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ width: 44, height: 44, borderRadius: 4, background: `${C.sienna}18`, border: `1px solid ${C.sienna}33`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>◉</div>
          <div>
            <h1 style={{ ...headingStyle, fontSize: 28 }}>Literature Review <em style={{ color: C.sienna }}>Agent</em></h1>
            <p style={{ ...bodyText, fontSize: 13, marginTop: 2 }}>Synthesize comparative reviews across multiple papers</p>
          </div>
        </div>
      </div>

      {/* Focus Topic */}
      <div className="fade-2" style={{ ...cardBase, padding: "20px 24px", marginBottom: 16 }}>
        <div style={{ ...sectionLabel, marginBottom: 10 }}>Focus Topic (Optional)</div>
        <input value={focusTopic} onChange={e => setFocusTopic(e.target.value)} placeholder="e.g., Machine learning in education assessment" style={{ width: "100%", padding: "12px 14px", border: `1.5px solid ${C.border}`, borderRadius: 3, background: C.creamLight, fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14.5, color: C.inkDark, outline: "none", transition: "border-color .2s" }} onFocus={e => { e.target.style.borderColor = C.sienna; }} onBlur={e => { e.target.style.borderColor = C.border; }} />
        <p style={{ ...bodyText, fontSize: 11.5, marginTop: 6 }}>Leave empty to review all themes, or specify a topic to focus the review.</p>
      </div>

      {/* Paper Selection */}
      <div className="fade-2" style={{ ...cardBase, padding: "24px", marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div style={{ ...sectionLabel }}>Select Papers ({selected.size} selected)</div>
          <button onClick={() => setSelected(selected.size === papers.length ? new Set() : new Set(papers.map(p => p.id)))} style={{ background: "none", border: "none", cursor: "pointer", ...bodyText, fontSize: 12, color: C.sienna, textDecoration: "underline" }}>
            {selected.size === papers.length ? "Deselect All" : "Select All"}
          </button>
        </div>
        {papers.length === 0 ? (
          <p style={{ ...bodyText, fontSize: 14, textAlign: "center", padding: "20px 0" }}>No indexed papers found.</p>
        ) : papers.map(p => (
          <label key={p.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 14px", marginBottom: 6, background: selected.has(p.id) ? C.siennaFaint : "transparent", border: `1px solid ${selected.has(p.id) ? "rgba(160,82,45,0.3)" : C.border}`, borderRadius: 4, cursor: "pointer", transition: "all .2s" }}>
            <input type="checkbox" checked={selected.has(p.id)} onChange={() => toggle(p.id)} style={{ accentColor: C.sienna, width: 16, height: 16 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 14, fontWeight: 700, color: C.inkDark }}>{p.title}</div>
              <div style={{ ...bodyText, fontSize: 11.5 }}>{p.authors || "Unknown"} · {p.chunks} chunks</div>
            </div>
          </label>
        ))}
      </div>

      <div className="fade-3" style={{ marginBottom: 32 }}>
        <button onClick={run} disabled={loading || selected.size === 0} style={{ width: "100%", padding: "14px 0", background: loading || selected.size === 0 ? C.creamDark : C.inkDark, color: loading || selected.size === 0 ? C.inkLight : C.cream, border: "none", borderRadius: 4, fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: loading || selected.size === 0 ? "not-allowed" : "pointer", transition: "all .3s" }}>
          {loading ? "⟳ Generating Literature Review..." : `Generate Review from ${selected.size} Paper${selected.size !== 1 ? "s" : ""}`}
        </button>
        {progress.length > 0 && (
          <div
            aria-live="polite"
            aria-label="Literature review progress"
            style={{ marginTop: 12, padding: "10px 14px", border: "1px solid #bbb", background: "#fff", color: "#111", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, lineHeight: 1.5 }}
          >
            <div style={{ fontWeight: 700, marginBottom: 4 }}>
              {loading ? "Agent progress" : "Recent progress"}
            </div>
            {progress.slice(-6).map((message, index) => (
              <div key={`${progress.length - 6 + index}-${message}`}>{message}</div>
            ))}
          </div>
        )}
      </div>

      {result && !result.error && (
        <div className="fade-1">
          <div style={{ ...cardBase, padding: "28px", background: "#fff", border: "1px solid #777", color: "#111", fontFamily: "Arial, Helvetica, sans-serif" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
              <div>
                <div style={{ fontFamily: "Arial, Helvetica, sans-serif", fontSize: 16, fontWeight: 700, color: "#000" }}>Literature Review</div>
                {result.focus_topic && <div style={{ ...bodyText, fontSize: 12, color: "#111" }}>Focus: {result.focus_topic}</div>}
              </div>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 10 }}>
                <div style={{ ...bodyText, fontSize: 12, color: "#111" }}>{result.papers_analyzed ?? 0} papers · {result.chunks_used ?? 0} excerpts</div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => exportToPDF("Literature_Review", result.review ?? "")} style={{ padding: "4px 8px", background: "#fff", color: "#000", border: "1px solid #555", borderRadius: 3, cursor: "pointer", fontSize: 12 }}>PDF</button>
                  <button onClick={() => exportToDOCX("Literature_Review", result.review ?? "")} style={{ padding: "4px 8px", background: "#fff", color: "#000", border: "1px solid #555", borderRadius: 3, cursor: "pointer", fontSize: 12 }}>DOCX</button>
                </div>
              </div>
            </div>
            <div style={{ padding: "4px 8px", color: "#111", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, lineHeight: 1.5, textAlign: "justify" }}>
              <FormattedReview content={result.review ?? ""} />
            </div>
            {result.scope_warning && (
              <div role="status" style={{ margin: "16px 8px", padding: 12, border: "1px solid #bbb", background: "#fff", color: "#111", ...bodyText, fontSize: 12, lineHeight: 1.5 }}>
                Paper-set scope: {result.scope_warning}
              </div>
            )}
            {result.analysis_error && (
              <div role="status" style={{ margin: "16px 8px", padding: 12, border: "1px solid #bbb", background: "#fff", color: "#111", ...bodyText, fontSize: 12, lineHeight: 1.5 }}>
                Cross-paper analysis was unavailable: {result.analysis_error}. The review still uses retrieved excerpts and individual paper cards.
              </div>
            )}
            {result.evidence_quality && (
              <p style={{ ...bodyText, color: "#111", margin: "12px 8px", fontSize: 12, lineHeight: 1.5 }}>
                Evidence cards: {result.evidence_quality.cards_ok ?? 0}/{result.papers_analyzed} with quote-checked findings · Quotes verified: {result.evidence_quality.quotes_verified ?? 0}/{result.evidence_quality.quotes_checked ?? 0}
              </p>
            )}
            {Array.isArray(result.comparison_matrix) && result.comparison_matrix.length > 0 && (
              <details style={{ margin: "16px 8px", ...bodyText, color: C.inkDark, fontSize: 13 }}>
                <summary style={{ cursor: "pointer", fontWeight: 700 }}>
                  Paper-by-paper comparison · {result.comparison_matrix.length} papers
                </summary>
                <div style={{ overflowX: "auto", marginTop: 10 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: 12 }}>
                    <thead>
                      <tr>
                        {["Paper", "Type / design", "Data / sample", "Quote-checked findings", "Stated limitations"].map(label => (
                          <th key={label} style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>{label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.comparison_matrix.map((row) => (
                        <tr key={row.paper_id}>
                          <td style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>{row.paper}</td>
                          <td style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>{row.paper_type || "Not identified"}<br />{row.method || ""}</td>
                          <td style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>{row.data_sample || "Not specified"}</td>
                          <td style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>
                            {row.key_findings?.length ? (
                              <ul style={{ margin: 0, paddingLeft: 18 }}>
                                {row.key_findings.map((finding: string, index: number) => <li key={index}>{finding}</li>)}
                              </ul>
                            ) : "No quote-verified findings"}
                          </td>
                          <td style={{ borderBottom: "1px solid #bbb", padding: 8, verticalAlign: "top" }}>
                            {row.stated_limitations?.length ? row.stated_limitations.join("; ") : "Not specified"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            {Array.isArray(result.evidence_quality?.card_errors) && result.evidence_quality.card_errors.length > 0 && (
              <details style={{ margin: "12px 8px", ...bodyText, fontSize: 12 }}>
                <summary style={{ cursor: "pointer" }}>Evidence-card notes · {result.evidence_quality.card_errors.length}</summary>
                <ul>
                  {result.evidence_quality.card_errors.map((item) => (
                    <li key={item.paper_id}>{item.paper_id}: {item.error}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </div>
      )}

      {result?.error && (
        <div style={{ ...cardBase, padding: "20px", borderColor: "rgba(160,82,45,0.3)", background: "rgba(160,82,45,0.05)" }}>
          <span style={{ ...bodyText, color: C.sienna }}>⚠ {result.error}</span>
        </div>
      )}
    </div>
  );
}
