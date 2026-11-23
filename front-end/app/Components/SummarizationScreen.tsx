"use client";
import { useState, useEffect } from "react";
import { C, sectionLabel, headingStyle, bodyText, cardBase } from "./theme";
import { getCurrentUserId, getUserPapers, type Paper } from "../../lib/db";
import { exportToPDF, exportToDOCX } from "../../lib/exportUtils";
import { API_BASE } from "../../lib/api";

type PaperSummary = {
  paper_id?: string;
  title: string;
  authors?: string[] | string;
  year?: number | null;
  paper_type?: string;
  doi?: string | null;
  venue?: string | null;
  metadata_url?: string | null;
  summary: string;
  chunks_used?: number;
  sections_analyzed?: number;
  status: string;
  error?: string;
  detail?: "quick" | "standard" | "deep";
  sources?: Array<{
    excerpt: string;
    page?: number | null;
    section?: string | null;
    text: string;
  }>;
  reproducibility?: {
    models?: string[];
    embedding_model?: string;
    chunk_size?: number;
    prompt_version?: string;
    runtime_seconds?: number;
  };
};

type SummaryResponse = {
  summaries?: PaperSummary[];
  error?: string;
};

function summaryExportText(summary: PaperSummary): string {
  const reproducibility = summary.reproducibility;
  const reproducibilityLine = reproducibility
    ? `Reproducibility: models ${reproducibility.models?.join(", ") || "unavailable"}; embedding model ${reproducibility.embedding_model || "unavailable"}; chunk size ${reproducibility.chunk_size ?? "unavailable"}; prompt ${reproducibility.prompt_version || "unavailable"}; runtime ${reproducibility.runtime_seconds ?? "unavailable"} seconds.`
    : "";
  return [summary.summary, reproducibilityLine].filter(Boolean).join("\n\n");
}

function SummaryContent({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks: Array<{ type: string; text: string; level?: number; items?: string[] }> = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: "paragraph", text: paragraph.join("\n") });
      paragraph = [];
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      blocks.push({ type: "heading", text: heading[2], level: heading[1].length });
      continue;
    }

    if (/^```/.test(trimmed)) {
      flushParagraph();
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index].trim())) {
        codeLines.push(lines[index]);
        index += 1;
      }
      blocks.push({ type: "code", text: codeLines.join("\n") });
      continue;
    }

    const listMatch = trimmed.match(/^([-*]|\d+\.)\s+(.+)$/);
    if (listMatch) {
      flushParagraph();
      const ordered = /^\d+\./.test(listMatch[1]);
      const items = [listMatch[2]];
      while (index + 1 < lines.length) {
        const nextItem = lines[index + 1].trim().match(/^([-*]|\d+\.)\s+(.+)$/);
        if (!nextItem || /^\d+\./.test(nextItem[1]) !== ordered) break;
        items.push(nextItem[2]);
        index += 1;
      }
      blocks.push({ type: ordered ? "ordered-list" : "unordered-list", text: "", items });
      continue;
    }

    paragraph.push(line);
  }
  flushParagraph();

  return (
    <div style={{ fontFamily: "Arial, Helvetica, sans-serif", color: "#111", fontSize: 12, lineHeight: 1.5, textAlign: "justify" }}>
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          const Heading = block.level === 1 ? "h2" : block.level === 2 ? "h3" : "h4";
          return <Heading key={index} style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", lineHeight: 1.5, textAlign: "left", margin: "16px 0 6px", fontSize: block.level === 1 ? 16 : 14 }}>{block.text}</Heading>;
        }
        if (block.type === "ordered-list" || block.type === "unordered-list") {
          const List = block.type === "ordered-list" ? "ol" : "ul";
          return (
            <List key={index} style={{ paddingLeft: 24, margin: "8px 0 14px", textAlign: "justify" }}>
              {block.items?.map((item, itemIndex) => <li key={itemIndex} style={{ marginBottom: 5 }}>{item}</li>)}
            </List>
          );
        }
        if (block.type === "code") {
          return <pre key={index} style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", background: "#fff", padding: 12, border: "1px solid #bbb", margin: "10px 0", color: "#111", fontSize: 12, lineHeight: 1.5 }}>{block.text}</pre>;
        }
        return <p key={index} style={{ whiteSpace: "pre-wrap", margin: "0 0 10px", textAlign: "justify" }}>{block.text}</p>;
      })}
    </div>
  );
}

export default function SummarizationScreen({ onBack }: { onBack: () => void }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SummaryResponse | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);
  const [detail, setDetail] = useState<"quick" | "standard" | "deep">("standard");

  useEffect(() => {
    getCurrentUserId().then(uid => {
      setUserId(uid);
      if (uid) getUserPapers(uid).then(({ data }) => setPapers(data.filter(p => p.status === "indexed")));
    });
  }, []);

  const toggle = (id: string) => {
    setSelected(prev => {
      const s = new Set(prev);
      if (s.has(id)) {
        s.delete(id);
      } else {
        s.add(id);
      }
      return s;
    });
  };

  const run = async () => {
    if (!userId || selected.size === 0) return;
    setLoading(true);
    setResult(null);
    setProgress(["Connecting to the summarization agent..."]);
    try {
      const res = await fetch(`${API_BASE}/agents/summarize/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, paper_ids: [...selected], detail }),
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
          throw new Error("The summarization stream returned an invalid event.");
        }
        const payload = event as { type?: string; message?: string; result?: SummaryResponse };
        if (payload.type === "progress" && payload.message) {
          const message = payload.message;
          setProgress(current => [...current.slice(-29), message]);
        } else if (payload.type === "done" && payload.result) {
          setResult(payload.result);
          receivedResult = true;
        } else if (payload.type === "error") {
          throw new Error(payload.message || "Summarization failed.");
        } else {
          throw new Error("The summarization stream returned an unknown event.");
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
      if (!receivedResult) throw new Error("The summarization stream ended before returning results.");
    } catch (error: unknown) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      {/* Header */}
      <div className="fade-1" style={{ marginBottom: 28 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", ...bodyText, fontSize: 13, color: C.gold, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          ← Back to Agents
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ width: 44, height: 44, borderRadius: 4, background: `${C.gold}18`, border: `1px solid ${C.gold}33`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>◈</div>
          <div>
            <h1 style={{ ...headingStyle, fontSize: 28 }}>Summarization <em style={{ color: C.gold }}>Agent</em></h1>
            <p style={{ ...bodyText, fontSize: 13, marginTop: 2 }}>Create detailed, evidence-linked explanations of your indexed papers</p>
          </div>
        </div>
      </div>

      {/* Paper Selection */}
      <div className="fade-2" style={{ ...cardBase, padding: "24px", marginBottom: 24 }}>
        <div style={{ ...sectionLabel, marginBottom: 14 }}>Select Papers to Summarize</div>
        {papers.length === 0 ? (
          <p style={{ ...bodyText, fontSize: 14, textAlign: "center", padding: "20px 0" }}>No indexed papers found. Upload papers first.</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ ...bodyText, fontSize: 12 }}>{selected.size} of {papers.length} selected</span>
              <button onClick={() => setSelected(prev => prev.size === papers.length ? new Set() : new Set(papers.map(p => p.id)))} style={{ background: "none", border: "none", cursor: "pointer", ...bodyText, fontSize: 12, color: C.gold, textDecoration: "underline" }}>
                {selected.size === papers.length ? "Deselect All" : "Select All"}
              </button>
            </div>
            {papers.map(p => (
              <label key={p.id} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 14px", background: selected.has(p.id) ? C.goldFaint : "transparent", border: `1px solid ${selected.has(p.id) ? C.borderGold : C.border}`, borderRadius: 4, cursor: "pointer", transition: "all .2s" }}>
                <input type="checkbox" checked={selected.has(p.id)} onChange={() => toggle(p.id)} style={{ accentColor: C.gold, width: 16, height: 16 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 14, fontWeight: 700, color: C.inkDark }}>{p.title}</div>
                  <div style={{ ...bodyText, fontSize: 11.5 }}>{p.authors || "Unknown author"} · {p.chunks} chunks · {p.file_size}</div>
                </div>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Run Button */}
      <div className="fade-3" style={{ marginBottom: 32 }}>
        <label style={{ ...bodyText, display: "flex", alignItems: "center", gap: 10, marginBottom: 10, fontSize: 13 }}>
          Summary detail
          <select
            value={detail}
            onChange={event => setDetail(event.target.value as "quick" | "standard" | "deep")}
            disabled={loading}
            style={{ padding: "6px 10px", background: "#fff", color: "#111", border: "1px solid #777", fontSize: 12 }}
          >
            <option value="quick">Quick · concise</option>
            <option value="standard">Standard · balanced</option>
            <option value="deep">Deep · fuller coverage</option>
          </select>
        </label>
        <button onClick={run} disabled={loading || selected.size === 0} style={{ width: "100%", padding: "14px 0", background: loading || selected.size === 0 ? C.creamDark : C.inkDark, color: loading || selected.size === 0 ? C.inkLight : C.cream, border: "none", borderRadius: 4, fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: loading || selected.size === 0 ? "not-allowed" : "pointer", transition: "all .3s" }}>
          {loading ? "Generating concise summaries..." : `Summarize ${selected.size} Paper${selected.size !== 1 ? "s" : ""}`}
        </button>
        {loading && progress.length > 0 && (
          <div
            aria-live="polite"
            aria-label="Summarization progress"
            style={{ marginTop: 12, padding: "10px 14px", border: "1px solid #bbb", background: "#fff", color: "#111", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, lineHeight: 1.5 }}
          >
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Agent progress</div>
            {progress.slice(-6).map((message, index) => (
              <div key={`${progress.length - 6 + index}-${message}`}>{message}</div>
            ))}
          </div>
        )}
      </div>

      {/* Results */}
      {result && !result.error && (
        <div className="fade-1">
          <div style={{ ...sectionLabel, marginBottom: 14 }}>Generated Summaries</div>
          {result.summaries?.map((s, i) => (
            <div key={i} style={{ ...cardBase, padding: "24px", marginBottom: 16, background: "#fff", border: "1px solid #777", color: "#111", fontFamily: "Arial, Helvetica, sans-serif" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
                <div style={{ fontFamily: "Arial, Helvetica, sans-serif", fontSize: 16, fontWeight: 700, lineHeight: 1.5, color: "#000" }}>
                  {s.title === "Unknown" || s.title === "Untitled Paper"
                    ? papers.find(p => p.id === s.paper_id)?.title || s.title
                    : s.title}
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  {s.summary && (
                    <>
                          <button onClick={() => exportToPDF(`Summary_${s.title}`, summaryExportText(s))} style={{ padding: "4px 8px", background: "#fff", color: "#000", border: "1px solid #555", cursor: "pointer", fontSize: 12 }}>PDF</button>
                          <button onClick={() => exportToDOCX(`Summary_${s.title}`, summaryExportText(s))} style={{ padding: "4px 8px", background: "#fff", color: "#000", border: "1px solid #555", cursor: "pointer", fontSize: 12 }}>DOCX</button>
                    </>
                  )}
                </div>
              </div>
              <div style={{ color: "#111", fontSize: 12, lineHeight: 1.5, marginBottom: 14 }}>
                {[s.authors && (Array.isArray(s.authors) ? s.authors.join(", ") : s.authors), s.year, s.paper_type, s.venue, s.doi].filter(Boolean).join(" · ")}
                <br />
                {s.chunks_used || 0} chunks analyzed · {s.sections_analyzed || 0} sections summarized · Status: {s.status}
              </div>
              {s.error ? (
                <p role="alert" style={{ ...bodyText, color: C.sienna, fontSize: 13, overflowWrap: "anywhere" }}>
                  {s.error}
                </p>
              ) : (
                <SummaryContent content={s.summary || ""} />
              )}
              {s.sources?.length ? (
                <details style={{ color: "#111", fontSize: 12, lineHeight: 1.5, marginTop: 12 }}>
                  <summary style={{ cursor: "pointer", fontWeight: 700 }}>Evidence appendix · {s.sources.length} excerpts</summary>
                  {s.sources.map((source, sourceIndex) => (
                    <details key={`${source.excerpt}-${sourceIndex}`} style={{ margin: "8px 0 0 12px" }}>
                      <summary style={{ cursor: "pointer" }}>{source.excerpt}{source.page ? ` · page ${source.page}` : ""}{source.section ? ` · ${source.section}` : ""}</summary>
                      <p style={{ marginTop: 6, textAlign: "justify", whiteSpace: "pre-wrap" }}>{source.text}</p>
                    </details>
                  ))}
                </details>
              ) : null}
              {s.reproducibility && (
                <p style={{ color: "#111", fontSize: 12, lineHeight: 1.5, marginTop: 12, borderTop: "1px solid #bbb", paddingTop: 8 }}>
                  Reproducibility: models {s.reproducibility.models?.join(", ") || "unavailable"} · embeddings {s.reproducibility.embedding_model || "unavailable"} · chunk size {s.reproducibility.chunk_size ?? "unavailable"} · prompt {s.reproducibility.prompt_version || "unavailable"} · runtime {s.reproducibility.runtime_seconds ?? "unavailable"}s
                </p>
              )}
            </div>
          ))}
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
