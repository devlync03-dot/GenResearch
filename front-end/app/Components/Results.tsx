"use client";

import { useState, useEffect } from "react";
import { cardBase } from "./theme";
import { getCurrentUserId, subscribeToResults, type TaskResult as DBResult } from "../../lib/db";
import type { DiscoverySearchPayload } from "../../lib/db";

import { exportToPDF, exportToDOCX } from "../../lib/exportUtils";

type ResultType = "summary" | "review" | "citation" | "proposal" | "case_studies" | "research_papers";
interface Result {
  id: string; title: string; agent: string; type: ResultType;
  score: number; date: string; preview: string;
  discovery?: DiscoverySearchPayload;
  output?: Record<string, unknown>;
}

const typeConfig: Record<ResultType, { label: string; agent: string }> = {
  summary:  { label: "Summary",  agent: "Summarization Agent" },
  review:   { label: "Review",   agent: "Literature Review Agent" },
  citation: { label: "Citation", agent: "Citation Agent" },
  proposal: { label: "Proposal", agent: "Proposal Drafting Agent" },
  case_studies: { label: "Case Studies", agent: "Case Study Search" },
  research_papers: { label: "Paper Search", agent: "Academic Discovery" },
};

function parseDiscoveryContent(content: string): DiscoverySearchPayload | null {
  try {
    const parsed = JSON.parse(content) as DiscoverySearchPayload;
    return parsed.kind === "discovery_search" ? parsed : null;
  } catch {
    return null;
  }
}

function parseAgentOutput(content: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(content);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function outputToText(value: unknown, label = ""): string {
  if (typeof value === "string") {
    return value.trim() ? `${label ? `${label}\n` : ""}${value}` : "";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return `${label ? `${label}: ` : ""}${value}`;
  }
  if (Array.isArray(value)) {
    return value.map((item, index) =>
      outputToText(item, typeof item === "object" && item !== null
        ? ""
        : `${label}${label ? " " : ""}${index + 1}`)
    ).filter(Boolean).join("\n\n");
  }
  if (value && typeof value === "object") {
    const fields = value as Record<string, unknown>;
    const verification = fields.verification as Record<string, unknown> | undefined;
    const title = typeof fields.title === "string" ? fields.title : "";
    const verificationText = verification
      ? `Verification: ${verification.supported_claims ?? 0}/${verification.total_claims ?? 0} claims supported (${verification.status ?? "unavailable"})`
      : "";
    const remaining = Object.entries(fields)
      .filter(([key]) => ![
        "agent", "paper_id", "chunk_id", "title", "sources", "issues_found", "verification",
      ].includes(key))
      .map(([key, item]) => outputToText(
        item,
        outputLabels[key] || key.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase())
      ))
      .filter(Boolean)
      .join("\n\n");
    return [title ? `## ${title}` : "", verificationText, remaining].filter(Boolean).join("\n\n");
  }
  return "";
}

function mapStoredResults(dbResults: DBResult[]): Result[] {
  const standardResults: Result[] = [];
  const discoveryGroups = new Map<string, { row: DBResult; payload: DiscoverySearchPayload }>();

  for (const row of dbResults) {
    const payload = parseDiscoveryContent(row.content);
    if (!payload) {
      const output = parseAgentOutput(row.content);
      const type = row.type as ResultType;
      standardResults.push({
        id: row.id,
        title: row.title,
        agent: typeConfig[type]?.agent || "Unknown Agent",
        type,
        score: row.score,
        date: new Date(row.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
        preview: output ? outputToText(output) : row.content,
        ...(output ? { output } : {}),
      });
      continue;
    }

    const groupId = row.task_id || row.id;
    const existing = discoveryGroups.get(groupId);
    if (!existing) {
      discoveryGroups.set(groupId, { row, payload });
      continue;
    }

    discoveryGroups.set(groupId, {
      row: existing.row,
      payload: {
        ...existing.payload,
        records: existing.payload.records.length ? existing.payload.records : payload.records,
        reports: {
          all: existing.payload.reports.all || payload.reports.all,
          byStudy: { ...payload.reports.byStudy, ...existing.payload.reports.byStudy },
        },
      },
    });
  }

  const discoveryResults = [...discoveryGroups.entries()].map(([id, group]) => {
    const type = group.payload.category;
    const preview = group.payload.reports.all || group.payload.query;
    return {
      id,
      title: group.row.title,
      agent: typeConfig[type].agent,
      type,
      score: group.row.score,
      date: new Date(group.row.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
      preview,
      discovery: group.payload,
    };
  });

  return [...standardResults, ...discoveryResults].sort((left, right) =>
    new Date(right.date).getTime() - new Date(left.date).getTime()
  );
}

function getResultText(result: Result): string {
  if (!result.discovery) return result.output ? outputToText(result.output) : result.preview;
  const discovery = result.discovery;
  const entries = discovery.records.map((record, index) => {
    const authors = Array.isArray(record.authors) ? record.authors.join(", ") : record.authors || "Unknown authors";
    const metadata = [authors, record.year || "n.d.", record.venue, record.source].filter(Boolean).join(" · ");
    return `## ${index + 1}. ${record.title || "Untitled"}\n${metadata}\n\n${record.abstract || "Abstract unavailable."}${record.url ? `\n\n${record.url}` : ""}`;
  });
  const summaries = Object.entries(discovery.reports.byStudy).map(([key, text]) => `## Summary: ${key}\n\n${text}`);
  return [
    `# ${discovery.category === "case_studies" ? "Case Study Search" : "Research Paper Search"}`,
    `Query: ${discovery.query}`,
    discovery.reports.all,
    ...summaries,
    `## ${discovery.category === "case_studies" ? "Case Studies" : "Research Papers"}`,
    ...entries,
  ].filter(Boolean).join("\n\n");
}

const reportTextStyle: React.CSSProperties = {
  color: "#111",
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 12,
  lineHeight: 1.5,
  textAlign: "justify",
  overflowWrap: "anywhere",
};

function ReportText({ text }: { text: string }) {
  const renderInline = (value: string) => value.split(/(\*\*.+?\*\*|__.+?__|`[^`]+`)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("__") && part.endsWith("__")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index} style={{ fontFamily: "monospace" }}>{part.slice(1, -1)}</code>;
    return part;
  });

  return (
    <div style={reportTextStyle}>
      {text.split(/\r?\n/).map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} style={{ height: 8 }} />;
        const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          const level = heading[1].length;
          const Heading = level === 1 ? "h2" : "h3";
          return (
            <Heading key={index} style={{
              color: "#000",
              fontFamily: "Arial, Helvetica, sans-serif",
              fontSize: level === 1 ? 16 : 14,
              fontWeight: 700,
              lineHeight: 1.5,
              textAlign: "left",
              margin: "10px 0 4px",
            }}>
              {renderInline(heading[2])}
            </Heading>
          );
        }
        const listItem = trimmed.match(/^([-*+]|\d+[.)])\s+(.+)$/);
        return (
          <p key={index} style={{ margin: "0 0 6px", textAlign: "justify" }}>
            {renderInline(listItem ? `${listItem[1]} ${listItem[2]}` : trimmed)}
          </p>
        );
      })}
    </div>
  );
}

const hiddenOutputFields = new Set(["agent", "paper_id", "chunk_id", "steps_log"]);
const outputLabels: Record<string, string> = {
  authors: "Authors",
  bibliography_quality: "Bibliography quality",
  citations_text: "Citations",
  chunks_used: "Chunks analyzed",
  coverage: "Coverage",
  section_coverage: "Section coverage",
  sections_analyzed: "Sections summarized",
  sections_covered: "Sections covered",
  total_sections: "Total sections",
  pages_covered: "Pages covered",
  total_pages: "Total pages",
  section_titles_missing: "Uncovered sections",
  count: "Citation count",
  focus_topic: "Focus topic",
  doi: "DOI",
  embedding_model: "Embedding model",
  issues_found: "Issues found",
  metadata_url: "Metadata record",
  metadata_lookup: "Metadata lookup",
  models: "Models",
  paper_type: "Paper type",
  papers_analyzed: "Papers analyzed",
  papers_used: "Papers used",
  paper_titles: "Papers",
  status: "Status",
  style: "Citation style",
  topic: "Topic",
  total_papers: "Total papers",
  venue: "Venue",
  verification: "Verification",
  runtime_seconds: "Runtime (seconds)",
  chunk_size: "Chunk size",
  prompt_version: "Prompt version",
  verified: "Verified",
};

function OutputValue({ value, label = "", paper = false }: {
  value: unknown;
  label?: string;
  paper?: boolean;
}) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "string") {
    return (
      <section style={{ marginBottom: 14 }}>
        {label && <h3 style={{ color: "#000", fontSize: 14, lineHeight: 1.5, margin: "0 0 4px" }}>{outputLabels[label] || label}</h3>}
        <ReportText text={value} />
      </section>
    );
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return (
      <p style={{ ...reportTextStyle, margin: "0 0 6px" }}>
        {label && <strong>{outputLabels[label] || label}: </strong>}{String(value)}
      </p>
    );
  }
  if (Array.isArray(value)) {
    if (!value.length) return null;
    if (label === "sources") {
      return (
        <details onClick={event => event.stopPropagation()} style={{ margin: "8px 0 14px", ...reportTextStyle }}>
          <summary style={{ cursor: "pointer", fontWeight: 700 }}>
            Evidence appendix · {value.length} excerpts
          </summary>
          <div style={{ padding: "8px 0 0 12px" }}>
            {value.map((item, index) => {
              const excerpt = item && typeof item === "object"
                ? item as Record<string, unknown>
                : {};
              const excerptLabel = String(excerpt.excerpt || `E${index + 1}`);
              const pageLabel = excerpt.page ? ` · page ${excerpt.page}` : "";
              const sectionLabel = excerpt.section ? ` · ${excerpt.section}` : "";
              return (
                <details key={excerptLabel} style={{ marginBottom: 8 }}>
                  <summary style={{ cursor: "pointer" }}>
                    {excerptLabel}{pageLabel}{sectionLabel}
                  </summary>
                  <div style={{ padding: "6px 0 6px 12px" }}>
                    <ReportText text={String(excerpt.text || "Excerpt text unavailable.")} />
                  </div>
                </details>
              );
            })}
          </div>
        </details>
      );
    }
    if (label === "authors" || label === "paper_titles") {
      return (
        <p style={{ ...reportTextStyle, margin: "0 0 6px" }}>
          <strong>{outputLabels[label]}: </strong>{value.map(String).join(", ")}
        </p>
      );
    }
    return (
      <section style={{ marginBottom: 14 }}>
        {label && <h3 style={{ color: "#000", fontSize: 14, lineHeight: 1.5, margin: "0 0 6px" }}>{outputLabels[label] || label}</h3>}
        {value.map((item, index) => (
          <div key={index} style={item && typeof item === "object" ? {
            borderTop: "1px solid #bbb",
            padding: "10px 0",
          } : undefined}>
            <OutputValue value={item} paper={Boolean(item && typeof item === "object")} />
          </div>
        ))}
      </section>
    );
  }
  if (typeof value !== "object") return null;

  const fields = Object.entries(value as Record<string, unknown>)
    .filter(([key]) => !hiddenOutputFields.has(key) && !["sources", "issues_found", "verification"].includes(key));
  const record = value as Record<string, unknown>;
  const title = record.title;
  const titleText = typeof title === "string" ? title : "";
  const verification = record.verification && typeof record.verification === "object"
    ? record.verification as Record<string, unknown>
    : null;
  const failures = Array.isArray(record.issues_found)
    ? record.issues_found
    : verification && Array.isArray(verification.failures) ? verification.failures : [];
  const sources = Array.isArray(record.sources) ? record.sources : [];
  return (
    <article style={paper ? { marginBottom: 12 } : undefined}>
      {titleText && (
        <h2 style={{
          color: "#000",
          fontSize: paper ? 14 : 16,
          fontWeight: 700,
          lineHeight: 1.5,
          margin: "0 0 6px",
          textAlign: "left",
        }}>{titleText}</h2>
      )}
      {fields
        .filter(([key]) => key !== "title")
        .map(([key, item]) => (
          <OutputValue key={key} value={item} label={key} />
        ))}
      {verification && (
        <p style={{ ...reportTextStyle, margin: "4px 0 10px" }}>
          <strong>Verification: </strong>
          {String(verification.supported_claims ?? 0)}/{String(verification.total_claims ?? 0)} claims supported
          {" · "}{String(verification.status || "unavailable")}
        </p>
      )}
      {failures.length > 0 && (
        <details onClick={event => event.stopPropagation()} style={{ ...reportTextStyle, margin: "8px 0 14px" }}>
          <summary style={{ cursor: "pointer", fontWeight: 700 }}>
            Verification appendix · {failures.length} claims need review
          </summary>
          <div style={{ padding: "8px 0 0 12px" }}>
            {failures.map((failure, index) => (
              <OutputValue key={index} value={failure} paper />
            ))}
          </div>
        </details>
      )}
      {sources.length > 0 && <OutputValue value={sources} label="sources" />}
    </article>
  );
}

function AgentOutput({ output }: { output: Record<string, unknown> }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #bbb", padding: 16 }}>
      <OutputValue value={output} />
    </div>
  );
}

function ResultCard({ result, expanded, onToggle }: { result: Result; expanded: boolean; onToggle: () => void }) {
  const tc = typeConfig[result.type] || typeConfig.summary;
  const resultText = getResultText(result);

  const handleShare = (e: React.MouseEvent) => {
    e.stopPropagation();
    const shareText = `GenResearch Output: ${result.title}\nAgent: ${result.agent}\n\n${resultText.substring(0, 300)}...`;
    navigator.clipboard.writeText(shareText).catch(() => {});
    alert("Shareable summary copied to clipboard!");
  };

  return (
    <div style={{ ...cardBase, padding: "22px 24px", background: "#fff", border: "1px solid #bbb", transition: "all .25s", cursor: "pointer" }} onClick={onToggle}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <div style={{ flex: 1, paddingRight: 12 }}>
          <div style={{ fontFamily: "Arial, Helvetica, sans-serif", fontSize: 16, fontWeight: 700, color: "#000", marginBottom: 4, lineHeight: 1.5 }}>{result.title}</div>
          <div style={{ ...reportTextStyle, textAlign: "left" }}>{result.agent} · {result.date}</div>
        </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <span style={{ padding: "3px 10px", borderRadius: 2, background: "#fff", border: "1px solid #777", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 11, fontWeight: 600, color: "#000", letterSpacing: "0.08em", textTransform: "uppercase" }}>{tc.label}</span>
          {result.discovery && <button className="btn-outline" onClick={(event) => { event.stopPropagation(); onToggle(); }}>{expanded ? "Hide search results" : "View search results"}</button>}
          {!result.discovery && <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontFamily: "Arial, Helvetica, sans-serif", fontSize: 16, fontWeight: 700, color: "#000" }}>{result.score}</span>
            <span style={{ ...reportTextStyle, textAlign: "left" }}>/100</span>
          </div>}
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid #bbb" }}>
          {result.discovery && <p style={{ ...reportTextStyle, marginBottom: 14, textAlign: "left" }}>Search query: <strong>{result.discovery.query}</strong></p>}
          {result.discovery?.reports.all && <><div style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, fontWeight: 700, marginBottom: 10 }}>Generated Report</div><div style={{ background: "#fff", border: "1px solid #bbb", padding: 16, marginBottom: 18 }}><ReportText text={result.discovery.reports.all} /></div></>}
          {result.discovery && Object.entries(result.discovery.reports.byStudy).map(([key, summary]) => (
            <div key={key} style={{ marginBottom: 18 }}><div style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, fontWeight: 700, marginBottom: 10 }}>Paper Summary · {key}</div><div style={{ background: "#fff", border: "1px solid #bbb", padding: 16 }}><ReportText text={summary} /></div></div>
          ))}
          {result.discovery && <div style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, fontWeight: 700, margin: "18px 0 10px" }}>{result.discovery.category === "case_studies" ? "Case Studies" : "Research Papers"} · {result.discovery.records.length}</div>}
          {result.discovery?.records.map((record, index) => {
            const authors = Array.isArray(record.authors) ? record.authors.join(", ") : record.authors || "Unknown authors";
            return <article key={`${record.title}-${index}`} style={{ padding: "12px 0", borderTop: "1px solid #bbb" }}>
              <h2 style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, lineHeight: 1.5, margin: "0 0 4px" }}>{index + 1}. {record.title || "Untitled"}</h2>
              <p style={{ ...reportTextStyle, textAlign: "left" }}>{[authors, record.year || "n.d.", record.venue, record.source].filter(Boolean).join(" · ")}</p>
              {record.abstract && <div style={{ marginTop: 5 }}><ReportText text={record.abstract} /></div>}
              {record.url && <a href={record.url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()} style={{ display: "inline-block", color: "#111", fontSize: 12, marginTop: 6, textDecoration: "underline" }}>Open source</a>}
            </article>;
          })}
          {!result.discovery && <><div style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, fontWeight: 700, marginBottom: 10 }}>Report</div>{result.output ? <AgentOutput output={result.output} /> : <div style={{ background: "#fff", border: "1px solid #bbb", padding: 16 }}><ReportText text={result.preview} /></div>}</>}
          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            <button style={{ padding: "7px 14px", fontSize: 12, color: "#000", background: "#fff", border: "1px solid #555", cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); exportToPDF(result.title, resultText); }}>↓ PDF</button>
            <button style={{ padding: "7px 14px", fontSize: 12, color: "#000", background: "#fff", border: "1px solid #555", cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); void exportToDOCX(result.title, resultText); }}>↓ DOCX</button>
            <button style={{ padding: "7px 14px", fontSize: 12, color: "#000", background: "#fff", border: "1px solid #555", cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(resultText).catch(() => {}); alert("Copied to clipboard!"); }}>Copy Text</button>
            <button style={{ padding: "7px 14px", fontSize: 12, color: "#000", background: "#fff", border: "1px solid #555", cursor: "pointer" }} onClick={handleShare}>Share</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Results({ focusedTaskId = null }: { focusedTaskId?: string | null }) {
  const [dbResults, setDbResults] = useState<DBResult[]>([]);
  const [activeFilter, setActiveFilter] = useState<ResultType | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(focusedTaskId);

  useEffect(() => {
    let isMounted = true;
    let unsub: (() => void) | undefined;
    getCurrentUserId().then(uid => {
      if (uid && isMounted) unsub = subscribeToResults(uid, setDbResults);
    });
    return () => {
      isMounted = false;
      unsub?.();
    };
  }, []);

  const results = mapStoredResults(dbResults);

  const filters: { key: ResultType | "all"; label: string }[] = [
    { key: "all", label: "All" }, { key: "summary", label: "Summaries" },
    { key: "review", label: "Reviews" }, { key: "citation", label: "Citations" },
    { key: "proposal", label: "Proposals" },
    { key: "case_studies", label: "Case Studies" },
    { key: "research_papers", label: "Paper Searches" },
  ];

  const filtered = results.filter(r => {
    const matchesFilter = activeFilter === "all" || r.type === activeFilter;
    const q = searchQuery.toLowerCase().trim();
    const matchesSearch = !q || r.title.toLowerCase().includes(q) || r.preview.toLowerCase().includes(q) || r.agent.toLowerCase().includes(q);
    return matchesFilter && matchesSearch;
  });

  const exportAllFiltered = () => {
    if (filtered.length === 0) return alert("No results to export.");
    const combinedText = filtered.map(r => `# ${r.title}\nAgent: ${r.agent} (${r.date})\n\n${getResultText(r)}\n\n---\n`).join("\n");
    exportToPDF("All_Research_Results", combinedText);
  };

  return (
    <>
      <div className="fade-1" style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ height: 1, width: 28, background: "#111" }} />
          <span style={{ color: "#111", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase" }}>Research Output</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h1 style={{ color: "#000", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 16, lineHeight: 1.5 }}>Task Results</h1>
            <p style={{ ...reportTextStyle, textAlign: "left", marginTop: 4 }}>{results.length} completed outputs ready to review</p>
          </div>
          <button onClick={exportAllFiltered} style={{ padding: "8px 18px", fontSize: 12, color: "#000", background: "#fff", border: "1px solid #555", cursor: "pointer" }}>Export All ({filtered.length})</button>
        </div>
      </div>

      <div className="fade-2" style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ position: "relative", flex: 1, minWidth: 220 }}>
          <input
            placeholder="Search within results & content..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            style={{ width: "100%", padding: "9px 14px", border: "1px solid #777", borderRadius: 0, background: "#fff", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, color: "#000", outline: "none" }}
            onFocus={e => (e.target.style.borderColor = "#000")}
            onBlur={e => (e.target.style.borderColor = "#777")}
          />
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {filters.map(f => (
            <button key={f.key} onClick={() => setActiveFilter(f.key)} style={{ padding: "6px 16px", borderRadius: 0, cursor: "pointer", border: `1px solid ${activeFilter === f.key ? "#000" : "#777"}`, background: activeFilter === f.key ? "#000" : "#fff", fontFamily: "Arial, Helvetica, sans-serif", fontSize: 12, fontWeight: activeFilter === f.key ? 700 : 400, color: activeFilter === f.key ? "#fff" : "#000", transition: "all .2s" }}>{f.label}</button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 24 }}>
        <div style={{ flex: 1, height: 1, background: "#bbb" }} />
        <div style={{ fontFamily: "Arial, Helvetica, sans-serif", fontSize: 14, color: "#000" }}>·</div>
        <div style={{ flex: 1, height: 1, background: "#bbb" }} />
      </div>

      <div className="fade-3" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {filtered.map((r, i) => (
          <div key={r.id} style={{ animation: `fadeUp .5s ${i * 0.06 + 0.1}s both` }}>
            <ResultCard result={r} expanded={expandedId === r.id} onToggle={() => setExpandedId(expandedId === r.id ? null : r.id)} />
          </div>
        ))}
      </div>
      {filtered.length === 0 && <div style={{ textAlign: "center", padding: "60px 20px", ...reportTextStyle }}>No results match your search and filter criteria.</div>}
    </>
  );
}
