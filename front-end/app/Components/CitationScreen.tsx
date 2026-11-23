"use client";
import { useState, useEffect, useRef } from "react";
import { C, sectionLabel, headingStyle, bodyText, cardBase } from "./theme";
import { getCurrentUserId, getUserPapers, type Paper } from "../../lib/db";
import { API_BASE } from "../../lib/api";
import { extractCitationDocumentText } from "../../lib/citationDocument";

const STYLES = [
  { key: "apa", label: "APA 7th Edition", desc: "Author-Date style, most common in social sciences" },
  { key: "mla", label: "MLA 9th Edition", desc: "Used in humanities and liberal arts" },
  { key: "ieee", label: "IEEE", desc: "Numbered style, used in engineering and CS" },
  { key: "chicago", label: "Chicago 17th", desc: "Notes-Bibliography, used in history and arts" },
];

interface CitationSource {
  id: string;
  title: string;
  year?: number | null;
  authors?: string | null;
  doi?: string | null;
  venue?: string | null;
  citation_count?: number | null;
  page?: number | null;
  source_type?: string;
  url?: string | null;
  published_version?: { title?: string; doi?: string; year?: number; url?: string };
  published_version_status?: string;
  preprint_version?: { title?: string; year?: number; url?: string };
  metadata_verification?: string;
  canonical_title?: string;
  canonical_doi?: string;
  retraction_status?: string;
}

interface CitationClaim {
  claim_id: string;
  claim: string;
  status?: string;
  verdict: string;
  confidence: number;
  citation_context: string;
  action: string;
  evidence_excerpt?: string;
  evidence_level?: string;
  source?: CitationSource | null;
  candidate_sources: CitationSource[];
  note?: string;
  references?: AuditReference[];
}

interface CitationAudit {
  summary?: {
    total_claims: number;
    verified_claims: number;
    partially_supported_claims: number;
    uncited_claims: number;
    incorrect_references: number;
    source_mismatches: number;
    needs_review: number;
    assessment: string;
  };
  uncited_claim_count: number;
  incorrect_reference_count?: number;
  source_mismatch_count?: number;
  needs_review_count?: number;
  quality_hints: string[];
  claims: CitationClaim[];
  reading_list: Array<{ id: string; title: string; year?: number | null; passage?: string }>;
  references?: AuditReference[];
}

interface AuditReference {
  ref_number?: number | null;
  raw: string;
  title?: string | null;
  authors?: string | null;
  year?: number | null;
  venue?: string | null;
  doi?: string | null;
  url?: string | null;
  verification_status?: string;
  published_version?: { title?: string; doi?: string; year?: number; url?: string };
  published_version_status?: string;
}

interface CitationResponse {
  error?: string;
  draft_audit?: CitationAudit;
  claims?: CitationClaim[];
  document_name?: string;
}

interface EvidenceCandidate extends CitationSource {
  passage: string;
  support_score: number;
  support_level: string;
  verdict: string;
  evidence_excerpt: string;
  evidence_level: string;
  metadata_verification?: string;
  retraction_status?: string;
  note?: string;
}

interface WriteCheckResult {
  status?: string;
  verdict?: string;
  note?: string;
  evidence_excerpt?: string;
  source?: EvidenceCandidate;
  candidates?: EvidenceCandidate[];
  suggested_sources?: EvidenceCandidate[];
  search_warning?: string | null;
  metadata?: { published_version?: { title?: string; doi?: string; year?: number } | null };
}

function makeDocumentSegments(documentText: string, claims: CitationClaim[]) {
  const matches = claims
    .map(claim => {
      const exactStart = documentText.indexOf(claim.claim);
      if (exactStart >= 0) return { start: exactStart, end: exactStart + claim.claim.length, claim };
      const core = claim.claim.replace(/[.!?]["'”’)]*$/, "").trimEnd();
      const start = documentText.indexOf(core);
      if (start < 0) return null;
      const endPattern = /[.!?]["'”’)]*(?:\s|$)/g;
      endPattern.lastIndex = start + core.length;
      const ending = endPattern.exec(documentText);
      return ending
        ? { start, end: ending.index + ending[0].search(/[.!?]/) + 1, claim }
        : null;
    })
    .filter((match): match is { start: number; end: number; claim: CitationClaim } => match !== null)
    .sort((left, right) => left.start - right.start);
  const segments: Array<{ text: string; claim?: CitationClaim }> = [];
  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) continue;
    if (match.start > cursor) segments.push({ text: documentText.slice(cursor, match.start) });
    segments.push({ text: documentText.slice(match.start, match.end), claim: match.claim });
    cursor = match.end;
  }
  if (cursor < documentText.length) segments.push({ text: documentText.slice(cursor) });
  return segments;
}

export default function CitationScreen({ onBack }: { onBack: () => void }) {
  const documentInputRef = useRef<HTMLInputElement | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [style, setStyle] = useState("apa");
  const [draftText, setDraftText] = useState("");
  const [reviewDocumentText, setReviewDocumentText] = useState("");
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [documentPreviewUrl, setDocumentPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CitationResponse | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [claimDecisions, setClaimDecisions] = useState<Record<string, { action: string; source?: EvidenceCandidate; editedClaim?: string }>>({});
  const [activeClaimId, setActiveClaimId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<Record<string, EvidenceCandidate[]>>({});
  const [searchWarnings, setSearchWarnings] = useState<Record<string, string>>({});
  const [selectedClaims, setSelectedClaims] = useState<Set<string>>(new Set());
  const [searchingClaims, setSearchingClaims] = useState<Set<string>>(new Set());
  const [editingClaims, setEditingClaims] = useState<Set<string>>(new Set());
  const [claimEdits, setClaimEdits] = useState<Record<string, string>>({});
  const [writeText, setWriteText] = useState("");
  const [writeCitation, setWriteCitation] = useState("");
  const [writeEvidence, setWriteEvidence] = useState("");
  const [writeResult, setWriteResult] = useState<WriteCheckResult | null>(null);
  const [writeLoading, setWriteLoading] = useState(false);
  const [decisionError, setDecisionError] = useState("");

  useEffect(() => {
    getCurrentUserId().then(uid => {
      setUserId(uid);
      if (uid) getUserPapers(uid).then(({ data }) => setPapers(data.filter(p => p.status === "indexed")));
    });
  }, []);

  useEffect(() => {
    if (!activeClaimId) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setActiveClaimId(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [activeClaimId]);

  useEffect(() => {
    if (!documentPreviewUrl) return;
    return () => URL.revokeObjectURL(documentPreviewUrl);
  }, [documentPreviewUrl]);

  const selectDocumentFile = (file: File | null) => {
    setDocumentFile(file);
    setDocumentPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const run = async () => {
    if (!userId || (papers.length === 0 && !draftText.trim() && !documentFile)) return;
    setLoading(true); setResult(null);
    setClaimDecisions({});
    setSearchResults({});
    setSearchWarnings({});
    setSelectedClaims(new Set());
    setDecisionError("");
    try {
      let response: Response;
      const paperIds = papers.map(paper => paper.id);
      if (documentFile || draftText.trim()) {
        const text = documentFile
          ? await extractCitationDocumentText(documentFile)
          : draftText.trim();
        setReviewDocumentText(text);
        response = await fetch(`${API_BASE}/agents/citations/audit`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, paper_ids: paperIds, text }),
        });
      } else {
        response = await fetch(`${API_BASE}/agents/citations`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
          user_id: userId,
          paper_ids: paperIds,
          style,
          }),
        });
      }
      const payload = await response.json() as CitationResponse & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || `Error ${response.status}`);
      setResult(payload.draft_audit || payload.claims ? {
        ...payload,
        document_name: documentFile?.name || payload.document_name,
        draft_audit: payload.draft_audit || payload as CitationAudit,
      } : payload);
    } catch (error: unknown) {
      setResult({ error: error instanceof Error ? error.message : "Citation request failed." });
    }
    finally { setLoading(false); }
  };

  const recordDecision = async (
    claimId: string,
    claim: string,
    action: "accepted" | "rejected" | "edited",
    source?: EvidenceCandidate,
    editedClaim?: string,
  ): Promise<boolean> => {
    if (!userId) return false;
    setDecisionError("");
    try {
      const response = await fetch(`${API_BASE}/agents/citations/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          claim_id: claimId,
          claim,
          action,
          ...(source ? { source } : {}),
          ...(editedClaim ? { edited_claim: editedClaim } : {}),
        }),
      });
      const payload = await response.json() as { detail?: string };
      if (!response.ok) throw new Error(payload.detail || `Error ${response.status}`);
      setClaimDecisions(previous => ({
        ...previous,
        [claimId]: { action, ...(source ? { source } : {}), ...(editedClaim ? { editedClaim } : {}) },
      }));
      return true;
    } catch (error) {
      setDecisionError(error instanceof Error ? error.message : "Could not save your decision.");
      return false;
    }
  };

  const decideAuditEvidence = async (
    claim: CitationClaim,
    source: EvidenceCandidate,
    action: "accepted" | "rejected",
  ) => {
    const editedClaim = claimEdits[claim.claim_id] ?? claimDecisions[claim.claim_id]?.editedClaim ?? claim.claim;
    await recordDecision(claim.claim_id, claim.claim, action, source, editedClaim);
  };

  const saveClaimEdit = async (claim: CitationClaim) => {
    const editedText = claimEdits[claim.claim_id]?.trim();
    if (!editedText) return;
    const saved = await recordDecision(
      claim.claim_id,
      claim.claim,
      "edited",
      undefined,
      editedText,
    );
    if (!saved) return;
    setActiveClaimId(null);
  };

  const findEvidence = async (claim: CitationClaim) => {
    if (!userId) return;
    setSearchingClaims(previous => new Set(previous).add(claim.claim_id));
    setDecisionError("");
    try {
      const text = claimDecisions[claim.claim_id]?.editedClaim || claim.claim;
      const response = await fetch(`${API_BASE}/agents/citations/find-evidence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          claim_id: claim.claim_id,
          claim: text,
          paper_ids: papers.map(paper => paper.id),
        }),
      });
      const payload = await response.json() as { candidates?: EvidenceCandidate[]; search_warning?: string | null; detail?: string };
      if (!response.ok) throw new Error(payload.detail || `Error ${response.status}`);
      setSearchResults(previous => ({ ...previous, [claim.claim_id]: payload.candidates || [] }));
      setSearchWarnings(previous => ({
        ...previous,
        [claim.claim_id]: payload.search_warning || "",
      }));
    } catch (error) {
      setDecisionError(error instanceof Error ? error.message : "Evidence search failed.");
    } finally {
      setSearchingClaims(previous => {
        const next = new Set(previous);
        next.delete(claim.claim_id);
        return next;
      });
    }
  };

  const findSelectedEvidence = async (claims: CitationClaim[]) => {
    for (const claim of claims) {
      if (selectedClaims.has(claim.claim_id)) await findEvidence(claim);
    }
  };

  const runWriteCheck = async () => {
    if (!userId || !writeText.trim()) return;
    setWriteLoading(true);
    setWriteResult(null);
    setDecisionError("");
    try {
      const response = await fetch(`${API_BASE}/agents/citations/write-check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          text: writeText,
          ...(writeCitation.trim() ? { citation: writeCitation } : {}),
          ...(writeEvidence.trim() ? { evidence_text: writeEvidence } : {}),
          paper_ids: papers.map(paper => paper.id),
        }),
      });
      const payload = await response.json() as WriteCheckResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail || `Error ${response.status}`);
      setWriteResult(payload);
    } catch (error) {
      setDecisionError(error instanceof Error ? error.message : "Write-and-check failed.");
    } finally {
      setWriteLoading(false);
    }
  };

  const getAcceptedReferences = (): CitationSource[] => {
    const supportedDocumentReferences = (result?.draft_audit?.claims || [])
      .filter(claim => claim.status === "verified")
      .flatMap(claim => claim.references || [])
      .filter(reference => reference.verification_status === "verified")
      .map(reference => ({
        id: reference.doi || `REF-${reference.ref_number || reference.raw}`,
        title: reference.title || reference.raw,
        authors: reference.authors,
        year: reference.year,
        doi: reference.doi,
        venue: reference.venue,
        url: reference.url,
      }));
    const userAccepted = Object.values(claimDecisions)
      .filter(decision => decision.action === "accepted" && decision.source)
      .map(decision => decision.source as EvidenceCandidate);
    return [...new Map(
      [...supportedDocumentReferences, ...userAccepted].map(source => [
        source.doi?.toLowerCase() || source.title.toLowerCase(),
        source,
      ]),
    ).values()];
  };

  const downloadAcceptedReferences = async (format: "text" | "bibtex") => {
    const references = getAcceptedReferences();
    if (!references.length) return;
    try {
      const response = await fetch(`${API_BASE}/agents/citations/format`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ references, style }),
      });
      const payload = await response.json() as { citations_text: string; bibtex: string; detail?: string };
      if (!response.ok) throw new Error(payload.detail || `Error ${response.status}`);
      const download = (name: string, value: string, type: string) => {
        const url = URL.createObjectURL(new Blob([value], { type }));
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = name;
        anchor.click();
        URL.revokeObjectURL(url);
      };
      if (format === "text" && payload.citations_text) {
        download("accepted-references.txt", payload.citations_text, "text/plain;charset=utf-8");
      }
      if (format === "bibtex" && payload.bibtex) {
        download("accepted-references.bib", payload.bibtex, "application/x-bibtex;charset=utf-8");
      }
    } catch (error) {
      setDecisionError(error instanceof Error ? error.message : "Reference export failed.");
    }
  };

  const downloadOriginalDocument = () => {
    if (!documentFile) return;
    const url = URL.createObjectURL(documentFile);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = documentFile.name;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const downloadReviewReport = () => {
    const auditResult = result?.draft_audit;
    const claims = auditResult?.claims || [];
    const summary = auditResult?.summary;
    const documentName = documentFile?.name || result?.document_name || "Pasted text";
    const statusLabel = (value: string) => value.replace(/_/g, " ");
    const lines = [
      "# Citation Review Report",
      "",
      `- **Document:** ${documentName}`,
      `- **Generated:** ${new Date().toLocaleString()}`,
      `- **Claims detected:** ${summary?.total_claims ?? claims.length}`,
      `- **Verified against full text:** ${summary?.verified_claims ?? claims.filter(claim => claim.status === "verified").length}`,
      `- **Partially supported:** ${summary?.partially_supported_claims ?? claims.filter(claim => claim.verdict === "partially_supported").length}`,
      `- **References evaluated:** ${auditResult?.references?.length ?? 0}`,
      `- **Uncited:** ${summary?.uncited_claims ?? auditResult?.uncited_claim_count ?? 0}`,
      `- **Incorrect references:** ${summary?.incorrect_references ?? auditResult?.incorrect_reference_count ?? 0}`,
      `- **Source mismatches:** ${summary?.source_mismatches ?? auditResult?.source_mismatch_count ?? 0}`,
      `- **Requiring review:** ${summary?.needs_review ?? auditResult?.needs_review_count ?? 0}`,
      "",
      "## Overall assessment",
      summary?.assessment || "Review the claim-level findings below. Automated results require researcher judgment.",
      "",
      "## Scope and limitations",
      "- Claims are detected with local sentence-level rules; relevant claims may be missed.",
      "- Bibliographic matching relies on detectable in-text citations and a reference list.",
      "- Support is reported only when a source passage can be checked. Abstract-only evidence is not treated as full-text verification.",
      "- The audit does not automatically search for replacement sources. Researchers can request evidence search for individual claims.",
      "- This report is an aid to review, not a substitute for reading the original sources or assessing study quality.",
      "",
      "## Claim-level findings",
      ...(claims.length ? [] : ["No citation-sensitive claims were detected by the local rules."]),
      ...claims.flatMap((claim, index) => {
        const decision = claimDecisions[claim.claim_id];
        const source = decision?.source || claim.source;
        const sourceLabel = decision?.source
          ? decision.action === "accepted"
            ? "Accepted source"
            : decision.action === "rejected"
              ? "Rejected source"
              : "Researcher-selected source"
          : "Audited source";
        return [
          `### ${index + 1}. ${claim.claim}`,
          `- **Audit status:** ${statusLabel(claim.status || claim.verdict || "needs_review")}`,
          `- **Evidence assessment:** ${statusLabel(claim.verdict || "needs_review")}`,
          `- **Confidence:** ${Math.round((claim.confidence || 0) * 100)}%`,
          `- **Citation context:** ${claim.citation_context || "unknown"}`,
          ...(claim.evidence_level ? [`- **Evidence level:** ${statusLabel(claim.evidence_level)}`] : []),
          `- **Researcher decision:** ${statusLabel(decision?.action || "not reviewed")}`,
          ...(claim.note ? [`- **Audit note:** ${claim.note}`] : []),
          ...(claim.evidence_excerpt ? ["", `> Evidence excerpt: “${claim.evidence_excerpt}”`] : []),
          ...(claim.references?.length
            ? [
                "",
                "**Matched reference(s):**",
                ...claim.references.map(reference => {
                  const details = [
                    reference.authors,
                    reference.year ? `(${reference.year})` : "",
                    reference.venue,
                    reference.doi ? `DOI: ${reference.doi}` : "",
                    reference.url && !reference.doi ? reference.url : "",
                  ].filter(Boolean).join(". ");
                  return `- ${reference.ref_number ? `[${reference.ref_number}] ` : ""}${reference.title || reference.raw}${details ? `. ${details}` : ""} — ${statusLabel(reference.verification_status || "needs review")}`;
                }),
              ]
            : []),
          ...(source
            ? [
                "",
                `**${sourceLabel}:** ${source.title}${source.year ? ` (${source.year})` : ""}`,
                ...(source.authors ? [`**Authors:** ${source.authors}`] : []),
                ...(source.venue ? [`**Venue:** ${source.venue}`] : []),
                ...(source.doi ? [`**DOI:** ${source.doi}`] : []),
                ...(source.url ? [`**URL:** ${source.url}`] : []),
                ...(source.page ? [`**Page:** ${source.page}`] : []),
              ]
            : []),
          ...(decision?.editedClaim && decision.editedClaim !== claim.claim
            ? ["", `**Researcher-edited claim:** ${decision.editedClaim}`]
            : []),
          "",
        ];
      }),
      ...(acceptedReferences.length
        ? [
            "## Accepted references",
            ...acceptedReferences.map((source, index) =>
              `${index + 1}. ${source.authors ? `${source.authors}. ` : ""}${source.title}${source.year ? ` (${source.year})` : ""}${source.venue ? `. ${source.venue}` : ""}${source.doi ? `. https://doi.org/${source.doi}` : source.url ? `. ${source.url}` : ""}`,
            ),
          ]
        : []),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "citation-review-report.md";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const audit = result?.draft_audit;
  const activeClaim = audit?.claims.find(claim => claim.claim_id === activeClaimId) || null;
  const auditableClaims = (audit?.claims || []).filter(claim =>
    claim.status === "evidence_required" || claim.status === "source_mismatch",
  );
  const acceptedReferences = getAcceptedReferences();
  const documentSegments = makeDocumentSegments(reviewDocumentText, audit?.claims || []);
  const canRun = Boolean(draftText.trim()) || Boolean(documentFile) || papers.length > 0;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <div className="fade-1" style={{ marginBottom: 28 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", ...bodyText, fontSize: 13, color: C.gold, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>← Back to Agents</button>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ width: 44, height: 44, borderRadius: 4, background: `${C.umber}18`, border: `1px solid ${C.umber}33`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>◎</div>
          <div>
            <h1 style={{ ...headingStyle, fontSize: 28 }}>Citation <em style={{ color: C.umber }}>Review Agent</em></h1>
            <p style={{ ...bodyText, fontSize: 13, marginTop: 2 }}>Review document citations and find evidence for claims in a separate workspace</p>
          </div>
        </div>
      </div>

      {/* Citation Style Picker */}
      <div className="fade-2" style={{ ...cardBase, padding: "20px 24px", marginBottom: 16 }}>
        <div style={{ ...sectionLabel, marginBottom: 12 }}>Citation Style</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {STYLES.map(s => (
            <button key={s.key} onClick={() => setStyle(s.key)} style={{ padding: "12px 16px", textAlign: "left", background: style === s.key ? C.goldFaint : "transparent", border: `1.5px solid ${style === s.key ? C.borderGold : C.border}`, borderRadius: 4, cursor: "pointer", transition: "all .2s" }}>
              <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 14, fontWeight: 700, color: style === s.key ? C.gold : C.inkDark }}>{s.label}</div>
              <div style={{ ...bodyText, fontSize: 11.5 }}>{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="fade-2" style={{ ...cardBase, padding: "24px", marginBottom: 24 }}>
        <label htmlFor="citation-document" style={{ ...sectionLabel, display: "block", marginBottom: 8 }}>
          Document audit
        </label>
        <p style={{ ...bodyText, fontSize: 12, marginBottom: 10 }}>
          Choose a PDF, DOCX, or TXT file to extract its text in your browser, or paste text. The file is not uploaded or stored; extracted text is sent for audit and is not saved as a task result. Missing evidence is searched only after you request it for a claim.
        </p>
        <input
          ref={documentInputRef}
          id="citation-document"
          type="file"
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={event => {
            selectDocumentFile(event.target.files?.[0] || null);
            setDraftText("");
          }}
          style={{ ...bodyText, fontSize: 12, marginBottom: 10 }}
        />
        {documentFile && (
          <div style={{ ...bodyText, fontSize: 12, marginBottom: 10 }}>
            Selected: {documentFile.name} · {(documentFile.size / (1024 * 1024)).toFixed(2)} MB
            <button type="button" onClick={() => {
              selectDocumentFile(null);
              if (documentInputRef.current) documentInputRef.current.value = "";
            }} style={{ marginLeft: 10, cursor: "pointer" }}>Remove</button>
          </div>
        )}
        <textarea
          id="citation-draft"
          value={draftText}
          onChange={event => {
            setDraftText(event.target.value);
            selectDocumentFile(null);
            if (documentInputRef.current) documentInputRef.current.value = "";
          }}
          maxLength={30000}
          rows={8}
          placeholder="Or paste text for claim-by-claim evidence checking..."
          style={{
            width: "100%",
            resize: "vertical",
            boxSizing: "border-box",
            padding: 12,
            border: `1px solid ${C.border}`,
            borderRadius: 4,
            background: "transparent",
            color: C.inkDark,
            fontFamily: "inherit",
            fontSize: 13,
            lineHeight: 1.6,
          }}
        />
        <div style={{ ...bodyText, fontSize: 11, textAlign: "right", marginTop: 4 }}>
          {draftText.length.toLocaleString()} / 30,000
        </div>
      </div>

      <div className="fade-3" style={{ marginBottom: 32 }}>
        <button onClick={run} disabled={loading || !canRun} style={{ width: "100%", padding: "14px 0", background: loading || !canRun ? C.creamDark : C.inkDark, color: loading || !canRun ? C.inkLight : C.cream, border: "none", borderRadius: 4, fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", cursor: loading || !canRun ? "not-allowed" : "pointer", transition: "all .3s" }}>
          {loading ? "⟳ Reviewing document..." : "Run citation review"}
        </button>
      </div>

      {audit && (
        <div className="fade-1" style={{ ...cardBase, padding: 24, marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline", marginBottom: 12 }}>
            <h2 style={{ ...headingStyle, fontSize: 20 }}>{result?.document_name ? `Audit: ${result.document_name}` : "Citation audit"}</h2>
            <span style={{ ...bodyText, fontSize: 12 }}>{audit.summary?.total_claims ?? audit.claims.length} detected claim(s)</span>
          </div>
          <p style={{ ...bodyText, fontSize: 13, color: C.inkDark, lineHeight: 1.6, marginBottom: 12 }}>
            {audit.summary?.assessment || "Review the claim-level findings below. Automated results require researcher judgment."}
          </p>
          <p style={{ ...bodyText, fontSize: 12, marginBottom: 12 }}>
            {audit.summary?.verified_claims ?? audit.claims.filter(claim => claim.status === "verified").length} verified ·{" "}
            {audit.summary?.partially_supported_claims ?? audit.claims.filter(claim => claim.verdict === "partially_supported").length} partially supported ·{" "}
            {audit.summary?.uncited_claims ?? audit.uncited_claim_count ?? 0} uncited ·{" "}
            {audit.summary?.incorrect_references ?? audit.incorrect_reference_count ?? 0} incorrect references ·{" "}
            {audit.summary?.source_mismatches ?? audit.source_mismatch_count ?? 0} source mismatches ·{" "}
            {audit.summary?.needs_review ?? audit.needs_review_count ?? 0} need review
          </p>
          {auditableClaims.length > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 10 }}>
              <label style={{ ...bodyText, fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={auditableClaims.length > 0 && auditableClaims.every(claim => selectedClaims.has(claim.claim_id))}
                  onChange={event => setSelectedClaims(event.target.checked ? new Set(auditableClaims.map(claim => claim.claim_id)) : new Set())}
                />{" "}
                Select all flagged claims
              </label>
              <button
                type="button"
                disabled={selectedClaims.size === 0 || searchingClaims.size > 0}
                onClick={() => void findSelectedEvidence(auditableClaims)}
                style={{ cursor: selectedClaims.size ? "pointer" : "not-allowed" }}
              >
                {searchingClaims.size ? "Searching..." : `Find evidence for selected (${selectedClaims.size})`}
              </button>
            </div>
          )}
          {audit.quality_hints?.map((hint: string, index: number) => (
            <p key={index} style={{ ...bodyText, fontSize: 12, color: C.umber, marginBottom: 8 }}>• {hint}</p>
          ))}
          {reviewDocumentText && (
            <section style={{ margin: "20px 0", padding: "20px 0", borderTop: `1px solid ${C.border}` }}>
              <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <div>
                  <h3 style={{ ...headingStyle, fontSize: 18 }}>{documentFile ? "Original document" : "Document preview"}</h3>
                  <p style={{ ...bodyText, fontSize: 11 }}>
                    {documentFile
                      ? "The uploaded file is shown unchanged. Review decisions are saved separately and never alter its layout or content."
                      : "Highlighted claims open their review cards. The pasted source text remains unchanged."}
                  </p>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  {documentFile && (
                    <button type="button" onClick={downloadOriginalDocument}>Download original</button>
                  )}
                  <button type="button" onClick={downloadReviewReport}>Download review report</button>
                </div>
              </div>
              {documentFile?.name.toLowerCase().endsWith(".pdf") && documentPreviewUrl && (
                <iframe
                  title={`Original PDF: ${documentFile.name}`}
                  src={documentPreviewUrl}
                  style={{ display: "block", width: "100%", height: "min(80vh, 900px)", border: `1px solid ${C.border}`, background: "#fff" }}
                />
              )}
              {documentFile?.name.toLowerCase().endsWith(".txt") && documentPreviewUrl && (
                <iframe
                  title={`Original text document: ${documentFile.name}`}
                  src={documentPreviewUrl}
                  style={{ display: "block", width: "100%", height: "min(70vh, 700px)", border: `1px solid ${C.border}`, background: "#fff" }}
                />
              )}
              {documentFile?.name.toLowerCase().endsWith(".docx") && (
                <div style={{ ...cardBase, padding: 20 }}>
                  <p style={{ ...bodyText, fontSize: 13, marginBottom: 10 }}>
                    DOCX formatting is not reconstructed in this page. Download and open the original file to view its exact layout.
                  </p>
                  <strong style={{ ...bodyText, fontSize: 12 }}>{documentFile.name}</strong>
                </div>
              )}
              {!documentFile && (
                <div style={{
                  maxWidth: 760,
                  minHeight: 240,
                  margin: "0 auto",
                  padding: "clamp(20px, 4vw, 48px)",
                  background: "#fffefa",
                  border: `1px solid ${C.border}`,
                  boxShadow: `0 5px 20px ${C.shadow}`,
                  color: C.inkDark,
                  fontFamily: "'Crimson Pro', Georgia, serif",
                  fontSize: 15,
                  lineHeight: 1.8,
                  whiteSpace: "pre-wrap",
                  overflowWrap: "anywhere",
                }}>
                  {documentSegments.map((segment, index) => segment.claim ? (
                    <button
                      key={`${segment.claim.claim_id}-${index}`}
                      type="button"
                      aria-haspopup="dialog"
                      aria-label={`Review claim: ${segment.text}`}
                      title="Click to review this claim"
                      onClick={() => setActiveClaimId(segment.claim!.claim_id)}
                      style={{
                        display: "inline",
                        padding: 0,
                        border: 0,
                        borderRadius: 2,
                        background: `${C.goldLight}66`,
                        color: "inherit",
                        font: "inherit",
                        lineHeight: "inherit",
                        textAlign: "left",
                        cursor: "pointer",
                        boxShadow: `inset 0 -2px 0 ${C.goldLight}`,
                      }}
                    >
                      {segment.text}
                    </button>
                  ) : <span key={`text-${index}`}>{segment.text}</span>)}
                </div>
              )}
              {documentFile && (audit.claims || []).length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4 style={{ ...headingStyle, fontSize: 15, marginBottom: 8 }}>Claims to review</h4>
                  <div style={{ display: "grid", gap: 8 }}>
                    {(audit.claims || []).map(claim => (
                      <button
                        key={claim.claim_id}
                        type="button"
                        aria-haspopup="dialog"
                        onClick={() => setActiveClaimId(claim.claim_id)}
                        style={{
                          padding: 12,
                          textAlign: "left",
                          color: C.inkDark,
                          background: "#fffefa",
                          border: `1px solid ${C.border}`,
                          cursor: "pointer",
                        }}
                      >
                        <span style={{ display: "block", ...bodyText, fontSize: 12 }}>{claim.claim}</span>
                        <span style={{ display: "block", ...sectionLabel, marginTop: 6 }}>
                          {String(claim.status || claim.verdict || "needs_review").replace(/_/g, " ")}
                          {claimDecisions[claim.claim_id] ? ` · decision: ${claimDecisions[claim.claim_id].action}` : ""}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}
          {(audit.claims || []).map(claim => {
            const decision = claimDecisions[claim.claim_id];
            const candidates = searchResults[claim.claim_id] || [];
            const canFind = claim.status === "evidence_required" || claim.status === "source_mismatch";
            const editedText = claimEdits[claim.claim_id] ?? decision?.editedClaim ?? claim.claim;
            const statusColor = claim.status === "verified"
              ? C.gold
              : claim.status === "incorrect_reference" || claim.status === "source_mismatch"
                ? C.sienna
                : C.umber;
            return (
              <article id={`claim-${claim.claim_id}`} key={claim.claim_id} style={{ borderTop: `1px solid ${C.border}`, borderLeft: `3px solid ${statusColor}`, padding: "14px 12px", marginTop: 8, background: `${statusColor}08`, scrollMarginTop: 84 }}>
                {editingClaims.has(claim.claim_id) ? (
                  <textarea
                    aria-label="Edit claim"
                    value={editedText}
                    onChange={event => setClaimEdits(previous => ({ ...previous, [claim.claim_id]: event.target.value }))}
                    rows={3}
                    style={{ width: "100%", boxSizing: "border-box", padding: 10, color: C.inkDark, background: "transparent" }}
                  />
                ) : (
                  <p style={{ ...bodyText, color: C.inkDark, fontSize: 13, lineHeight: 1.6 }}>{editedText}</p>
                )}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, margin: "8px 0", ...bodyText, fontSize: 11 }}>
                  <strong style={{ color: statusColor }}>{String(claim.status || claim.verdict || "needs_review").replace(/_/g, " ")}</strong>
                  {claim.verdict !== claim.status && <span>Support: {String(claim.verdict || "unknown").replace(/_/g, " ")}</span>}
                  <span>Confidence: {Math.round((claim.confidence || 0) * 100)}%</span>
                  <span>Use: {claim.citation_context || "unknown"}</span>
                  {decision && <strong>Decision: {decision.action}</strong>}
                </div>
                {claim.evidence_excerpt && (
                  <blockquote style={{ ...bodyText, margin: "8px 0", paddingLeft: 12, borderLeft: `2px solid ${C.umber}`, fontSize: 12 }}>
                    “{claim.evidence_excerpt}”
                    {claim.source?.title && (
                      <div style={{ marginTop: 4 }}>
                        — {claim.source.title}
                        {claim.source.authors ? ` · ${claim.source.authors}` : ""}
                        {claim.source.year ? ` · ${claim.source.year}` : ""}
                        {claim.source.venue ? ` · ${claim.source.venue}` : ""}
                        {claim.source.citation_count != null ? ` · ${claim.source.citation_count} citations` : ""}
                        {claim.source.page ? ` · p. ${claim.source.page}` : ""}
                      </div>
                    )}
                    {claim.source?.published_version && (
                      <div style={{ marginTop: 4, color: C.umber }}>
                        Possible published version: {claim.source.published_version.title}
                        {claim.source.published_version.year ? ` (${claim.source.published_version.year})` : ""}
                        {claim.source.published_version.doi ? ` · DOI ${claim.source.published_version.doi}` : ""}
                      </div>
                    )}
                    {claim.source?.metadata_verification && (
                      <div style={{ marginTop: 4, color: claim.source.retraction_status === "retracted" ? C.sienna : C.umber }}>
                        Metadata: {claim.source.metadata_verification.replace(/_/g, " ")}
                        {claim.source.retraction_status === "retracted" ? " · RETRACTED" : ""}
                        {claim.source.canonical_title ? ` · Crossref title: ${claim.source.canonical_title}` : ""}
                      </div>
                    )}
                  </blockquote>
                )}
                {claim.note && <p style={{ ...bodyText, fontSize: 12, margin: "8px 0" }}>{claim.note}</p>}
                {searchWarnings[claim.claim_id] && (
                  <p role="status" style={{ ...bodyText, fontSize: 12, color: C.sienna, margin: "8px 0" }}>
                    Search warning: {searchWarnings[claim.claim_id]}
                  </p>
                )}
                {searchResults[claim.claim_id]?.length === 0 && !searchWarnings[claim.claim_id] && (
                  <p role="status" style={{ ...bodyText, fontSize: 12, margin: "8px 0" }}>
                    No readable candidate passages were found for this claim.
                  </p>
                )}
                {claim.references?.map((reference, index) => (
                  <div key={index} style={{ ...bodyText, fontSize: 11, margin: "4px 0" }}>
                    <p>Reference: {reference.raw} · {String(reference.verification_status || "needs review").replace(/_/g, " ")}</p>
                    {reference.published_version && (
                      <p style={{ color: C.umber }}>
                        Published version may exist: {reference.published_version.title}
                        {reference.published_version.year ? ` (${reference.published_version.year})` : ""}
                        {reference.published_version.doi ? ` · DOI ${reference.published_version.doi}` : ""}
                      </p>
                    )}
                  </div>
                ))}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginTop: 8 }}>
                  {canFind && (
                    <>
                      <label style={{ ...bodyText, fontSize: 11 }}>
                        <input
                          type="checkbox"
                          checked={selectedClaims.has(claim.claim_id)}
                          onChange={event => setSelectedClaims(previous => {
                            const next = new Set(previous);
                            if (event.target.checked) next.add(claim.claim_id);
                            else next.delete(claim.claim_id);
                            return next;
                          })}
                        /> Select for batch search
                      </label>
                      <button type="button" disabled={searchingClaims.has(claim.claim_id)} onClick={() => void findEvidence(claim)}>
                        {searchingClaims.has(claim.claim_id) ? "Searching..." : "Find evidence"}
                      </button>
                    </>
                  )}
                  {editingClaims.has(claim.claim_id) ? (
                    <button
                      type="button"
                      disabled={!editedText.trim()}
                      onClick={() => {
                        void recordDecision(claim.claim_id, claim.claim, "edited", undefined, editedText.trim());
                        setEditingClaims(previous => {
                          const next = new Set(previous);
                          next.delete(claim.claim_id);
                          return next;
                        });
                      }}
                    >
                      Save edited claim
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setEditingClaims(previous => new Set(previous).add(claim.claim_id))}
                    >
                      Edit claim
                    </button>
                  )}
                </div>
                {candidates.length > 0 && (
                  <div style={{ marginTop: 12, paddingLeft: 12, borderLeft: `2px solid ${C.borderGold}` }}>
                    <h3 style={{ ...sectionLabel, marginBottom: 8 }}>Ranked evidence candidates</h3>
                    {candidates.map(candidate => (
                      <div key={candidate.id} style={{ borderTop: `1px solid ${C.border}`, padding: "10px 0" }}>
                        <strong style={{ ...bodyText, color: C.inkDark, fontSize: 12 }}>
                          {candidate.title}{candidate.year ? ` (${candidate.year})` : ""}
                        </strong>
                        <span style={{ ...bodyText, fontSize: 11 }}> · {candidate.support_score}% · {candidate.support_level.replace(/_/g, " ")} · {candidate.evidence_level.replace(/_/g, " ")}</span>
                        {(candidate.metadata_verification || candidate.retraction_status) && (
                          <p style={{ ...bodyText, fontSize: 11, color: candidate.retraction_status === "retracted" || candidate.metadata_verification === "metadata_mismatch" ? C.sienna : C.umber }}>
                            Metadata: {String(candidate.metadata_verification || "not checked").replace(/_/g, " ")}
                            {candidate.retraction_status ? ` · retraction: ${candidate.retraction_status.replace(/_/g, " ")}` : ""}
                          </p>
                        )}
                        {candidate.evidence_excerpt && (
                          <blockquote style={{ ...bodyText, margin: "6px 0", paddingLeft: 10, borderLeft: `2px solid ${C.umber}`, fontSize: 12 }}>
                            “{candidate.evidence_excerpt}”
                          </blockquote>
                        )}
                        {candidate.note && <p style={{ ...bodyText, fontSize: 11, marginBottom: 6 }}>{candidate.note}</p>}
                        {candidate.published_version && (
                          <p style={{ ...bodyText, fontSize: 11, color: C.umber }}>
                            Possible published version: {candidate.published_version.title}
                            {candidate.published_version.doi ? ` · DOI ${candidate.published_version.doi}` : ""}
                          </p>
                        )}
                        <div style={{ display: "flex", gap: 8 }}>
                          <button
                            type="button"
                            onClick={() => void decideAuditEvidence(claim, candidate, "accepted")}
                          >
                            Accept evidence
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              void decideAuditEvidence(claim, candidate, "rejected");
                              setSearchResults(previous => ({
                                ...previous,
                                [claim.claim_id]: (previous[claim.claim_id] || []).filter(item => item.id !== candidate.id),
                              }));
                            }}
                          >
                            Reject for this claim
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            );
          })}
          {audit.reading_list?.length > 0 && (
            <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 12 }}>
              <h3 style={{ ...sectionLabel, marginBottom: 8 }}>Prioritized reading list</h3>
              {audit.reading_list.map(item => (
                <p key={item.id} style={{ ...bodyText, fontSize: 12, marginBottom: 8 }}>
                  {item.title}{item.year ? ` (${item.year})` : ""}{item.passage ? ` — “${item.passage}”` : ""}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {decisionError && (
        <div role="alert" style={{ ...cardBase, padding: 14, marginBottom: 16, color: C.sienna }}>
          {decisionError}
        </div>
      )}

      {activeClaim && (
        <div
          role="presentation"
          onMouseDown={event => {
            if (event.target === event.currentTarget) setActiveClaimId(null);
          }}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            padding: 16,
            background: "rgba(25, 18, 10, 0.62)",
            backdropFilter: "blur(3px)",
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="claim-review-title"
            style={{
              width: "min(760px, 100%)",
              maxHeight: "min(88vh, 900px)",
              overflowY: "auto",
              padding: "clamp(20px, 4vw, 32px)",
              border: `1px solid ${C.borderGold}`,
              borderRadius: 6,
              background: C.creamLight,
              boxShadow: `0 24px 80px rgba(0,0,0,.32)`,
            }}
          >
            <header style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start", marginBottom: 18 }}>
              <div>
                <div style={{ ...sectionLabel, marginBottom: 6 }}>In-document citation review</div>
                <h2 id="claim-review-title" style={{ ...headingStyle, fontSize: 22 }}>Review this claim</h2>
              </div>
              <button
                type="button"
                aria-label="Close claim review"
                onClick={() => setActiveClaimId(null)}
                style={{ border: `1px solid ${C.border}`, borderRadius: 4, padding: "6px 10px", background: "transparent", color: C.inkDark, cursor: "pointer" }}
              >
                Close
              </button>
            </header>

            <div style={{ padding: 16, borderLeft: `3px solid ${C.gold}`, background: C.goldFaint, marginBottom: 18 }}>
              <label htmlFor="active-claim-edit" style={{ ...sectionLabel, display: "block", marginBottom: 8 }}>Claim text</label>
              <textarea
                id="active-claim-edit"
                value={claimEdits[activeClaim.claim_id] ?? claimDecisions[activeClaim.claim_id]?.editedClaim ?? activeClaim.claim}
                onChange={event => setClaimEdits(previous => ({ ...previous, [activeClaim.claim_id]: event.target.value }))}
                rows={3}
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding: 10,
                  border: `1px solid ${C.border}`,
                  borderRadius: 4,
                  background: C.white,
                  color: C.inkDark,
                  font: "inherit",
                  lineHeight: 1.6,
                }}
              />
              <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
                <button type="button" onClick={() => void saveClaimEdit(activeClaim)}>Save claim edit</button>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <div>
                <h3 style={{ ...headingStyle, fontSize: 17 }}>Evidence</h3>
                <p style={{ ...bodyText, fontSize: 12 }}>Search indexed library papers and review each passage before deciding.</p>
              </div>
              <button
                type="button"
                disabled={searchingClaims.has(activeClaim.claim_id)}
                onClick={() => void findEvidence(activeClaim)}
              >
                {searchingClaims.has(activeClaim.claim_id) ? "Searching..." : "Find evidence"}
              </button>
            </div>

            {searchWarnings[activeClaim.claim_id] && (
              <p role="status" style={{ ...bodyText, color: C.sienna, fontSize: 12, marginBottom: 10 }}>
                {searchWarnings[activeClaim.claim_id]}
              </p>
            )}
            {(searchResults[activeClaim.claim_id] || []).length === 0 && !searchingClaims.has(activeClaim.claim_id) && (
              <p style={{ ...bodyText, fontSize: 13, padding: 16, border: `1px dashed ${C.border}`, borderRadius: 4 }}>
                No evidence candidates yet. Choose “Find evidence” to search indexed library papers.
              </p>
            )}
            {(searchResults[activeClaim.claim_id] || []).map(candidate => (
              <article key={candidate.id} style={{ border: `1px solid ${C.border}`, borderRadius: 4, padding: 14, marginTop: 10, background: C.white }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start" }}>
                  <div>
                    <h4 style={{ ...headingStyle, fontSize: 16 }}>{candidate.title}{candidate.year ? ` (${candidate.year})` : ""}</h4>
                    <p style={{ ...bodyText, fontSize: 11, marginTop: 4 }}>
                      {candidate.authors || "Author unavailable"}
                      {candidate.venue ? ` · ${candidate.venue}` : ""}
                      {candidate.page ? ` · p. ${candidate.page}` : ""}
                    </p>
                  </div>
                  <span style={{ ...sectionLabel, whiteSpace: "nowrap", color: candidate.support_level === "strong" ? C.green : C.umber }}>
                    {candidate.support_score}% · {candidate.support_level.replace(/_/g, " ")}
                  </span>
                </div>
                {candidate.evidence_excerpt && (
                  <blockquote style={{ ...bodyText, margin: "12px 0", paddingLeft: 12, borderLeft: `2px solid ${C.gold}`, fontSize: 13, color: C.inkDark }}>
                    “{candidate.evidence_excerpt}”
                  </blockquote>
                )}
                {candidate.note && <p style={{ ...bodyText, fontSize: 12, marginBottom: 10 }}>{candidate.note}</p>}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  <button
                    type="button"
                    onClick={() => void decideAuditEvidence(activeClaim, candidate, "accepted")}
                  >
                    Accept evidence
                  </button>
                  <button
                    type="button"
                    onClick={() => void decideAuditEvidence(activeClaim, candidate, "rejected")}
                    style={{ color: C.sienna }}
                  >
                    Reject for this claim
                  </button>
                </div>
              </article>
            ))}

            {claimDecisions[activeClaim.claim_id] && (
              <p role="status" style={{ ...bodyText, color: C.green, fontSize: 12, marginTop: 14 }}>
                Decision saved: {claimDecisions[activeClaim.claim_id].action.replace(/_/g, " ")}. The source document remains unchanged; the decision is included in the review report.
              </p>
            )}
            {decisionError && <p role="alert" style={{ ...bodyText, color: C.sienna, fontSize: 12, marginTop: 12 }}>{decisionError}</p>}
          </section>
        </div>
      )}

      <div className="fade-2" style={{ ...cardBase, padding: 24, marginBottom: 20 }}>
        <h2 style={{ ...headingStyle, fontSize: 20, marginBottom: 8 }}>Write-and-check</h2>
        <p style={{ ...bodyText, fontSize: 12, marginBottom: 10 }}>
          Add a citation to verify it, or leave it blank to search for sources. Results are suggestions only; choose one to accept it.
        </p>
        <textarea
          aria-label="Text to check"
          value={writeText}
          onChange={event => setWriteText(event.target.value)}
          maxLength={10000}
          rows={4}
          placeholder="Paste a claim or passage to check..."
          style={{ width: "100%", boxSizing: "border-box", padding: 12, marginBottom: 8, color: C.inkDark, background: "transparent" }}
        />
        <input
          aria-label="Optional citation"
          value={writeCitation}
          onChange={event => setWriteCitation(event.target.value)}
          maxLength={3000}
          placeholder="Optional citation (DOI, author-year, or reference entry)"
          style={{ width: "100%", boxSizing: "border-box", padding: 10, marginBottom: 8, color: C.inkDark, background: "transparent" }}
        />
        <textarea
          aria-label="Optional pasted evidence"
          value={writeEvidence}
          onChange={event => setWriteEvidence(event.target.value)}
          maxLength={5000}
          rows={2}
          placeholder="Optional: paste the source evidence to check for a misquote"
          style={{ width: "100%", boxSizing: "border-box", padding: 10, marginBottom: 8, color: C.inkDark, background: "transparent" }}
        />
        <button type="button" disabled={writeLoading || !writeText.trim()} onClick={() => void runWriteCheck()}>
          {writeLoading ? "Checking..." : "Check text"}
        </button>
        {writeResult && (
          <div style={{ borderTop: `1px solid ${C.border}`, marginTop: 16, paddingTop: 12 }}>
            <strong>{String(writeResult.status || writeResult.verdict || "needs_review").replace(/_/g, " ")}</strong>
            {writeResult.note && <p style={{ ...bodyText, fontSize: 12, margin: "6px 0" }}>{writeResult.note}</p>}
            {writeResult.metadata?.published_version && (
              <p style={{ ...bodyText, fontSize: 12, color: C.umber }}>
                Possible published version: {writeResult.metadata.published_version.title}
                {writeResult.metadata.published_version.year ? ` (${writeResult.metadata.published_version.year})` : ""}
                {writeResult.metadata.published_version.doi ? ` · DOI ${writeResult.metadata.published_version.doi}` : ""}
              </p>
            )}
            {writeResult.search_warning && <p role="status" style={{ ...bodyText, fontSize: 12, color: C.sienna }}>{writeResult.search_warning}</p>}
            {(writeResult.candidates || writeResult.suggested_sources)?.length === 0 && !writeResult.search_warning && (
              <p role="status" style={{ ...bodyText, fontSize: 12 }}>No readable candidate passages were found.</p>
            )}
            {writeResult.evidence_excerpt && (
              <blockquote style={{ ...bodyText, margin: "8px 0", paddingLeft: 12, borderLeft: `2px solid ${C.umber}`, fontSize: 12 }}>
                “{writeResult.evidence_excerpt}”
                {writeResult.source?.title && <div>— {writeResult.source.title}{writeResult.source.year ? ` (${writeResult.source.year})` : ""}</div>}
              </blockquote>
            )}
            {(writeResult.candidates || writeResult.suggested_sources || []).map(candidate => (
              <div key={candidate.id} style={{ borderTop: `1px solid ${C.border}`, padding: "10px 0" }}>
                <strong style={{ ...bodyText, fontSize: 12 }}>{candidate.title}{candidate.year ? ` (${candidate.year})` : ""}</strong>
                <span style={{ ...bodyText, fontSize: 11 }}> · {candidate.support_score}% · {candidate.support_level.replace(/_/g, " ")} · {candidate.evidence_level.replace(/_/g, " ")}</span>
                {(candidate.metadata_verification || candidate.retraction_status) && (
                  <p style={{ ...bodyText, fontSize: 11, color: candidate.retraction_status === "retracted" || candidate.metadata_verification === "metadata_mismatch" ? C.sienna : C.umber }}>
                    Metadata: {String(candidate.metadata_verification || "not checked").replace(/_/g, " ")}
                    {candidate.retraction_status ? ` · retraction: ${candidate.retraction_status.replace(/_/g, " ")}` : ""}
                  </p>
                )}
                {candidate.evidence_excerpt && <blockquote style={{ ...bodyText, margin: "6px 0", paddingLeft: 10, borderLeft: `2px solid ${C.umber}`, fontSize: 12 }}>“{candidate.evidence_excerpt}”</blockquote>}
                <button type="button" onClick={() => void recordDecision("write-check", writeText, "accepted", candidate)}>
                  Accept this source
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {acceptedReferences.length > 0 && (
        <div className="fade-1" style={{ ...cardBase, padding: 24, marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <h2 style={{ ...headingStyle, fontSize: 20 }}>Accepted references</h2>
            <div style={{ display: "flex", gap: 8 }}>
              <button type="button" onClick={() => void downloadAcceptedReferences("text")}>Download cleaned references</button>
              <button type="button" onClick={() => void downloadAcceptedReferences("bibtex")}>Download .bib</button>
            </div>
          </div>
          <p style={{ ...bodyText, fontSize: 12, margin: "8px 0" }}>
            Includes references that passed checks and sources you accepted during review. The original document is not modified.
          </p>
          {acceptedReferences.map(source => (
            <p key={`${source.id}-${source.doi || source.title}`} style={{ ...bodyText, fontSize: 12, borderTop: `1px solid ${C.border}`, paddingTop: 8 }}>
              {source.authors ? `${source.authors}. ` : ""}{source.title}{source.year ? ` (${source.year})` : ""}
              {source.doi ? ` · DOI ${source.doi}` : ""}
              {source.retraction_status === "retracted" && <strong style={{ color: C.sienna }}> · RETRACTED</strong>}
            </p>
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
