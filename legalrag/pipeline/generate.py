"""Stages 4 and 7: draft generation (never shown) and final grounded generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalrag.evidence import Claim, Evidence, Outcome, Status
from legalrag.llm import LLMAdapter, LLMRequest, parse_structured
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.pipeline.schemas import FinalAnswer

_WS = re.compile(r"\s+")

DISCLAIMER = (
    "Research assistant, not legal advice. Statements are checked against retrieved text; a checked statement is "
    "not a legal opinion. Judgments from the mirror are unofficial copies, and section text may not reflect "
    "amendments after the Gazette copy. Verify against the official source and consult a qualified lawyer.")

# What kind of text a piece of evidence is. The verifier treats them differently (party arguments are not
# the court's view; a headnote is the reporter's summary, not the judgment), and so should the reader.
SOURCE_KINDS = {
    "statute_section": "statutory_text", "headnote": "reporter_headnote", "arguments": "party_arguments",
    "facts": "factual_or_procedural_background", "reasoning": "court_reasoning", "holding": "court_holding",
    "judgment_text": "judgment_text_unclassified",
}


def source_kind(e: Evidence) -> str:
    return SOURCE_KINDS.get(e.section_label or "", "unclassified")


def pinpoint(e: Evidence) -> str:
    """Where in the source: section/sub-section, paragraph, PDF page(s). Empty for fixtures."""
    bits = []
    if e.section_number:
        bits.append(f"s. {e.section_number}" + (f"({e.subsection})" if e.subsection else ""))
    if e.paragraph_number:
        bits.append(f"para {e.paragraph_number}")
    if e.page_start:
        bits.append(f"p. {e.page_start}" + (f"-{e.page_end}" if e.page_end and e.page_end != e.page_start else ""))
    return ", ".join(bits)


def source_detail(e: Evidence) -> dict:
    return {"label": source_label(e), "kind": source_kind(e), "pinpoint": pinpoint(e), "doc_uid": e.doc_uid,
            "verification_status": e.verification_status, "source_authority": e.source_authority}


def label_evidence(evidence: list[Evidence]) -> dict[str, Evidence]:
    """Stable labels E1..En in rerank order; the draft cites by label."""
    return {f"E{i}": e for i, e in enumerate(evidence, start=1)}


def format_evidence(labelled: dict[str, Evidence]) -> str:
    lines = []
    for label, e in labelled.items():
        who = e.case_name or (f"{e.act_name} s.{e.section_number}" if e.act_name else "unknown source")
        meta = f"{who}; {e.citation or 'no citation'}; {e.section_label or 'unlabelled'}"
        lines.append(f"[{label}] {meta} :: {_WS.sub(' ', e.text).strip()}")
    return "\n".join(lines)


def draft_answer(llm: LLMAdapter, query: str, labelled: dict[str, Evidence]) -> str:
    pid = PROMPT_IDS["draft"]
    prompt = render(pid, evidence=format_evidence(labelled), query=query)
    return llm.complete(LLMRequest(task="draft", prompt=prompt, prompt_id=pid)).text.strip()


def source_label(e: Evidence) -> str:
    if e.case_name:
        return f"{e.case_name} ({e.citation})" if e.citation else e.case_name
    # Written the way a claim would cite it, so the same extractors can read it back.
    return f"Section {e.section_number} of {e.act_name}" if e.act_name else "unknown source"


def evidence_confidence(e: Evidence) -> float:
    """Weakest link: the lowest score among the checks that actually ran."""
    scores = [r.score for r in e.checks.values() if r.outcome in (Outcome.PASS, Outcome.WARN)]
    return min(scores) if scores else 0.0


def claim_confidence(c: Claim) -> float:
    """Best VERIFIED evidence; FLAGGED-only claims are halved so caveated answers never
    look as certain as clean ones."""
    verified = [evidence_confidence(e) for e in c.evidence if e.status == Status.VERIFIED]
    if verified:
        return max(verified)
    flagged = [evidence_confidence(e) for e in c.evidence if e.status == Status.FLAGGED]
    return 0.5 * max(flagged) if flagged else 0.0


@dataclass
class FinalResult:
    abstained: bool
    # {"text", "claim_ids", "status": verified|flagged, "caveats": [...], "sources": [...]}
    statements: list[dict] = field(default_factory=list)
    used_fallback: str | None = None  # why we bypassed the model output, if we did
    llm_called: bool = False
    # Verification-derived, not model-stated: the weakest claim behind the answer. None when
    # abstaining.
    confidence: float | None = None
    disclaimer: str = DISCLAIMER  # added in code, never by the model


def _annotate(statements: list[dict], surviving: list[Claim]) -> float | None:
    """Attach status, caveats and sources to each statement from the claims it cites. Done in
    code so a caveat can never be dropped or reworded by the generator."""
    by_id = {c.idx: c for c in surviving}
    used: set[int] = set()
    for st in statements:
        cited = [by_id[i] for i in st["claim_ids"]]
        used.update(c.idx for c in cited)
        st["status"] = "flagged" if any(c.support == "flagged" for c in cited) else "verified"
        st["caveats"] = list(dict.fromkeys(cv for c in cited for cv in c.caveats))
        want = Status.FLAGGED if st["status"] == "flagged" else Status.VERIFIED
        st["sources"] = list(dict.fromkeys(source_label(e) for c in cited for e in c.evidence if e.status == want))
        details = [source_detail(e) for c in cited for e in c.evidence if e.status == want]
        st["source_details"] = list({(d["label"], d["pinpoint"], d["kind"]): d for d in details}.values())
        st["source_types"] = sorted({d["kind"] for d in details})
        # the statement itself is always model-written text built from the verified claims
        st["origin"] = "model_generated_from_verified_claims"
    return min((claim_confidence(by_id[i]) for i in used), default=None)


def final_answer(llm: LLMAdapter, query: str, surviving: list[Claim]) -> FinalResult:
    """Regenerate from verified claims and their evidence only.

    No survivors means abstain without calling the model at all: abstention is decided by
    the verification layer, not left to the generator's discretion.
    """
    if not surviving:
        return FinalResult(abstained=True)
    lines = []
    for c in surviving:
        lines.append(f"C{c.idx}: {c.text}")
        for e in c.evidence:
            if e.status == (Status.FLAGGED if c.support == "flagged" else Status.VERIFIED):
                src = e.case_name or f"{e.act_name} s.{e.section_number}"
                lines.append(f"    source: {src} ({e.citation or 'no citation'}), {e.section_label}")
    pid = PROMPT_IDS["final"]
    resp = llm.complete(LLMRequest(task="final", prompt=render(pid, query=query, claims="\n".join(lines)),
                                   json_output=True, prompt_id=pid))
    parsed, err = parse_structured(resp.text, FinalAnswer)
    valid_ids = {c.idx for c in surviving}
    if parsed is None:
        # Deterministic, still-grounded fallback: the verified claims themselves.
        stmts = [{"text": c.text, "claim_ids": [c.idx]} for c in surviving]
        return FinalResult(False, stmts, used_fallback=f"final parse failed: {err}", llm_called=True,
                           confidence=_annotate(stmts, surviving))
    if parsed.abstain:
        return FinalResult(True, llm_called=True)
    kept = [s for s in parsed.statements if s.claim_ids and set(s.claim_ids) <= valid_ids]
    dropped = len(parsed.statements) - len(kept)
    if not kept:
        return FinalResult(True, used_fallback="all statements lacked a valid claim citation",
                           llm_called=True)
    note = f"dropped {dropped} statement(s) without a valid claim citation" if dropped else None
    stmts = [s.model_dump() for s in kept]
    return FinalResult(False, stmts, used_fallback=note, llm_called=True, confidence=_annotate(stmts, surviving))
