"""Check `claim_support`: does the passage actually establish the claim?

The judgment itself sits behind `SupportJudge` so the same check can run with an offline
heuristic (`LexicalSupportJudge`) or with the configured generator (`LLMSupportJudge`). The
evaluation harness reuses these judges to audit answers, which is why they live here.

Two guards apply whichever judge is used:
  * a claim resting on an 'arguments' section (what a party contended, not what the court
    held) is capped at WARN;
  * an LLM verdict of supported/partial must come with a quote that really occurs in the
    passage; a quote that does not is treated as a failed grounding and downgrades the
    verdict, so a judge cannot vouch for text it invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from legalrag.evidence import CheckResult, Claim, Evidence, Outcome
from legalrag.llm import LLMAdapter, LLMError, LLMRequest, parse_structured
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.verify.authority import pointer
from legalrag.verify.extract import find_act_mentions, find_case_refs, find_section_refs

Label = Literal["supported", "partial", "unsupported", "contradicted", "error"]
_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "contradicted": 0.0, "error": 0.0}

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split())
_NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty sixty ninety hundred".split())


def _terms(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


def _numbers(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w.isdigit() or w in _NUMBER_WORDS}


def strip_references(claim: str) -> str:
    """Remove case names, citations, Act names and 'Section N' from a claim. They point at
    an authority; they are not part of the proposition being tested, and a passage does not
    repeat them."""
    spans = [(r.start, r.end) for r in find_case_refs(claim)]
    for r in find_section_refs(claim):
        # 'Section 12 of the X Act' is one reference: take the connecting words with it.
        spans.append((min(r.start, r.act.start), max(r.end, r.act.end)) if r.act else (r.start, r.end))
    spans += [(a.start, a.end) for a in find_act_mentions(claim)]
    out, last = [], 0
    for a, b in sorted(spans):
        if a >= last:
            out.append(claim[last:a])
            last = b
    out.append(claim[last:])
    return " ".join("".join(out).split())


@dataclass
class SupportVerdict:
    label: Label
    score: float
    reason: str = ""
    quote_span: tuple[int, int] | None = None  # offsets into the passage
    kind: str = "direct"  # "inferred": the claim goes beyond the passage's words (judge prompt v2)


class SupportJudge(Protocol):
    name: str

    def judge(self, claim: str, passage: str) -> SupportVerdict: ...


class LexicalSupportJudge:
    """Content-word overlap, plus one guard that matters in law: a number in the claim
    ('three months', '30 days') that the passage does not contain caps the verdict at
    partial. Deterministic and offline; it measures overlap, not entailment."""

    name = "lexical"

    def __init__(self, pass_at: float = 0.5, warn_at: float = 0.3, can_verify: bool = True):
        self.pass_at, self.warn_at, self.can_verify = pass_at, warn_at, can_verify

    def judge(self, claim: str, passage: str) -> SupportVerdict:
        claim = strip_references(claim)
        c, p = _terms(claim), _terms(passage)
        overlap = len(c & p) / len(c) if c else 0.0
        label: Label = "supported" if overlap >= self.pass_at else "partial" if overlap >= self.warn_at else "unsupported"
        reason = f"word overlap {overlap:.2f}"
        missing = _numbers(claim) - _numbers(passage)
        if missing and label == "supported":
            label, reason = "partial", f"{reason}; claim states {sorted(missing)} which the passage does not"
        if label == "supported" and not self.can_verify:
            label, reason = "partial", f"{reason}; word overlap is similarity, not entailment, so it cannot verify a claim"
        return SupportVerdict(label, min(overlap, _SCORE[label] or overlap), reason)


class SupportJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Literal["supported", "partial", "unsupported", "contradicted"]
    support_kind: Literal["direct", "inferred"] = "direct"  # only asked for by judge prompt v2
    quote: str = ""
    reason: str = ""


def locate_quote(quote: str, passage: str) -> tuple[int, int] | None:
    """Offsets of `quote` in `passage`, tolerant of whitespace and case differences."""
    quote = quote.strip()
    if not quote:
        return None
    i = passage.find(quote)
    if i != -1:
        return i, i + len(quote)
    pat = r"\s+".join(re.escape(w) for w in quote.split())
    m = re.search(pat, passage, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


class LLMSupportJudge:
    name = "llm"

    def __init__(self, llm: LLMAdapter, prompt_version: str = "v1"):
        self.llm = llm
        self.prompt_id = f"judge@{prompt_version}"

    def judge(self, claim: str, passage: str) -> SupportVerdict:
        pid = self.prompt_id
        try:
            resp = self.llm.complete(LLMRequest(task="judge", prompt=render(pid, claim=claim, passage=passage),
                                                json_output=True, prompt_id=pid, max_tokens=300))
        except LLMError as e:
            return SupportVerdict("error", 0.0, f"judge call failed: {e}")
        parsed, err = parse_structured(resp.text, SupportJudgment)
        if parsed is None:
            return SupportVerdict("error", 0.0, f"judge output unparseable: {err}")
        span = locate_quote(parsed.quote, passage)
        label: Label = parsed.verdict
        reason = parsed.reason
        if label in ("supported", "partial") and span is None:
            label, reason = "partial" if label == "supported" else "unsupported", \
                f"{reason} [downgraded: the judge's quote does not occur in the passage]"
        kind = parsed.support_kind
        if label == "supported" and kind == "inferred":
            label, reason = "partial", f"{reason} [interpretation inferred from the passage, not stated in it]"
        return SupportVerdict(label, _SCORE[label], reason, span, kind)


class ClaimSupportCheck:
    name = "claim_support"

    def __init__(self, judge: SupportJudge):
        self.judge = judge

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        v = self.judge.judge(claim.text, evidence.text)
        ptr = pointer(evidence)
        if v.quote_span:
            ptr = f"chunk:{evidence.chunk_id}[{evidence.span[0] + v.quote_span[0]}:{evidence.span[0] + v.quote_span[1]}]"
        explanation = f"[{self.judge.name}] {v.label}: {v.reason}".rstrip(": ")
        if v.label == "error":
            return CheckResult(self.name, Outcome.WARN, 0.0, ptr, explanation)
        if evidence.section_label == "arguments" and v.label in ("supported", "partial"):
            return CheckResult(self.name, Outcome.WARN, min(v.score, 0.4), ptr,
                               f"{explanation}; source is an 'arguments' section, not the court's view")
        outcome = {"supported": Outcome.PASS, "partial": Outcome.WARN}.get(v.label, Outcome.FAIL)
        return CheckResult(self.name, outcome, v.score, ptr, explanation)
