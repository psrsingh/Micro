"""The single object that flows through every pipeline stage and into the logs.

Every evaluation metric must be derivable from serialised Evidence alone. If a metric
needs something that is not here, add a field; do not recompute it inside a stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class Outcome(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class CheckResult:
    name: str
    outcome: Outcome
    score: float  # 0..1
    evidence_pointer: str | None = None  # e.g. "chunk:12[45:210]" or "statute_section:7"
    explanation: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"check {self.name!r}: score {self.score} outside [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "outcome": str(self.outcome)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckResult:
        return cls(**{**d, "outcome": Outcome(d["outcome"])})


@dataclass
class Evidence:
    chunk_id: int
    document_id: int
    text: str
    span: tuple[int, int]  # (char_start, char_end) into the source document

    case_name: str | None = None
    citation: str | None = None
    court: str | None = None
    decided_on: str | None = None  # ISO date
    act_name: str | None = None
    section_number: str | None = None
    # Structural section of the source ("holding", "reasoning", "arguments", ...).
    # Not in the original spec's field list but the claim_support check and the
    # reranker both need it, and it must be on the logged object.
    section_label: str | None = None

    # Location in the original document and provenance (real corpus; None for the synthetic
    # fixtures). They let an answer cite "Section 103(1), p. 34" and let the reader see whether the
    # text came from an official or a supplementary copy.
    doc_uid: str | None = None
    subsection: str | None = None
    heading: str | None = None
    paragraph_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    verification_status: str | None = None
    source_authority: str | None = None

    bm25_score: float | None = None
    dense_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    checks: dict[str, CheckResult] = field(default_factory=dict)
    status: Status | None = None
    status_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        d["status"] = str(self.status) if self.status else None
        d["checks"] = {k: v.to_dict() for k, v in self.checks.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        d = dict(d)
        d["span"] = tuple(d["span"])
        d["status"] = Status(d["status"]) if d.get("status") else None
        d["checks"] = {k: CheckResult.from_dict(v) for k, v in d.get("checks", {}).items()}
        return cls(**d)


@dataclass
class Claim:
    """An atomic proposition from the (never shown) draft, with the evidence it cites."""

    idx: int
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    # True if the draft attached citations that could not be mapped to any retrieved
    # evidence. Keeping the count avoids silently dropping a fabricated citation.
    unresolved_citations: list[str] = field(default_factory=list)
    survives: bool | None = None
    # "verified" when the claim rests on VERIFIED evidence, "flagged" when it survives only
    # because `include_flagged` let FLAGGED evidence through (then `caveats` says why).
    support: str | None = None
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "text": self.text,
            "evidence": [e.to_dict() for e in self.evidence],
            "unresolved_citations": self.unresolved_citations,
            "survives": self.survives,
            "support": self.support,
            "caveats": self.caveats,
        }
