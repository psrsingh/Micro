"""The evidence-based benchmark: each record stores not just a reference answer but the gold
authority, the exact supporting passage and what a correct answer must (and must not) say,
so evaluation can separate "right answer" from "right answer for the right reason".

JSONL, one record per line. `answerable: false` records are questions the corpus cannot
answer (out of scope, or about a case we do not hold): a correct system abstains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legalrag.authority import norm_citation, norm_name, norm_ws
from legalrag.ingest.corpus_view import CorpusView

ERROR_CATEGORIES = (
    "fabricated_citation", "incorrect_metadata", "wrong_section", "real_case_wrong_proposition",
    "unsupported_reasoning", "outdated_authority", "conflicting_authority",
    # behaviour classes beyond the synopsis list
    "missed_answer", "answered_out_of_scope", "wrong_conclusion",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldAuthority(_Strict):
    kind: Literal["case", "statute"]
    citation: str | None = None  # case: neutral citation
    case_name: str | None = None
    act: str | None = None  # statute
    section: str | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.kind == "case" and not (self.citation or self.case_name):
            raise ValueError("a case authority needs a citation or a case_name")
        if self.kind == "statute" and not (self.act and self.section):
            raise ValueError("a statute authority needs act and section")
        return self

    @property
    def key(self) -> str:
        return norm_citation(self.citation) if self.kind == "case" and self.citation else \
            f"{norm_name(self.act or '')}|{(self.section or '').upper()}" if self.kind == "statute" else \
            norm_name(self.case_name or "")


class GoldPassage(_Strict):
    authority: str  # a case citation, or "Act name|section"
    text: str  # verbatim snippet from the corpus
    # Real-corpus location of the passage in the original document (checked against the ingested chunks).
    page: str | None = None       # PDF page, or "52-53" when the passage crosses a page break
    section: str | None = None    # "103" or "103(2)"
    paragraph: str | None = None  # judgment paragraph number


class Forbidden(_Strict):
    phrase: str
    category: Literal[ERROR_CATEGORIES]  # type: ignore[valid-type]
    # The phrase is only an error if none of these appear (e.g. "two months" is fine when the
    # answer explains it was the pre-amendment cap).
    unless_any: list[str] = Field(default_factory=list)


class ClaimLabel(_Strict):
    text: str
    label: Literal["supported", "unsupported", "contradicted"]


CATEGORIES = (
    "statutory_fact", "section_specific", "judgment_based", "multi_document", "amendment",
    "outdated_or_superseded", "unanswerable", "hallucination_probe",
)


class BenchmarkItem(_Strict):
    id: str
    question: str
    legal_issue: str = ""
    answerable: bool = True
    act: str | None = None
    section: str | None = None
    court: str | None = None
    gold_authorities: list[GoldAuthority] = Field(default_factory=list)
    passages: list[GoldPassage] = Field(default_factory=list)
    current_status: Literal["current", "superseded_text_trap", "not_applicable"] = "current"
    expected_conclusion: str = ""
    # Every group must be satisfied by at least one of its alternatives (case-insensitive).
    must_include: list[list[str]] = Field(default_factory=list)
    must_not_include: list[Forbidden] = Field(default_factory=list)
    claim_labels: list[ClaimLabel] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # -- real-corpus fields (all optional, so the synthetic benchmark is unchanged) ------------
    category: Literal[CATEGORIES] | None = None  # type: ignore[valid-type]
    difficulty: Literal["easy", "medium", "hard"] | None = None
    document_ids: list[str] = Field(default_factory=list)   # manifest doc_uids of the gold documents
    source_citation: str | None = None                      # exact citation of the gold source(s)
    section_ref: str | None = None
    paragraph_ref: str | None = None
    page_ref: str | None = None
    # `pending_review`: the gold passage is verified verbatim by code, but the expected answer was written by
    # the assistant and has NOT been checked by a person. Only `human_verified` items count as ground truth.
    review_status: Literal["not_applicable", "pending_review", "human_verified"] = "not_applicable"
    verified_by: str | None = None
    verified_on: str | None = None
    verification_notes: str = ""

    @model_validator(mode="after")
    def _reviewed(self):
        if self.review_status == "human_verified" and not (self.verified_by and self.verified_on):
            raise ValueError(f"{self.id}: human_verified needs verified_by and verified_on")
        return self

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and not (self.gold_authorities and self.passages and self.must_include):
            raise ValueError(f"{self.id}: an answerable item needs gold_authorities, passages and must_include")
        if not self.answerable and (self.gold_authorities or self.must_include):
            raise ValueError(f"{self.id}: an unanswerable item must not list gold authorities or must_include")
        return self


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip() and not line.lstrip().startswith("#"):
                try:
                    items.append(BenchmarkItem.model_validate(json.loads(line)))
                except Exception as e:  # noqa: BLE001 - report the line, whatever the parser said
                    raise ValueError(f"{path}:{n}: {e}") from e
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def validate_against_corpus(items: list[BenchmarkItem], view: CorpusView) -> list[str]:
    """Every gold authority must exist and every gold passage must occur verbatim in it."""
    problems: list[str] = []
    for it in items:
        for g in it.gold_authorities:
            if g.kind == "case":
                if view.judgment(g.citation or g.case_name or "") is None:
                    problems.append(f"{it.id}: gold case {g.citation or g.case_name!r} not in corpus")
            elif not view.index.versions(g.act or "", g.section or ""):
                problems.append(f"{it.id}: gold statute {g.act} s.{g.section} not in corpus")
        keys = {g.key for g in it.gold_authorities}
        for p in it.passages:
            if p.authority.count("|"):
                act, _, sec = p.authority.partition("|")
                pool = view.passages_for_section(act, sec)
                pkey = f"{norm_name(act)}|{sec.upper()}"
            else:
                pool = view.passages_for_case(p.authority)
                pkey = norm_citation(p.authority)
            if pkey not in keys:
                problems.append(f"{it.id}: passage authority {p.authority!r} is not a gold authority of the item")
            if not any(norm_ws(p.text) in norm_ws(t) for t in pool):
                problems.append(f"{it.id}: passage not found verbatim in {p.authority!r}: {p.text[:50]!r}")
    return problems
