"""The three systems the synopsis compares, behind one result type so the evaluation harness
never needs to know which arm produced an answer:

    baseline      LLM alone, no retrieval
    standard_rag  retrieve -> rerank -> LLM answers from the passages, no verification
    verified_rag  the full pipeline (retrieve -> rerank -> draft -> decompose -> verify -> final)

All three share the same configured generator, retriever and reranker so the only thing
that differs between arms is what the synopsis says should differ.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legalrag.evidence import Evidence, Status
from legalrag.llm import LLMError, LLMRequest, parse_structured
from legalrag.pipeline.generate import format_evidence, label_evidence, source_label
from legalrag.pipeline.orchestrate import Pipeline, PipelineResult
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.pipeline.query_analysis import analyze_query
from legalrag.pipeline.rerank import rerank

ARMS = ("baseline", "standard_rag", "verified_rag")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class ArmAnswer(BaseModel):
    """What the baseline and standard-RAG prompts ask the model to return."""

    model_config = ConfigDict(extra="ignore")

    abstain: bool = False
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def _unit(cls, v: float | None) -> float | None:
        return None if v is None else min(1.0, max(0.0, float(v)))


@dataclass
class ArmResult:
    arm: str
    query: str
    outcome: str  # answered | abstained | error
    answer_text: str = ""
    statements: list[str] = field(default_factory=list)  # units the auditor judges, one per sentence
    citations: list[str] = field(default_factory=list)  # authorities the answer relies on, as strings
    confidence: float | None = None
    confidence_source: str = "none"  # verification | model_stated | none
    caveated: bool = False  # the answer itself says some of it rests on FLAGGED evidence
    retrieved: list[Evidence] = field(default_factory=list)  # ranked candidates
    evidence_pool: list[Evidence] = field(default_factory=list)  # what the answer was allowed to rest on
    latency_ms: dict[str, float] = field(default_factory=dict)
    run_id: int | None = None
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def split_statements(text: str, min_chars: int = 12) -> list[str]:
    """Sentence split shared by every arm, so claim-level metrics compare like with like."""
    return [s.strip() for s in _SENTENCE.split(text.strip()) if len(s.strip()) >= min_chars]


def _ask(pipe: Pipeline, task: str, prompt: str, pid: str, res: ArmResult) -> ArmAnswer | None:
    resp = pipe.llm.complete(LLMRequest(task=task, prompt=prompt, json_output=True, prompt_id=pid))
    parsed, err = parse_structured(resp.text, ArmAnswer)
    if parsed is None:
        # A model that will not follow the format is a model failure, not a missing answer:
        # keep what it wrote so the audit sees it.
        res.notes["parse_error"] = err
        res.answer_text, res.confidence_source = resp.text.strip(), "none"
        return None
    return parsed


def _finish_free_text(res: ArmResult, parsed: ArmAnswer | None) -> None:
    if parsed is not None:
        if parsed.abstain or not parsed.answer.strip():
            res.outcome = "abstained"
            return
        res.answer_text, res.citations = parsed.answer.strip(), list(parsed.citations)
        res.confidence = parsed.confidence
        res.confidence_source = "model_stated" if parsed.confidence is not None else "none"
    res.outcome = "answered" if res.answer_text else "abstained"
    res.statements = split_statements(res.answer_text)


def run_baseline(pipe: Pipeline, query: str, experiment: str = "adhoc") -> ArmResult:
    res = ArmResult("baseline", query, "error")
    t0 = time.perf_counter()
    try:
        pid = PROMPT_IDS["baseline"]
        _finish_free_text(res, _ask(pipe, "baseline", render(pid, query=query), pid, res))
    except LLMError as e:
        res.error = str(e)
    res.latency_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
    _log(pipe, res, experiment)
    return res


def run_standard_rag(pipe: Pipeline, query: str, experiment: str = "adhoc") -> ArmResult:
    res = ArmResult("standard_rag", query, "error")
    t0 = time.perf_counter()
    try:
        analysis = analyze_query(pipe.llm, query).analysis
        cands = pipe.retriever.retrieve(analysis).candidates
        res.retrieved, kept = rerank(cands, analysis, pipe.scorer, pipe.cfg.rerank)
        res.evidence_pool = kept
        if not kept:
            res.outcome = "abstained"
        else:
            pid = PROMPT_IDS["rag_answer"]
            prompt = render(pid, evidence=format_evidence(label_evidence(kept)), query=query)
            _finish_free_text(res, _ask(pipe, "rag_answer", prompt, pid, res))
    except LLMError as e:
        res.error = str(e)
    res.latency_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
    _log(pipe, res, experiment)
    return res


def run_verified_rag(pipe: Pipeline, query: str, experiment: str = "adhoc") -> ArmResult:
    r: PipelineResult = pipe.run(query, experiment)  # the pipeline logs itself
    res = ArmResult("verified_rag", query, r.outcome, retrieved=r.retrieved, latency_ms=r.latency_ms,
                    run_id=r.run_id, error=r.error)
    if r.final and not r.final.abstained:
        res.statements = [s["text"] for s in r.final.statements]
        res.answer_text = " ".join(res.statements)
        cited_claims = {i for s in r.final.statements for i in s["claim_ids"]}
        pool: dict[int, Evidence] = {}
        for c in r.claims:
            if c.idx in cited_claims:
                want = Status.FLAGGED if c.support == "flagged" else Status.VERIFIED
                for e in c.evidence:
                    if e.status == want:
                        pool.setdefault(e.chunk_id, e)
        res.evidence_pool = list(pool.values())
        # What the user is shown as the authorities behind the answer.
        res.citations = list(dict.fromkeys(src for s in r.final.statements for src in s.get("sources", [])))
        res.confidence, res.confidence_source = r.final.confidence, "verification"
        res.caveated = any(s.get("status") == "flagged" for s in r.final.statements)
        res.notes["statements"] = r.final.statements
    res.notes["n_claims"] = len(r.claims)
    res.notes["n_claims_surviving"] = sum(bool(c.survives) for c in r.claims)
    return res


def _log(pipe: Pipeline, res: ArmResult, experiment: str) -> None:
    final = None if res.outcome != "answered" else {
        "abstained": False, "statements": [{"text": t, "claim_ids": []} for t in res.statements],
        "citations": res.citations, "confidence": res.confidence, "confidence_source": res.confidence_source}
    res.run_id = pipe.runs.insert_run(
        experiment=experiment, config_name=f"{pipe.cfg.name}:{res.arm}", config_hash=pipe.cfg.config_hash,
        corpus_sha=pipe.corpus_sha, model_info=pipe.model_info, query=res.query, outcome=res.outcome,
        final_answer=final, trace={"arm": res.arm, "error": res.error, **res.notes}, latency_ms=res.latency_ms)
    for rank, ev in enumerate(res.retrieved, start=1):
        pipe.runs.insert_item(res.run_id, "retrieved", rank, ev.to_dict())
    for rank, ev in enumerate(res.evidence_pool, start=1):
        pipe.runs.insert_item(res.run_id, "reranked", rank, ev.to_dict())
    pipe.conn.commit()


RUNNERS = {"baseline": run_baseline, "standard_rag": run_standard_rag, "verified_rag": run_verified_rag}

__all__ = ["ARMS", "ArmResult", "RUNNERS", "run_baseline", "run_standard_rag", "run_verified_rag",
           "split_statements", "source_label"]
