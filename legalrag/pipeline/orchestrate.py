"""Pipeline orchestration: query analysis -> retrieval -> rerank -> draft -> decompose
-> verify -> final. One `runs` row per query, evidence rows per stage.

    python -m legalrag.pipeline.orchestrate --config phase0 --query "..."
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import psycopg

from legalrag.authority import AuthorityIndex
from legalrag.embed import Embedder, build_embedder
from legalrag.evidence import Claim, Evidence
from legalrag.llm import LLMAdapter, LLMError, build_llm
from legalrag.pipeline.decompose import decompose
from legalrag.pipeline.generate import FinalResult, draft_answer, final_answer, label_evidence
from legalrag.pipeline.query_analysis import analyze_query
from legalrag.pipeline.rerank import PairScorer, build_scorer, rerank
from legalrag.pipeline.retrieval import HybridRetriever
from legalrag.settings import Config, load_config
from legalrag.store.db import connect
from legalrag.store.repos import AuthorityRepo, DocumentRepo, RunRepo
from legalrag.verify.build import build_verifier
from legalrag.verify.query_check import unavailable_authorities
from legalrag.verify.verifier import Verifier


@dataclass
class PipelineResult:
    query: str
    outcome: str  # answered | abstained | error
    final: FinalResult | None = None
    claims: list[Claim] = field(default_factory=list)
    retrieved: list[Evidence] = field(default_factory=list)  # every candidate, rerank order
    kept: list[Evidence] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    run_id: int | None = None
    error: str | None = None


class Pipeline:
    def __init__(self, cfg: Config, conn: psycopg.Connection, llm: LLMAdapter,
                 embedder: Embedder, scorer: PairScorer, verifier: Verifier,
                 authority_index: AuthorityIndex | None = None):
        self.cfg, self.conn, self.llm = cfg, conn, llm
        self.authority_index = authority_index
        self.retriever = HybridRetriever(conn, embedder, cfg.retrieval)
        self.scorer, self.verifier = scorer, verifier
        self.runs = RunRepo(conn)
        self.corpus_sha = DocumentRepo(conn).corpus_sha()
        # Identity of everything that can change an answer, recorded on every run.
        self.model_info = {
            "llm": llm.describe(),
            "embedding": {"kind": cfg.embedding.kind, "model": cfg.embedding.model, "dim": cfg.embedding.dim},
            "reranker": {"kind": cfg.rerank.kind, "model": cfg.rerank.model},
            "verifier": {"kind": cfg.verify.kind, "enabled_checks": cfg.verify.enabled_checks,
                         "support_judge": cfg.verify.support_judge if cfg.verify.kind == "real" else "fake",
                         "as_of": str(cfg.verify.as_of) if cfg.verify.as_of else "today",
                         "include_flagged": cfg.verify.include_flagged},
        }

    @classmethod
    def from_config(cls, cfg: Config, conn: psycopg.Connection, llm: LLMAdapter | None = None) -> Pipeline:
        llm = llm or build_llm(cfg.llm)
        index = AuthorityRepo(conn).load() if cfg.verify.kind == "real" else None
        return cls(cfg, conn, llm, build_embedder(cfg.embedding), build_scorer(cfg.rerank),
                   build_verifier(cfg.verify, index, llm), index)

    @contextmanager
    def _stage(self, res: PipelineResult, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            res.latency_ms[name] = round((time.perf_counter() - t0) * 1000, 2)

    def run(self, query: str, experiment: str = "adhoc") -> PipelineResult:
        res = PipelineResult(query=query, outcome="error")
        t_total = time.perf_counter()
        try:
            self._run(res)
        except LLMError as e:
            res.outcome, res.error = "error", str(e)
            res.trace["error"] = str(e)
        res.latency_ms["total"] = round((time.perf_counter() - t_total) * 1000, 2)
        self._log(res, experiment)
        return res

    def _run(self, res: PipelineResult) -> None:
        cfg, trace = self.cfg, res.trace
        with self._stage(res, "query_analysis"):
            qa = analyze_query(self.llm, res.query)
        trace["query_analysis"] = {**qa.analysis.model_dump(mode="json"), "fallback_reason": qa.fallback_reason}

        if self.authority_index is not None and cfg.pipeline.abstain_on_unavailable_authority:
            missing = unavailable_authorities(self.authority_index, res.query, qa.analysis.case_names)
            if missing:
                # Nothing we could retrieve would be about the authority that was asked for.
                trace["unavailable_authorities"] = missing
                res.final = FinalResult(abstained=True, used_fallback="question names an authority the corpus cannot ground: "
                                        + "; ".join(missing))
                res.outcome = "abstained"
                trace["final"] = {"llm_called": False, "note": res.final.used_fallback,
                                  "n_claims_total": 0, "n_claims_surviving": 0}
                return

        with self._stage(res, "retrieval"):
            rr = self.retriever.retrieve(qa.analysis)
        trace["retrieval"] = {"n_candidates": len(rr.candidates), "filters_applied": rr.filters_applied,
                              "filters_dropped": rr.filters_dropped}

        with self._stage(res, "rerank"):
            res.retrieved, res.kept = rerank(rr.candidates, qa.analysis, self.scorer, cfg.rerank)

        labelled = label_evidence(res.kept)
        with self._stage(res, "draft"):
            draft = draft_answer(self.llm, res.query, labelled) if labelled else ""
        trace["draft"] = draft  # logged for diagnosis; never shown to the user

        with self._stage(res, "decompose"):
            dec = decompose(self.llm, draft, labelled, cfg.pipeline.max_claims) if draft else None
        res.claims = dec.claims if dec else []
        trace["decompose"] = {"n_claims": len(res.claims), "fallback_reason": dec.fallback_reason if dec else None}

        with self._stage(res, "verify"):
            self.verifier.verify(res.claims)

        surviving = [c for c in res.claims if c.survives]
        with self._stage(res, "final"):
            res.final = final_answer(self.llm, res.query, surviving)
        trace["final"] = {"llm_called": res.final.llm_called, "note": res.final.used_fallback,
                          "n_claims_total": len(res.claims), "n_claims_surviving": len(surviving)}
        res.outcome = "abstained" if res.final.abstained else "answered"

    def _log(self, res: PipelineResult, experiment: str) -> None:
        trace = {**res.trace, "claims": [
            {"idx": c.idx, "text": c.text, "survives": c.survives,
             "evidence_chunk_ids": [e.chunk_id for e in c.evidence],
             "unresolved_citations": c.unresolved_citations} for c in res.claims]}
        final = None
        if res.final is not None:
            final = {"abstained": res.final.abstained, "statements": res.final.statements,
                     "confidence": res.final.confidence, "disclaimer": res.final.disclaimer}
        res.run_id = self.runs.insert_run(
            experiment=experiment, config_name=self.cfg.name, config_hash=self.cfg.config_hash,
            corpus_sha=self.corpus_sha, model_info=self.model_info, query=res.query,
            outcome=res.outcome, final_answer=final, trace=trace, latency_ms=res.latency_ms)
        for rank, ev in enumerate(res.retrieved, start=1):
            self.runs.insert_item(res.run_id, "retrieved", rank, ev.to_dict())
        for rank, ev in enumerate(res.kept, start=1):
            self.runs.insert_item(res.run_id, "reranked", rank, ev.to_dict())
        for c in res.claims:
            for rank, ev in enumerate(c.evidence, start=1):
                self.runs.insert_item(res.run_id, "verified", rank, ev.to_dict(), claim_idx=c.idx)
        self.conn.commit()  # a per-query record survives a later crash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--query", action="append", required=True)
    ap.add_argument("--experiment", default="adhoc")
    args = ap.parse_args()
    cfg = load_config(args.config)
    with connect(cfg) as conn:
        pipe = Pipeline.from_config(cfg, conn)
        for q in args.query:
            r = pipe.run(q, args.experiment)
            print(json.dumps({
                "run_id": r.run_id, "query": q, "outcome": r.outcome, "error": r.error,
                "confidence": r.final.confidence if r.final else None,
                "answer": r.final.statements if r.final else None,
                "disclaimer": r.final.disclaimer if r.final else None,
                "claims": [{"idx": c.idx, "text": c.text, "survives": c.survives,
                            "statuses": [e.status.value if e.status else None for e in c.evidence]}
                           for c in r.claims],
                "latency_ms": r.latency_ms,
            }, indent=2))


if __name__ == "__main__":
    main()
