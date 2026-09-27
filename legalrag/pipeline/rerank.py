"""Stage 3: rerank all candidates, keep the top N. Every candidate's score is retained.

`score = base + boosts`. The base scorer sits behind `PairScorer`: a lexical overlap
stand-in for offline runs, or a cross-encoder, with the boost logic unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from legalrag.evidence import Evidence
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import RerankConfig

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split())


class PairScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class LexicalOverlapScorer:
    """Fraction of distinct query terms present in the passage. Deterministic stand-in
    for the cross-encoder; carries no semantic signal."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = {w for w in _WORD.findall(query.lower()) if w not in _STOP}
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(_WORD.findall(t.lower()))) / len(q) for t in texts]


class CrossEncoderScorer:
    """Cross-encoder relevance (optional dependency: `pip install -e ".[ml]"`). Raw logits
    are squashed to (0, 1) so the additive metadata boosts, which are on a 0..1 scale, keep
    a stable meaning whichever model is plugged in."""

    def __init__(self, cfg: RerankConfig, model=None):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover
                raise RuntimeError('sentence-transformers is not installed: pip install -e ".[ml]"') from e
            model = CrossEncoder(cfg.model, device=cfg.device)
        self._model, self._batch = model, cfg.batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._model.predict([(query, t) for t in texts], batch_size=self._batch, show_progress_bar=False)
        return [float(1.0 / (1.0 + math.exp(-float(x)))) for x in raw]


def build_scorer(cfg: RerankConfig) -> PairScorer:
    if cfg.kind == "lexical_fake":
        return LexicalOverlapScorer()
    if cfg.kind == "cross_encoder":
        return CrossEncoderScorer(cfg)
    raise ValueError(f"unknown reranker kind {cfg.kind!r}")


def _norm_name(s: str) -> str:
    return " ".join(_WORD.findall(s.lower().replace("(synthetic)", "")))


def boosts(ev: Evidence, analysis: QueryAnalysis, cfg: RerankConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    if analysis.section_numbers:
        wanted = set(analysis.section_numbers)
        mentioned = set(re.findall(r"\b[Ss]ection\s+(\d+[A-Z]?)", ev.text))
        if ev.section_number in wanted or (wanted & mentioned):
            out["section_match"] = cfg.boost_section_match
    if ev.case_name and analysis.case_names:
        title = _norm_name(ev.case_name)
        if any(_norm_name(n) and (_norm_name(n) in title or title in _norm_name(n))
               for n in analysis.case_names):
            out["case_name_match"] = cfg.boost_case_name_match
    if ev.section_label in ("holding", "reasoning"):
        out["holding_or_reasoning"] = cfg.boost_holding_reasoning
    elif ev.section_label == "arguments":
        out["arguments_penalty"] = -cfg.penalty_arguments
    return out


def rerank(candidates: list[Evidence], analysis: QueryAnalysis, scorer: PairScorer,
           cfg: RerankConfig) -> tuple[list[Evidence], list[Evidence]]:
    """Returns (all candidates sorted best-first with rerank_score set, kept top-N)."""
    base = scorer.score(analysis.rewritten_query, [c.text for c in candidates])
    for ev, b in zip(candidates, base, strict=True):
        ev.rerank_score = b + sum(boosts(ev, analysis, cfg).values())
    ranked = sorted(candidates, key=lambda e: (-(e.rerank_score or 0.0), e.chunk_id))
    return ranked, ranked[: cfg.keep]
