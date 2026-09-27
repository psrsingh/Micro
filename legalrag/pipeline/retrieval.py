"""Stage 2: hybrid retrieval. Lexical + dense, metadata-filtered, fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import psycopg

from legalrag.embed import Embedder
from legalrag.evidence import Evidence
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import RetrievalConfig
from legalrag.store.repos import ChunkRepo, MetadataFilter


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Each ranking is a best-first list of ids. Ties on the
    fused score break by id so results are deterministic."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def evidence_from_row(row: dict[str, Any]) -> Evidence:
    is_statute = row["doc_type"] == "statute"
    return Evidence(
        chunk_id=row["chunk_id"], document_id=row["document_id"], text=row["text"],
        span=(row["char_start"], row["char_end"]),
        case_name=None if is_statute else row["title"],
        citation=row["neutral_citation"], court=row["court"],
        decided_on=row["decided_on"].isoformat() if row["decided_on"] else None,
        act_name=row["act_name"], section_number=row["section_number"],
        section_label=row["section_label"], doc_uid=row.get("doc_uid"), subsection=row.get("subsection"),
        heading=row.get("heading"), paragraph_number=row.get("paragraph_number"),
        page_start=row.get("page_start"), page_end=row.get("page_end"),
        verification_status=row.get("verification_status"), source_authority=row.get("source_authority"),
    )


@dataclass
class RetrievalResult:
    candidates: list[Evidence]
    filters_applied: dict[str, Any]
    filters_dropped: bool  # True if filtered search was empty and we retried unfiltered


class HybridRetriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder, cfg: RetrievalConfig):
        self.repo, self.embedder, self.cfg = ChunkRepo(conn), embedder, cfg

    def retrieve(self, analysis: QueryAnalysis) -> RetrievalResult:
        # Which documents may be searched at all is a corpus-trust rule, not a query filter: it is
        # kept when an over-eager query filter is dropped below.
        allowed = self.cfg.allowed_verification_statuses
        flt = MetadataFilter(court=analysis.court, date_from=analysis.date_from,
                             date_to=analysis.date_to, verification_statuses=allowed)
        applied = {k: str(v) for k, v in vars(flt).items() if v and k != "verification_statuses"}
        result = self._search(analysis.rewritten_query, flt)
        dropped = False
        if not result and applied:
            # An over-eager filter from query analysis must not turn into a silent miss.
            result = self._search(analysis.rewritten_query, MetadataFilter(verification_statuses=allowed))
            dropped = True
        return RetrievalResult(result, applied, dropped)

    def _search(self, query: str, flt: MetadataFilter) -> list[Evidence]:
        lexical = dict(self.repo.lexical_search(query, self.cfg.lexical_top_k, flt))
        vec = self.embedder.embed([query])[0]
        dense = dict(self.repo.dense_search(vec, self.cfg.dense_top_k, flt))
        fused = rrf_fuse([list(lexical), list(dense)], self.cfg.rrf_k)[: self.cfg.fused_top_k]
        rows = self.repo.evidence_rows([cid for cid, _ in fused])
        out: list[Evidence] = []
        for cid, score in fused:
            ev = evidence_from_row(rows[cid])
            ev.bm25_score, ev.dense_score, ev.fused_score = lexical.get(cid), dense.get(cid), score
            out.append(ev)
        return out
