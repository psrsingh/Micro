"""One idempotent command builds the whole corpus:

    python -m legalrag.ingest.build --config phase0

Re-running does not duplicate rows: documents are keyed by raw_text_path, unchanged
documents (same sha256 and a full chunk set) are skipped, and changed documents have
their chunks and statute rows replaced inside one transaction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from legalrag.embed import Embedder, build_embedder
from legalrag.ingest.chunk import chunk_span
from legalrag.ingest.parse import ParsedDocument, parse_document
from legalrag.ingest.registry import load_listing
from legalrag.settings import Config, load_config
from legalrag.store.db import connect, migrate
from legalrag.store.repos import (
    ChunkRepo, ChunkRow, DocumentRepo, RegistryRepo, StatuteRepo, StatuteRow, check_consistency,
)


@dataclass
class BuildReport:
    migrations_applied: list[str] = field(default_factory=list)
    documents_new_or_changed: int = 0
    documents_unchanged: int = 0
    registry: dict[str, int] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _chunk_rows(doc: ParsedDocument, cfg: Config) -> tuple[list[ChunkRow], list[int | None]]:
    """Chunk every structural section separately. For statutes, returns for each chunk the
    index of the section it came from (1 chunk per section version, never split)."""
    ic = cfg.ingest
    rows: list[ChunkRow] = []
    section_of: list[int | None] = []
    for si, sec in enumerate(doc.sections):
        if doc.doc_type == "statute":
            # One retrievable chunk per section version keeps statute_sections <-> chunks 1:1.
            spans = chunk_span(doc.raw_text, sec.start, sec.end, target=10**6, max_tokens=10**6, overlap=0)
        else:
            spans = chunk_span(doc.raw_text, sec.start, sec.end, target=ic.chunk_target_tokens,
                               max_tokens=ic.chunk_max_tokens, overlap=ic.chunk_overlap_tokens)
        for sp in spans:
            rows.append(ChunkRow(len(rows), doc.raw_text[sp.start:sp.end], sp.start, sp.end, sec.label))
            section_of.append(si if doc.doc_type == "statute" else None)
    return rows, section_of


def _ingest_document(conn: psycopg.Connection, cfg: Config, embedder: Embedder,
                     path: Path, rel: str) -> bool:
    doc = parse_document(path)
    docs, chunks_repo = DocumentRepo(conn), ChunkRepo(conn)
    doc_id, changed = docs.upsert(
        doc_type=doc.doc_type, title=doc.title, court=doc.court, decided_on=doc.decided_on,
        neutral_citation=doc.neutral_citation, reporter_citations=doc.reporter_citations,
        source_url=doc.source_url, raw_text_path=rel, sha256=doc.sha256)
    have_chunks = chunks_repo.count_for_document(doc_id)
    if not changed and have_chunks > 0:
        _register_in_corpus(conn, doc, doc_id)
        return False

    rows, section_of = _chunk_rows(doc, cfg)
    if not rows:
        raise ValueError(f"{rel}: produced no chunks")
    chunk_ids = chunks_repo.replace_for_document(
        doc_id, rows, embedder.embed([r.text for r in rows]))

    if doc.doc_type == "statute":
        assert doc.act_name and doc.act_year is not None
        statute_rows: list[StatuteRow] = []
        by_key: dict[tuple, list[int]] = {}
        for i, si in enumerate(section_of):
            v = doc.sections[si].statute  # type: ignore[index]
            assert v is not None
            statute_rows.append(StatuteRow(doc.act_name, doc.act_year, v.section_number, v.subsection,
                                           rows[i].text, v.valid_from, v.valid_until, v.amendment_note))
            by_key.setdefault((v.section_number, v.subsection), []).append(i)
        # Link each superseded version to the version that replaces it (next by valid_from).
        for idxs in by_key.values():
            idxs.sort(key=lambda i: statute_rows[i].valid_from)
            for prev, nxt in zip(idxs, idxs[1:]):
                statute_rows[nxt].supersedes_valid_from = statute_rows[prev].valid_from
        StatuteRepo(conn).replace_for_document(doc_id, statute_rows, chunk_ids)
    _register_in_corpus(conn, doc, doc_id)
    return True


def _register_in_corpus(conn: psycopg.Connection, doc: ParsedDocument, doc_id: int) -> None:
    if doc.doc_type == "judgment" and doc.neutral_citation:
        RegistryRepo(conn).upsert(case_name=doc.title, citation=doc.neutral_citation, court=doc.court,
                                  decided_on=doc.decided_on, in_corpus=True, document_id=doc_id)


def corpus_stats(conn: psycopg.Connection) -> dict:
    q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
    by_type = {r["doc_type"]: r["n"] for r in q("SELECT doc_type, count(*) n FROM documents GROUP BY 1")}
    by_label = {r["section_label"]: r["n"] for r in q(
        "SELECT section_label, count(*) n FROM chunks GROUP BY 1 ORDER BY 1")}
    lens = conn.execute(
        "SELECT count(*) n, coalesce(avg(array_length(regexp_split_to_array(text, '\\s+'), 1)), 0) avg_words, "
        "coalesce(max(array_length(regexp_split_to_array(text, '\\s+'), 1)), 0) max_words FROM chunks"
    ).fetchone()
    statutes = conn.execute(
        "SELECT count(*) n, count(*) FILTER (WHERE superseded_by_id IS NOT NULL) superseded FROM statute_sections"
    ).fetchone()
    reg = RegistryRepo(conn).counts()
    return {
        "documents": {"total": sum(by_type.values()), **by_type},
        "chunks": {"total": lens["n"], "avg_words": round(float(lens["avg_words"]), 1),
                   "max_words": lens["max_words"], "by_section_label": by_label},
        "statute_sections": {"total": statutes["n"], "superseded_versions": statutes["superseded"]},
        "citation_registry": reg,
        "corpus_sha": DocumentRepo(conn).corpus_sha(),
    }


def build(cfg: Config) -> BuildReport:
    report = BuildReport(migrations_applied=migrate(cfg))
    corpus_dir = cfg.resolve_path(cfg.ingest.corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus_dir does not exist: {corpus_dir}")
    embedder = build_embedder(cfg.embedding)
    files = sorted(p for p in corpus_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt documents under {corpus_dir}")
    with connect(cfg) as conn:  # single transaction: a failed build leaves nothing half-written
        for path in files:
            changed = _ingest_document(conn, cfg, embedder, path, path.relative_to(corpus_dir).as_posix())
            report.documents_new_or_changed += changed
            report.documents_unchanged += not changed
        for listing in sorted(corpus_dir.rglob("registry*.csv")):
            for k, v in load_listing(conn, listing).items():
                report.registry[k] = report.registry.get(k, 0) + v
        report.problems = check_consistency(conn)
        report.stats = corpus_stats(conn)
        if report.problems:
            raise RuntimeError("corpus consistency check failed:\n  " + "\n  ".join(report.problems))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the corpus (idempotent).")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = build(cfg)
    print(json.dumps({
        "config": cfg.name, "config_hash": cfg.config_hash,
        "migrations_applied": r.migrations_applied,
        "documents_new_or_changed": r.documents_new_or_changed,
        "documents_unchanged": r.documents_unchanged,
        "registry_listing": r.registry, "stats": r.stats,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
