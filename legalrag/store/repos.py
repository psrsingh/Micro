"""Repository classes. All SQL lives here; stages never build queries."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from legalrag.authority import AuthorityIndex, RegistryEntry, SectionVersion


@dataclass
class MetadataFilter:
    doc_type: str | None = None
    court: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    # Only documents with one of these verification statuses are searchable (None = all).
    verification_statuses: list[str] | None = None

    def sql(self) -> tuple[str, dict[str, Any]]:
        clauses, params = [], {}
        if self.verification_statuses is not None:
            clauses.append("d.verification_status = ANY(%(f_status)s)")
            params["f_status"] = list(self.verification_statuses)
        if self.doc_type:
            clauses.append("d.doc_type = %(f_doc_type)s")
            params["f_doc_type"] = self.doc_type
        if self.court:
            clauses.append("d.court ILIKE %(f_court)s")
            params["f_court"] = f"%{self.court}%"
        if self.date_from:
            clauses.append("d.decided_on >= %(f_from)s")
            params["f_from"] = self.date_from
        if self.date_to:
            clauses.append("d.decided_on <= %(f_to)s")
            params["f_to"] = self.date_to
        return (" AND " + " AND ".join(clauses)) if clauses else "", params


_WORD = re.compile(r"[A-Za-z0-9]+")


def or_tsquery(text: str) -> str | None:
    """OR-joined tsquery from free text. Returns None if there is nothing to search."""
    words = sorted({w.lower() for w in _WORD.findall(text) if len(w) > 1})
    return " | ".join(words) if words else None


class DocumentRepo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def upsert(self, *, doc_type: str, title: str, court: str | None, decided_on: date | None,
               neutral_citation: str | None, reporter_citations: list[str], source_url: str | None,
               raw_text_path: str, sha256: str) -> tuple[int, bool]:
        """Insert or update by raw_text_path. Returns (document_id, content_changed)."""
        row = self.conn.execute(
            "SELECT id, sha256 FROM documents WHERE raw_text_path = %s", (raw_text_path,)
        ).fetchone()
        args = (doc_type, title, court, decided_on, neutral_citation,
                Jsonb(reporter_citations), source_url, sha256)
        if row is None:
            new = self.conn.execute(
                "INSERT INTO documents (doc_type, title, court, decided_on, neutral_citation, "
                "reporter_citations, source_url, sha256, raw_text_path) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (*args, raw_text_path),
            ).fetchone()
            return new["id"], True
        self.conn.execute(
            "UPDATE documents SET doc_type=%s, title=%s, court=%s, decided_on=%s, "
            "neutral_citation=%s, reporter_citations=%s, source_url=%s, sha256=%s WHERE id=%s",
            (*args, row["id"]),
        )
        return row["id"], row["sha256"] != sha256

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]

    def corpus_sha(self) -> str:
        """One hash over every document sha + statute row: identifies the exact corpus."""
        h = hashlib.sha256()
        for r in self.conn.execute("SELECT raw_text_path, sha256 FROM documents ORDER BY raw_text_path"):
            h.update(f"{r['raw_text_path']}:{r['sha256']}\n".encode())
        return h.hexdigest()[:16]


@dataclass
class ChunkRow:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    section_label: str | None
    # Real-corpus location metadata; all optional so the fixture ingest is unchanged.
    page_start: int | None = None
    page_end: int | None = None
    act_name: str | None = None
    chapter: str | None = None
    section_number: str | None = None
    subsection: str | None = None
    heading: str | None = None
    paragraph_number: str | None = None
    metadata: dict | None = None


class ChunkRepo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def replace_for_document(self, document_id: int, chunks: Sequence[ChunkRow],
                             embeddings: np.ndarray) -> list[int]:
        """Delete and reinsert all chunks of a document. Returns new chunk ids in order."""
        self.conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        ids = []
        for c, emb in zip(chunks, embeddings, strict=True):
            r = self.conn.execute(
                "INSERT INTO chunks (document_id, ordinal, text, char_start, char_end, section_label, "
                "embedding, page_start, page_end, act_name, chapter, section_number, subsection, heading, "
                "paragraph_number, metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (document_id, c.ordinal, c.text, c.char_start, c.char_end, c.section_label, emb,
                 c.page_start, c.page_end, c.act_name, c.chapter, c.section_number, c.subsection,
                 c.heading, c.paragraph_number, Jsonb(c.metadata or {})),
            ).fetchone()
            ids.append(r["id"])
        return ids

    def count_for_document(self, document_id: int) -> int:
        return self.conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE document_id = %s", (document_id,)
        ).fetchone()["n"]

    # -- retrieval -----------------------------------------------------------------
    def lexical_search(self, query: str, k: int, flt: MetadataFilter) -> list[tuple[int, float]]:
        tsq = or_tsquery(query)
        if tsq is None:
            return []
        where, params = flt.sql()
        rows = self.conn.execute(
            "SELECT c.id AS chunk_id, ts_rank_cd(c.tsv, q) AS score "
            "FROM chunks c JOIN documents d ON d.id = c.document_id, "
            "to_tsquery('english', %(q)s) q "
            f"WHERE c.tsv @@ q{where} ORDER BY score DESC, c.id LIMIT %(k)s",
            {"q": tsq, "k": k, **params},
        ).fetchall()
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    def dense_search(self, vec: np.ndarray, k: int, flt: MetadataFilter) -> list[tuple[int, float]]:
        where, params = flt.sql()
        rows = self.conn.execute(
            "SELECT c.id AS chunk_id, 1 - (c.embedding <=> %(v)s) AS score "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE true{where} ORDER BY c.embedding <=> %(v)s, c.id LIMIT %(k)s",
            {"v": vec, "k": k, **params},
        ).fetchall()
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    def evidence_rows(self, chunk_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Chunk + document + statute metadata, keyed by chunk id."""
        if not chunk_ids:
            return {}
        rows = self.conn.execute(
            "SELECT c.id AS chunk_id, c.document_id, c.text, c.char_start, c.char_end, "
            "c.section_label, d.doc_type, d.title, d.court, d.decided_on, d.neutral_citation, "
            "COALESCE(ss.act_name, c.act_name) AS act_name, "
            "COALESCE(ss.section_number, c.section_number) AS section_number, "
            "c.subsection, c.heading, c.paragraph_number, c.page_start, c.page_end, "
            "d.doc_uid, d.verification_status, d.source_authority "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "LEFT JOIN statute_sections ss ON ss.chunk_id = c.id "
            "WHERE c.id = ANY(%s)",
            (list(chunk_ids),),
        ).fetchall()
        return {r["chunk_id"]: r for r in rows}


@dataclass
class StatuteRow:
    act_name: str
    act_year: int
    section_number: str
    subsection: str
    text: str
    valid_from: date
    valid_until: date | None
    amendment_note: str | None
    supersedes_valid_from: date | None = None  # valid_from of the version this row replaces


class StatuteRepo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def replace_for_document(self, document_id: int, rows: Sequence[StatuteRow],
                             chunk_ids: Sequence[int]) -> None:
        self.conn.execute("DELETE FROM statute_sections WHERE document_id = %s", (document_id,))
        inserted: dict[tuple, int] = {}
        for r, chunk_id in zip(rows, chunk_ids, strict=True):
            new = self.conn.execute(
                "INSERT INTO statute_sections (document_id, chunk_id, act_name, act_year, "
                "section_number, subsection, text, valid_from, valid_until, amendment_note) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (document_id, chunk_id, r.act_name, r.act_year, r.section_number, r.subsection,
                 r.text, r.valid_from, r.valid_until, r.amendment_note),
            ).fetchone()
            inserted[(r.act_name, r.act_year, r.section_number, r.subsection, r.valid_from)] = new["id"]
        # An old version points at the version that replaced it.
        for r in rows:
            if r.supersedes_valid_from is None:
                continue
            old = inserted[(r.act_name, r.act_year, r.section_number, r.subsection, r.supersedes_valid_from)]
            new_id = inserted[(r.act_name, r.act_year, r.section_number, r.subsection, r.valid_from)]
            self.conn.execute("UPDATE statute_sections SET superseded_by_id = %s WHERE id = %s",
                              (new_id, old))


class RegistryRepo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def upsert(self, *, case_name: str, citation: str, court: str | None, decided_on: date | None,
               in_corpus: bool, document_id: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO citation_registry (case_name, citation, court, decided_on, in_corpus, document_id) "
            "VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (citation, case_name) DO UPDATE SET court = EXCLUDED.court, "
            "decided_on = EXCLUDED.decided_on, in_corpus = EXCLUDED.in_corpus, "
            "document_id = EXCLUDED.document_id",
            (case_name, citation, court, decided_on, in_corpus, document_id),
        )

    def counts(self) -> dict[str, int]:
        r = self.conn.execute(
            "SELECT count(*) AS total, count(*) FILTER (WHERE in_corpus) AS in_corpus "
            "FROM citation_registry"
        ).fetchone()
        return {"total": r["total"], "in_corpus": r["in_corpus"]}


class AuthorityRepo:
    """Loads the registry and every statute version into an in-memory `AuthorityIndex`, which
    is what the verification checks and evaluation metrics query."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def load(self) -> AuthorityIndex:
        registry = [RegistryEntry(r["case_name"], r["citation"], r["court"], r["decided_on"], r["in_corpus"])
                    for r in self.conn.execute(
                        "SELECT case_name, citation, court, decided_on, in_corpus FROM citation_registry")]
        sections = [SectionVersion(r["act_name"], r["act_year"], r["section_number"], r["subsection"], r["text"],
                                   r["valid_from"], r["valid_until"], r["amendment_note"])
                    for r in self.conn.execute(
                        "SELECT act_name, act_year, section_number, subsection, text, valid_from, valid_until, "
                        "amendment_note FROM statute_sections")]
        return AuthorityIndex(registry, sections)

    def case_passages(self, citation: str | None, name: str | None) -> list[str]:
        """Every chunk of the judgment identified by neutral citation (or, failing that, title)."""
        rows = self.conn.execute(
            "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.doc_type = 'judgment' AND (d.neutral_citation = %(c)s::text "
            "OR (%(c)s::text IS NULL AND d.title = %(n)s::text)) "
            "ORDER BY c.ordinal", {"c": citation, "n": name}).fetchall()
        return [r["text"] for r in rows]

    def section_passages(self, act_name: str, section: str) -> list[str]:
        return [r["text"] for r in self.conn.execute(
            "SELECT text FROM statute_sections WHERE act_name = %s AND section_number = %s "
            "ORDER BY valid_from", (act_name, section))]


class RunRepo:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def insert_run(self, *, experiment: str, config_name: str, config_hash: str, corpus_sha: str,
                   model_info: dict, query: str, outcome: str, final_answer: dict | None,
                   trace: dict, latency_ms: dict) -> int:
        return self.conn.execute(
            "INSERT INTO runs (experiment, config_name, config_hash, corpus_sha, model_info, query, "
            "outcome, final_answer, trace, latency_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "RETURNING id",
            (experiment, config_name, config_hash, corpus_sha, Jsonb(model_info), query, outcome,
             Jsonb(final_answer) if final_answer is not None else None, Jsonb(trace), Jsonb(latency_ms)),
        ).fetchone()["id"]

    def insert_item(self, run_id: int, stage: str, rank: int, evidence: dict,
                    claim_idx: int | None = None) -> int:
        item_id = self.conn.execute(
            "INSERT INTO run_items (run_id, stage, claim_idx, rank, chunk_id, status, evidence) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (run_id, stage, claim_idx, rank, evidence.get("chunk_id"), evidence.get("status"),
             Jsonb(evidence)),
        ).fetchone()["id"]
        for name, cr in evidence.get("checks", {}).items():
            self.conn.execute(
                "INSERT INTO check_results (run_item_id, check_name, outcome, score, "
                "evidence_pointer, explanation) VALUES (%s,%s,%s,%s,%s,%s)",
                (item_id, name, cr["outcome"], cr["score"], cr.get("evidence_pointer"),
                 cr.get("explanation", "")),
            )
        return item_id


class CorpusRepo:
    """Real-corpus bookkeeping: provenance fields on `documents`, `document_provenance` and the
    `ingestion_events` log (which also records the documents that were refused)."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def set_document_fields(self, document_id: int, *, doc_uid: str, source_authority: str,
                            verification_status: str, original_url: str | None, retrieved_at,
                            source_sha256: str, language: str | None) -> None:
        self.conn.execute(
            "UPDATE documents SET doc_uid=%s, source_authority=%s, verification_status=%s, original_url=%s, "
            "retrieved_at=%s, source_sha256=%s, language=%s WHERE id=%s",
            (doc_uid, source_authority, verification_status, original_url, retrieved_at, source_sha256,
             language, document_id))

    def document_id_by_uid(self, doc_uid: str) -> int | None:
        r = self.conn.execute("SELECT id FROM documents WHERE doc_uid = %s", (doc_uid,)).fetchone()
        return r["id"] if r else None

    def upsert_provenance(self, document_id: int, **f: Any) -> None:
        cols = ["manifest_id", "source_name", "source_url", "source_license", "retrieved_at", "source_version",
                "source_file_path", "source_sha256", "source_size_bytes", "extraction_method", "extraction_tool",
                "ocr_used", "n_pages", "processing_status", "validation_errors", "validation_warnings",
                "applicable_from", "applicable_until", "period_basis"]
        vals = [Jsonb(f[c]) if c in ("validation_errors", "validation_warnings") else f.get(c) for c in cols]
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols) + ", processed_at=now()"
        self.conn.execute(
            f"INSERT INTO document_provenance (document_id, {', '.join(cols)}) VALUES (%s, {', '.join(['%s'] * len(cols))}) "
            f"ON CONFLICT (document_id) DO UPDATE SET {sets}", (document_id, *vals))

    def event(self, manifest_id: str, stage: str, status: str, message: str = "", detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO ingestion_events (manifest_id, stage, status, message, detail) VALUES (%s,%s,%s,%s,%s)",
            (manifest_id, stage, status, message, Jsonb(detail or {})))

    def delete_document(self, document_id: int) -> None:
        self.conn.execute("DELETE FROM citation_registry WHERE document_id = %s", (document_id,))
        self.conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))  # chunks/statutes cascade


def check_consistency(conn: psycopg.Connection) -> list[str]:
    """Return a list of human-readable problems; empty means chunks and metadata agree."""
    problems: list[str] = []
    for r in conn.execute(
        "SELECT d.id, d.title FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
        "WHERE c.id IS NULL"
    ):
        problems.append(f"document {r['id']} ({r['title']}) has no chunks")
    for r in conn.execute(
        "SELECT c.id FROM chunks c WHERE c.char_end <= c.char_start"
    ):
        problems.append(f"chunk {r['id']} has an empty/invalid span")
    for r in conn.execute(
        "SELECT ss.id FROM statute_sections ss WHERE ss.chunk_id IS NULL"
    ):
        problems.append(f"statute_section {r['id']} has no chunk")
    for r in conn.execute(
        "SELECT a.id FROM statute_sections a JOIN statute_sections b ON b.id = a.superseded_by_id "
        "WHERE a.valid_until IS NULL OR a.valid_until > b.valid_from"
    ):
        problems.append(f"statute_section {r['id']} overlaps the version that supersedes it")
    return problems


__all__ = [
    "MetadataFilter", "DocumentRepo", "ChunkRepo", "ChunkRow", "StatuteRepo", "StatuteRow",
    "RegistryRepo", "RunRepo", "AuthorityRepo", "CorpusRepo", "check_consistency", "or_tsquery",
]
