"""Manifest -> validated documents -> Postgres. One document at a time, one commit per document:
a document that fails validation never reaches `documents`, and the reason is recorded in
`ingestion_events` instead of being lost. Re-running is idempotent: an unchanged document is not
re-embedded, a document that has become invalid/pending is removed from the index and logged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from legalrag.corpus.extract import ExtractedDoc, ExtractionError, extract_pdf
from legalrag.corpus.manifest import (
    INDEXABLE_STATUSES, LockRecord, Manifest, ManifestEntry, VerificationStatus, load_lock, load_manifest)
from legalrag.corpus.structure import (
    ParsedJudgment, ParsedStatute, StructuredChunk, parse_constitution, parse_india_code, parse_judgment,
    parse_statute)
from legalrag.corpus.validate import (
    ValidationReport, decide_status, validate_extracted, validate_manifest, validate_source)
from legalrag.embed import Embedder
from legalrag.settings import Config
from legalrag.store.repos import (
    ChunkRepo, ChunkRow, CorpusRepo, DocumentRepo, RegistryRepo, StatuteRepo, StatuteRow)


@dataclass
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def extracted(self) -> Path:
        return self.root / "extracted"


def paths_for(cfg: Config) -> Paths:
    return Paths(cfg.corpus.corpus_path)


def manifest_path(cfg: Config) -> Path:
    return cfg.resolve_path(cfg.corpus.manifest_path)


@dataclass
class Processed:
    entry: ManifestEntry
    report: ValidationReport
    status: VerificationStatus
    doc: ExtractedDoc | None = None
    parsed: ParsedStatute | ParsedJudgment | None = None
    lock: LockRecord | None = None


@dataclass
class EntryResult:
    doc_uid: str
    status: str            # official_verified | supplementary_source | pending_review | invalid
    action: str            # indexed | unchanged | rejected | removed | failed
    n_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def embed_text(entry: ManifestEntry, c: StructuredChunk) -> str:
    """What the dense embedder sees: the chunk plus the context that its verbatim text lacks
    (which Act/section, which case). The stored chunk text is never altered."""
    if entry.doc_type == "statute":
        head = f"{entry.act_name}, {entry.act_year}, section {c.section_number}"
        if c.subsection:
            head += f"({c.subsection})"
        if c.heading:
            head += f" - {c.heading}"
        return _norm_ws(f"{head}. {c.text}")
    kind = "Headnote (editorial summary)" if c.section_label == "headnote" else "Judgment"
    return _norm_ws(f"{kind}: {entry.title} ({entry.neutral_citation or 'no neutral citation'}). {c.text}")


def process_entry(entry: ManifestEntry, cfg: Config, lock: LockRecord | None,
                  all_locks: dict[str, LockRecord], extra: list | None = None) -> Processed:
    """Validate the source file, extract, parse, validate again, decide the trust status. Pure:
    touches neither the database nor the extracted-text directory."""
    paths = paths_for(cfg)
    rep = validate_source(entry, lock, paths.raw, cfg.corpus, all_locks)
    for f in extra or []:
        rep.findings.append(f)
    if rep.errors:
        return Processed(entry, rep, decide_status(entry, rep), lock=lock)
    try:
        doc = extract_pdf(paths.raw / entry.local_file, cfg.corpus, margin_x=entry.margin_x,
                          margin_left_x=entry.margin_left_x, strip_footnotes=entry.strip_footnotes)
    except ExtractionError as e:
        rep.add("error", "EXTRACTION_FAILED", str(e))
        return Processed(entry, rep, decide_status(entry, rep), lock=lock)
    for w in doc.warnings:
        rep.add("warning", "EXTRACTION_WARNING", w)
    if entry.doc_type == "statute":
        parse_fn = {"article": parse_constitution, "india_code": parse_india_code}.get(entry.numbering, parse_statute)
        parsed: ParsedStatute | ParsedJudgment = parse_fn(
            doc, entry.act_name or "", max_words=cfg.corpus.statute_max_words)
    else:
        parsed = parse_judgment(doc, max_words=cfg.corpus.judgment_max_words)
    validate_extracted(entry, doc, parsed, rep)
    return Processed(entry, rep, decide_status(entry, rep), doc, parsed, lock)


def write_extracted(paths: Paths, entry: ManifestEntry, doc: ExtractedDoc) -> tuple[str, str]:
    """Persist the extracted text (the thing chunk offsets index into) and a page map. Returns the
    corpus-relative path and the SHA-256 of the text file."""
    paths.extracted.mkdir(parents=True, exist_ok=True)
    data = doc.text.encode("utf-8")
    rel = f"extracted/{entry.doc_uid}.txt"
    (paths.root / rel).write_bytes(data)
    pages = [{"page": p.number, "start": s, "chars": len(p.text), "removed_furniture": p.removed,
              "margin_notes": len(p.margin), "needs_ocr": p.needs_ocr, "ocr_used": p.ocr_used}
             for p, s in zip(doc.pages, doc.page_starts())]
    (paths.root / f"extracted/{entry.doc_uid}.pages.json").write_text(
        json.dumps({"tool": doc.tool, "method": doc.method, "pages": pages}, indent=1), encoding="utf-8")
    return rel, hashlib.sha256(data).hexdigest()


def _all_chunks(p: Processed) -> tuple[list[StructuredChunk], list[int]]:
    """Chunks in document order, plus (statutes) the index of each unit's first chunk."""
    assert p.parsed is not None
    if isinstance(p.parsed, ParsedStatute):
        chunks, firsts = [], []
        for u in p.parsed.units:
            firsts.append(len(chunks))
            chunks.extend(u.chunks)
        return chunks, firsts
    return list(p.parsed.chunks), []


def _to_row(i: int, c: StructuredChunk) -> ChunkRow:
    return ChunkRow(i, c.text, c.char_start, c.char_end, c.section_label, c.page_start, c.page_end, c.act_name,
                    c.chapter, c.section_number, c.subsection, c.heading, c.paragraph_number, c.metadata)


def _same_chunks(conn: psycopg.Connection, doc_id: int, rows: list[ChunkRow]) -> bool:
    have = conn.execute(
        "SELECT char_start, char_end, section_label, section_number, subsection, paragraph_number, heading "
        "FROM chunks WHERE document_id = %s ORDER BY ordinal", (doc_id,)).fetchall()
    want = [(r.char_start, r.char_end, r.section_label, r.section_number, r.subsection, r.paragraph_number, r.heading)
            for r in rows]
    return [tuple(h.values()) for h in have] == want


def index_document(conn: psycopg.Connection, cfg: Config, embedder: Embedder, p: Processed) -> tuple[str, int]:
    """Write one validated document. Returns (action, n_chunks)."""
    e, assert_doc = p.entry, p.doc
    assert assert_doc is not None and p.parsed is not None and p.lock is not None
    paths = paths_for(cfg)
    rel, text_sha = write_extracted(paths, e, assert_doc)
    chunks, firsts = _all_chunks(p)
    if not chunks:
        raise ValueError(f"{e.doc_uid}: produced no chunks")
    rows = [_to_row(i, c) for i, c in enumerate(chunks)]

    corpus = CorpusRepo(conn)
    docs, chunk_repo = DocumentRepo(conn), ChunkRepo(conn)
    title = e.title
    doc_id, _changed = docs.upsert(
        doc_type=e.doc_type, title=title, court=e.court, decided_on=e.decided_on or e.enactment_date,
        neutral_citation=e.neutral_citation, reporter_citations=list(e.reporter_citations),
        source_url=e.source.landing_url or e.source.url, raw_text_path=rel, sha256=text_sha)
    corpus.set_document_fields(
        doc_id, doc_uid=e.doc_uid, source_authority=e.source_authority, verification_status=p.status,
        original_url=e.source.url, retrieved_at=p.lock.retrieved_at, source_sha256=p.lock.sha256, language=e.language)

    unchanged = chunk_repo.count_for_document(doc_id) > 0 and _same_chunks(conn, doc_id, rows)
    if not unchanged:
        vecs = embedder.embed([embed_text(e, c) for c in chunks])
        chunk_ids = chunk_repo.replace_for_document(doc_id, rows, vecs)
        if isinstance(p.parsed, ParsedStatute):
            assert e.act_name and e.act_year and e.period.valid_from
            srows = [StatuteRow(e.act_name, e.act_year, u.section_number, u.subsection, u.text,
                                e.period.valid_from, e.period.valid_until, e.period.note or None)
                     for u in p.parsed.units]
            StatuteRepo(conn).replace_for_document(doc_id, srows, [chunk_ids[i] for i in firsts])
    if e.doc_type == "judgment":
        reg = RegistryRepo(conn)
        for cit in ([e.neutral_citation] if e.neutral_citation else []) + list(e.reporter_citations):
            reg.upsert(case_name=e.title, citation=cit, court=e.court, decided_on=e.decided_on,
                       in_corpus=True, document_id=doc_id)
    warnings = p.report.warnings
    corpus.upsert_provenance(
        doc_id, manifest_id=e.doc_uid, source_name=e.source.name, source_url=e.source.url,
        source_license=e.source.license, retrieved_at=p.lock.retrieved_at,
        source_version=p.lock.etag or p.lock.last_modified, source_file_path=f"raw/{e.local_file}",
        source_sha256=p.lock.sha256, source_size_bytes=p.lock.size_bytes, extraction_method=assert_doc.method,
        extraction_tool=assert_doc.tool, ocr_used=assert_doc.ocr_used, n_pages=assert_doc.n_pages,
        processing_status="ingested_with_warnings" if warnings else "ingested",
        validation_errors=[], validation_warnings=warnings, applicable_from=e.period.valid_from,
        applicable_until=e.period.valid_until, period_basis=e.period.basis)
    return ("unchanged" if unchanged else "indexed"), len(chunks)


def build(cfg: Config, embedder: Embedder, conn_factory, *, only: list[str] | None = None,
          include_pending: bool = False, manifest: Manifest | None = None) -> list[EntryResult]:
    """Process every manifest entry (or `only`). `conn_factory()` yields a context-managed
    connection; each document gets its own transaction."""
    mpath = manifest_path(cfg)
    manifest = manifest or load_manifest(mpath)
    locks = load_lock(mpath)
    cross = validate_manifest(manifest, cfg.corpus)
    entries = [e for e in manifest.entries if not only or e.doc_uid in only]
    if len(entries) > cfg.corpus.max_documents:
        raise ValueError(f"{len(entries)} documents exceed corpus.max_documents={cfg.corpus.max_documents}")
    results: list[EntryResult] = []
    for e in entries:
        try:
            p = process_entry(e, cfg, locks.get(e.doc_uid), locks, cross.get(e.doc_uid))
            res = EntryResult(e.doc_uid, p.status, "rejected", 0, p.report.errors, p.report.reviews, p.report.warnings)
            indexable = p.status in INDEXABLE_STATUSES or (include_pending and p.status == "pending_review")
            with conn_factory() as conn:
                corpus = CorpusRepo(conn)
                if indexable and p.doc is not None:
                    res.action, res.n_chunks = index_document(conn, cfg, embedder, p)
                    corpus.event(e.doc_uid, "index", "warning" if p.report.warnings else "ok",
                                 f"{p.status}: {res.action}, {res.n_chunks} chunks",
                                 {"warnings": p.report.warnings, "status": p.status})
                else:
                    existing = corpus.document_id_by_uid(e.doc_uid)
                    if existing is not None:
                        corpus.delete_document(existing)
                        res.action = "removed"
                    corpus.event(e.doc_uid, "validate", "rejected",
                                 f"{p.status}: not indexed" + (" (removed the previously indexed copy)" if existing else ""),
                                 {"errors": p.report.errors, "reviews": p.report.reviews, "warnings": p.report.warnings})
        except Exception as ex:  # noqa: BLE001 - record and continue with the next document
            res = EntryResult(e.doc_uid, "invalid", "failed", errors=[f"{type(ex).__name__}: {ex}"])
            try:
                with conn_factory() as conn:
                    CorpusRepo(conn).event(e.doc_uid, "index", "failed", res.errors[0])
            except Exception:  # noqa: BLE001
                pass
        results.append(res)
    return results
