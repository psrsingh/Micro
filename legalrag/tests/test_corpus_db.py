"""Integration: manifest -> validation -> Postgres -> retrieval -> pipeline, on a throwaway database with
synthetic PDFs. Proves: unverified/invalid documents are not silently ingested and their reasons are
recorded; chunks are verbatim slices of the extracted text with page numbers; re-runs are idempotent;
a document that becomes invalid is removed; retrieval never returns untrusted documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import yaml

from legalrag.corpus.fetch import import_file
from legalrag.corpus.ingest import build, paths_for
from legalrag.corpus.manifest import load_manifest, load_lock, save_lock
from legalrag.embed import build_embedder
from legalrag.eval.benchmark import BenchmarkItem, validate_against_corpus
from legalrag.eval.compare import _DbBenchmarkView
from legalrag.pipeline.orchestrate import Pipeline
from legalrag.pipeline.retrieval import HybridRetriever
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import Config, load_config
from legalrag.store.db import connect, migrate
from legalrag.store.repos import AuthorityRepo
from legalrag.tests.pdfmaker import make_pdf
from legalrag.tests.test_corpus_units import gazette_pages, judgment_pages

pytestmark = pytest.mark.db

MIRROR = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"


class CountingEmbedder:
    def __init__(self, inner):
        self.inner, self.calls, self.dim, self.model_id = inner, 0, inner.dim, inner.model_id

    def embed(self, texts):
        self.calls += len(texts)
        return self.inner.embed(texts)


def _scan_pages():
    """A judgment whose header says INSC 820 (so it validates on its own) plus one page with no text layer."""
    pages = judgment_pages()
    pages[0] = [(x, y, t.replace("2024 INSC 819", "2024 INSC 820")) for x, y, t in pages[0]]
    return pages + [[]]


def _entries() -> list[dict]:
    src = lambda url: {"name": "test publisher", "url": url}  # noqa: E731
    return [
        {"doc_uid": "ACT-TEST-2023", "doc_type": "statute", "title": "The Test Sanhita, 2023", "source_authority": "official",
         "source": src("https://www.mha.gov.in/test.pdf"), "local_file": "acts/test.pdf",
         "identity": {"pages": 1, "all_of": ["CHAPTER"]}, "act_name": "Test Sanhita", "act_year": 2023,
         "enactment_date": "2023-12-25", "document_status": "in_force", "margin_x": 483, "margin_left_x": 116,
         "period": {"valid_from": "2024-07-01", "basis": "unverified_secondary"}},
        {"doc_uid": "SC-2024-INSC-819", "doc_type": "judgment", "title": "Central Bureau of Investigation v. Ashok Sirpal",
         "source_authority": "supplementary", "source": src(f"{MIRROR}/a.pdf"), "local_file": "sc/a.pdf",
         "identity": {"pages": 2, "all_of": ["2024 INSC 819"]}, "court": "Supreme Court of India", "decided_on": "2024-10-24",
         "neutral_citation": "2024 INSC 819", "reporter_citations": ["[2024] 10 S.C.R. 930"], "document_status": "reported_judgment"},
        # wrong document: the manifest claims a citation the PDF does not carry
        {"doc_uid": "SC-2024-INSC-999", "doc_type": "judgment", "title": "Someone v. Someone Else",
         "source_authority": "supplementary", "source": src(f"{MIRROR}/b.pdf"), "local_file": "sc/b.pdf",
         "identity": {"pages": 2, "all_of": ["2024 INSC 999"]}, "court": "Supreme Court of India", "decided_on": "2024-10-24",
         "neutral_citation": "2024 INSC 999", "reporter_citations": ["[2024] 10 S.C.R. 999"], "document_status": "reported_judgment"},
        # a page with no text layer and OCR off: needs a person
        {"doc_uid": "SC-2024-INSC-820-SCAN", "doc_type": "judgment", "title": "Scanned copy",
         "source_authority": "supplementary", "source": src(f"{MIRROR}/c.pdf"), "local_file": "sc/c.pdf",
         "identity": {"pages": 2, "all_of": ["2024 INSC 820"]}, "court": "Supreme Court of India", "decided_on": "2024-10-24",
         "neutral_citation": "2024 INSC 820", "reporter_citations": [], "document_status": "reported_judgment"},
    ]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = load_config("phase0")
    admin_dsn = os.environ.get("LEGALRAG_DSN") or base.db.dsn
    parts = urlsplit(admin_dsn)
    name = f"legalrag_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({e}); start it with `docker compose up -d db`")
    admin.execute(f'CREATE DATABASE "{name}"')
    old = {k: os.environ.get(k) for k in ("LEGALRAG_DSN", "LEGALRAG_CORPUS_DIR")}
    tmp = tmp_path_factory.mktemp("realcorpus")
    os.environ["LEGALRAG_DSN"] = urlunsplit(parts._replace(path=f"/{name}"))
    os.environ["LEGALRAG_CORPUS_DIR"] = str(tmp / "corpus")
    try:
        cfg: Config = load_config("phase0")
        cfg.corpus.manifest_path = str(tmp / "manifest.yaml")
        cfg.retrieval.allowed_verification_statuses = ["official_verified", "supplementary_source"]
        cfg.verify.kind, cfg.verify.include_flagged = "real", True
        cfg.verify.lexical_can_verify = False
        (tmp / "manifest.yaml").write_text(yaml.safe_dump({"version": 1, "entries": _entries()}), encoding="utf-8")
        pdfs = {
            "acts/test.pdf": make_pdf(gazette_pages()),
            "sc/a.pdf": make_pdf(judgment_pages()),
            "sc/b.pdf": make_pdf(judgment_pages()[:2] + [judgment_pages()[2]] + [[(118, 90, "unrelated " * 3)]]),
            "sc/c.pdf": make_pdf(_scan_pages()),
        }
        m = load_manifest(cfg.resolve_path(cfg.corpus.manifest_path))
        locks, raw = {}, paths_for(cfg).raw
        for e in m.entries:
            src = tmp / f"src_{e.local_file.replace('/', '_')}"
            src.write_bytes(pdfs[e.local_file])
            locks[e.doc_uid] = import_file(e, src, raw)
        save_lock(cfg.resolve_path(cfg.corpus.manifest_path), locks)
        migrate(cfg)
        yield cfg
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def run_build(cfg, embedder=None, **kw):
    emb = embedder or CountingEmbedder(build_embedder(cfg.embedding))
    return {r.doc_uid: r for r in build(cfg, emb, lambda: connect(cfg), **kw)}, emb


@pytest.fixture(scope="module")
def first_build(env):
    results, emb = run_build(env)
    return results, emb


def q(cfg, sql, *args):
    with connect(cfg) as conn:
        return conn.execute(sql, args).fetchall()


# ---------------------------------------------------------------------------------------------

def test_trusted_documents_are_indexed_with_the_right_status(env, first_build):
    results, _ = first_build
    assert results["ACT-TEST-2023"].action == "indexed" and results["ACT-TEST-2023"].status == "official_verified"
    assert results["SC-2024-INSC-819"].action == "indexed" and results["SC-2024-INSC-819"].status == "supplementary_source"
    rows = {r["doc_uid"]: r for r in q(env, "SELECT doc_uid, verification_status, source_authority, language FROM documents WHERE doc_uid IS NOT NULL")}
    assert set(rows) == {"ACT-TEST-2023", "SC-2024-INSC-819"}
    assert rows["ACT-TEST-2023"]["source_authority"] == "official" and rows["SC-2024-INSC-819"]["source_authority"] == "supplementary"


def test_invalid_and_pending_documents_are_not_ingested_and_the_reason_is_recorded(env, first_build):
    results, _ = first_build
    assert results["SC-2024-INSC-999"].status == "invalid" and results["SC-2024-INSC-999"].action == "rejected"
    assert any("IDENTITY_MISMATCH" in m or "CITATION" in m for m in results["SC-2024-INSC-999"].errors)
    assert results["SC-2024-INSC-820-SCAN"].status == "pending_review"         # valid except for the page needing OCR
    assert any("NEEDS_OCR" in m for m in results["SC-2024-INSC-820-SCAN"].reviews)
    held = q(env, "SELECT count(*) n FROM documents WHERE doc_uid IN ('SC-2024-INSC-999', 'SC-2024-INSC-820-SCAN')")[0]["n"]
    assert held == 0
    ev = {r["manifest_id"]: r for r in q(env, "SELECT manifest_id, status, message, detail FROM ingestion_events WHERE status = 'rejected'")}
    assert {"SC-2024-INSC-999", "SC-2024-INSC-820-SCAN"} <= set(ev)
    assert ev["SC-2024-INSC-999"]["detail"]["errors"]            # the reasons are stored, not just the fact


def test_every_chunk_is_a_verbatim_slice_of_the_stored_extracted_text_with_pages(env, first_build):
    paths = paths_for(env)
    docs = q(env, "SELECT id, doc_uid, raw_text_path, sha256 FROM documents WHERE doc_uid IS NOT NULL")
    assert docs
    for d in docs:
        data = (paths.root / d["raw_text_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == d["sha256"]
        text = data.decode("utf-8")
        n_pages = q(env, "SELECT n_pages FROM document_provenance WHERE document_id = %s", d["id"])[0]["n_pages"]
        for c in q(env, "SELECT text, char_start, char_end, page_start, page_end FROM chunks WHERE document_id = %s", d["id"]):
            assert text[c["char_start"]:c["char_end"]] == c["text"]
            assert 1 <= c["page_start"] <= c["page_end"] <= n_pages


def test_statute_sections_are_linked_to_chunks_with_the_applicable_period(env, first_build):
    rows = q(env, "SELECT section_number, subsection, valid_from, valid_until, chunk_id, act_name, act_year FROM statute_sections "
                  "WHERE act_name = 'Test Sanhita' ORDER BY section_number::int, subsection")
    assert {r["section_number"] for r in rows} == {"1", "2", "3", "4", "5", "6"}
    assert all(r["chunk_id"] is not None and str(r["valid_from"]) == "2024-07-01" and r["valid_until"] is None for r in rows)
    assert {r["subsection"] for r in rows if r["section_number"] == "4"} <= {"", "1", "2", "3"}
    heading = q(env, "SELECT heading, section_number, chapter FROM chunks WHERE act_name = 'Test Sanhita' AND section_number = '3'")[0]
    assert heading["heading"] == "Murder penalty." and heading["chapter"].startswith("CHAPTER I")


def test_provenance_records_the_original_file_and_how_it_was_processed(env, first_build):
    locks = load_lock(env.resolve_path(env.corpus.manifest_path))
    rows = q(env, "SELECT d.doc_uid, d.source_sha256, p.source_sha256 AS psha, p.extraction_method, p.extraction_tool, p.ocr_used, "
                  "p.processing_status, p.source_file_path, p.period_basis FROM documents d JOIN document_provenance p ON p.document_id = d.id")
    assert len(rows) == 2
    for r in rows:
        assert r["source_sha256"] == r["psha"] == locks[r["doc_uid"]].sha256
        assert r["extraction_method"] == "pdfplumber" and r["extraction_tool"].startswith("pdfplumber") and r["ocr_used"] is False
        assert (paths_for(env).root / r["source_file_path"]).is_file()
    assert {r["doc_uid"]: r["period_basis"] for r in rows}["ACT-TEST-2023"] == "unverified_secondary"


def test_judgment_citations_enter_the_registry_as_in_corpus(env, first_build):
    rows = q(env, "SELECT citation, in_corpus FROM citation_registry ORDER BY citation")
    assert {"2024 INSC 819", "[2024] 10 S.C.R. 930"} <= {r["citation"] for r in rows} and all(r["in_corpus"] for r in rows)


def test_a_rerun_is_idempotent_and_does_not_re_embed(env, first_build):
    before = q(env, "SELECT count(*) n FROM chunks")[0]["n"]
    results, emb = run_build(env)
    assert results["ACT-TEST-2023"].action == "unchanged" and results["SC-2024-INSC-819"].action == "unchanged"
    assert emb.calls == 0
    assert q(env, "SELECT count(*) n FROM chunks")[0]["n"] == before
    assert q(env, "SELECT count(*) n FROM documents WHERE doc_uid IS NOT NULL")[0]["n"] == 2


def test_retrieval_only_returns_trusted_documents(env, first_build):
    # index the scanned document *as pending* on purpose, then prove the retriever still cannot see it
    results, _ = run_build(env, include_pending=True, only=["SC-2024-INSC-820-SCAN"])
    assert results["SC-2024-INSC-820-SCAN"].status == "pending_review" and results["SC-2024-INSC-820-SCAN"].action == "indexed"
    assert q(env, "SELECT verification_status FROM documents WHERE doc_uid = 'SC-2024-INSC-820-SCAN'")[0]["verification_status"] == "pending_review"
    qa = QueryAnalysis(issue="i", rewritten_query="factual controversy appeal limited")
    with connect(env) as conn:
        hits = HybridRetriever(conn, build_embedder(env.embedding), env.retrieval).retrieve(qa).candidates
        assert hits and all(h.verification_status in ("official_verified", "supplementary_source") for h in hits)
        assert "SC-2024-INSC-820-SCAN" not in {h.doc_uid for h in hits}
        open_cfg = env.retrieval.model_copy(update={"allowed_verification_statuses": None})
        all_hits = HybridRetriever(conn, build_embedder(env.embedding), open_cfg).retrieve(qa).candidates
        assert "SC-2024-INSC-820-SCAN" in {h.doc_uid for h in all_hits}      # ...so the filter, not absence, is what hid it
    # and a normal build removes it again, because it is not trusted
    results, _ = run_build(env, only=["SC-2024-INSC-820-SCAN"])
    assert results["SC-2024-INSC-820-SCAN"].action == "removed"


def test_retrieved_evidence_carries_location_and_provenance(env, first_build):
    with connect(env) as conn:
        hits = HybridRetriever(conn, build_embedder(env.embedding), env.retrieval).retrieve(
            QueryAnalysis(issue="murder", rewritten_query="whoever commits murder punished with death")).candidates
    top = next(h for h in hits if h.act_name == "Test Sanhita" and h.section_number == "3")
    assert top.doc_uid == "ACT-TEST-2023" and top.verification_status == "official_verified"
    assert top.page_start == 2 and top.heading == "Murder penalty." and top.section_label == "statute_section"


def test_a_tampered_file_is_removed_from_the_index_and_logged_then_restored(env, first_build):
    raw = paths_for(env).raw / "acts/test.pdf"
    good = raw.read_bytes()
    try:
        raw.write_bytes(good + b"\ntampered\n%%EOF\n")
        results, _ = run_build(env, only=["ACT-TEST-2023"])
        assert results["ACT-TEST-2023"].action == "removed" and results["ACT-TEST-2023"].status == "invalid"
        assert q(env, "SELECT count(*) n FROM documents WHERE doc_uid = 'ACT-TEST-2023'")[0]["n"] == 0
        assert q(env, "SELECT count(*) n FROM statute_sections WHERE act_name = 'Test Sanhita'")[0]["n"] == 0
        msg = q(env, "SELECT message FROM ingestion_events WHERE manifest_id = 'ACT-TEST-2023' AND status = 'rejected' ORDER BY id DESC")[0]["message"]
        assert "removed the previously indexed copy" in msg
    finally:
        raw.write_bytes(good)
    results, _ = run_build(env, only=["ACT-TEST-2023"])
    assert results["ACT-TEST-2023"].action == "indexed"


def test_corpus_report_counts_match_the_database(env, first_build):
    from legalrag.corpus.__main__ import cmd_report

    assert cmd_report(env, argparse.Namespace(only=None)) == 0
    data = json.loads((paths_for(env).root / "reports" / "corpus_report.json").read_text(encoding="utf-8"))
    stats = data["stats"]
    n_chunks = q(env, "SELECT count(*) n FROM chunks")[0]["n"]
    assert stats["total_documents"] == 2 and stats["total_statutes"] == 1 and stats["total_judgments"] == 1
    assert stats["total_chunks"] == stats["total_embeddings"] == n_chunks and stats["total_pages"] > 0
    assert stats["manifest_entries"] == 4    # the invalid + pending_review entries are counted too, not hidden
    assert stats["documents_with_errors"] >= 1     # SC-2024-INSC-999's IDENTITY_MISMATCH/CITATION error
    assert stats["duplicate_documents"] == 0 and stats["documents_without_source_urls"] == 0


def test_db_backed_benchmark_view_validates_gold_passages_against_the_real_database(env, first_build):
    """`eval.corpus_source: db` (real_corpus.yaml): the benchmark's gold passages are checked
    against the live database, not the synthetic-fixture files `CorpusView` reads."""
    good = BenchmarkItem(
        id="rc01", question="q", answerable=True,
        gold_authorities=[{"kind": "statute", "act": "Test Sanhita", "section": "3"}],
        passages=[{"authority": "Test Sanhita|3", "text": "shall be punished with death"}],
        must_include=[["death"]])
    bad = BenchmarkItem(
        id="rc02", question="q", answerable=True,
        gold_authorities=[{"kind": "statute", "act": "Test Sanhita", "section": "3"}],
        passages=[{"authority": "Test Sanhita|3", "text": "this sentence is not in the Act at all"}],
        must_include=[["death"]])
    with connect(env) as conn:
        repo = AuthorityRepo(conn)
        view = _DbBenchmarkView(repo.load(), repo)
        assert validate_against_corpus([good], view) == []
        problems = validate_against_corpus([bad], view)
        assert problems and "not found verbatim" in problems[0]
        # a gold authority the corpus doesn't hold is also caught, not silently accepted
        missing = BenchmarkItem(
            id="rc03", question="q", answerable=True,
            gold_authorities=[{"kind": "case", "citation": "1999 INSC 1"}],
            passages=[{"authority": "1999 INSC 1", "text": "x"}], must_include=[["x"]])
        assert any("not in corpus" in p for p in validate_against_corpus([missing], view))


def test_the_pipeline_answers_from_the_real_corpus_with_typed_sources_and_a_disclaimer(env, first_build):
    with connect(env) as conn:
        pipe = Pipeline.from_config(env, conn)
        res = pipe.run("What is the punishment for murder?", experiment="test_real_corpus")
    assert res.outcome in ("answered", "abstained") and res.error is None
    assert res.final is not None and "not legal advice" in res.final.disclaimer.lower()
    assert any(e.doc_uid == "ACT-TEST-2023" for e in res.retrieved)
    with connect(env) as conn:
        logged = conn.execute("SELECT final_answer FROM runs WHERE id = %s", (res.run_id,)).fetchone()["final_answer"]
    assert "not legal advice" in logged["disclaimer"].lower()
    for st in res.final.statements:
        assert st["source_details"] and all(d["kind"] and "verification_status" in d for d in st["source_details"])
        assert st["origin"] == "model_generated_from_verified_claims"
