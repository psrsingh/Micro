"""Database tests: ingestion idempotency, consistency, retrieval, and run logging."""

from __future__ import annotations

import pytest

from legalrag.ingest.build import build, corpus_stats
from legalrag.ingest.parse import read_raw
from legalrag.pipeline.orchestrate import Pipeline
from legalrag.store.db import StoreError, connect, migrate
from legalrag.store.repos import check_consistency

pytestmark = pytest.mark.db


def counts(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
            for t in ("documents", "chunks", "statute_sections", "citation_registry")}


def test_rebuild_is_idempotent(test_db_cfg):
    with connect(test_db_cfg) as conn:
        before = counts(conn)
        sha_before = corpus_stats(conn)["corpus_sha"]
    report = build(test_db_cfg)
    assert report.documents_new_or_changed == 0 and report.documents_unchanged == 10
    with connect(test_db_cfg) as conn:
        assert counts(conn) == before
        assert corpus_stats(conn)["corpus_sha"] == sha_before
        assert check_consistency(conn) == []


def test_corpus_shape(test_db_cfg):
    with connect(test_db_cfg) as conn:
        stats = corpus_stats(conn)
    assert stats["documents"] == {"total": 10, "judgment": 8, "statute": 2}
    assert stats["statute_sections"]["superseded_versions"] == 1
    assert stats["citation_registry"] == {"total": 18, "in_corpus": 8}  # 10 outside-corpus + 8 held
    assert set(stats["chunks"]["by_section_label"]) >= {"headnote", "facts", "arguments", "reasoning", "holding"}


def test_every_chunk_span_is_verbatim_in_the_source_file(test_db_cfg):
    corpus = test_db_cfg.resolve_path(test_db_cfg.ingest.corpus_dir)
    with connect(test_db_cfg) as conn:
        rows = conn.execute("SELECT c.text, c.char_start, c.char_end, d.raw_text_path "
                            "FROM chunks c JOIN documents d ON d.id = c.document_id").fetchall()
    assert rows
    raw = {}
    for r in rows:
        raw.setdefault(r["raw_text_path"], read_raw(corpus / r["raw_text_path"])[0])
        assert raw[r["raw_text_path"]][r["char_start"]:r["char_end"]] == r["text"]


def test_amended_section_versions_do_not_overlap_and_are_linked(test_db_cfg):
    with connect(test_db_cfg) as conn:
        v = conn.execute("SELECT id, valid_from, valid_until, superseded_by_id FROM statute_sections "
                         "WHERE act_name = 'Sample Tenancy Act' AND section_number = '7' "
                         "ORDER BY valid_from").fetchall()
    assert len(v) == 2
    assert v[0]["superseded_by_id"] == v[1]["id"] and v[1]["superseded_by_id"] is None
    assert v[0]["valid_until"] < v[1]["valid_from"] and v[1]["valid_until"] is None


def test_registry_distinguishes_held_from_outside_corpus(test_db_cfg):
    with connect(test_db_cfg) as conn:
        held = conn.execute("SELECT in_corpus, document_id FROM citation_registry WHERE citation = '2004 SYN 21'").fetchone()
        outside = conn.execute("SELECT in_corpus, document_id FROM citation_registry WHERE citation = '2002 SYN 8'").fetchone()
        nothing = conn.execute("SELECT 1 FROM citation_registry WHERE citation = '1999 SYN 999'").fetchone()
    assert held["in_corpus"] and held["document_id"]
    assert not outside["in_corpus"] and outside["document_id"] is None
    assert nothing is None  # "found nowhere" stays distinguishable


def test_embedding_model_mismatch_is_refused(test_db_cfg):
    other = test_db_cfg.model_copy(update={"embedding": test_db_cfg.embedding.model_copy(update={"model": "other"})})
    with pytest.raises(StoreError):
        migrate(other)


def test_end_to_end_run_is_logged_with_every_stage(test_db_cfg):
    q = "How much notice must a landlord give before seeking eviction under Section 12 of the Sample Tenancy Act?"
    with connect(test_db_cfg) as conn:
        pipe = Pipeline.from_config(test_db_cfg, conn)
        r = pipe.run(q, experiment="test")
        assert r.outcome in ("answered", "abstained") and r.error is None
        assert any("Asha Devi" in (e.case_name or "") for e in r.kept)  # sanity, not a metric
        assert len(r.retrieved) <= test_db_cfg.retrieval.fused_top_k
        assert all(e.rerank_score is not None and e.fused_score is not None for e in r.retrieved)
        assert {"query_analysis", "retrieval", "rerank", "draft", "decompose", "verify", "final", "total"} <= set(r.latency_ms)
        run = conn.execute("SELECT * FROM runs WHERE id = %s", (r.run_id,)).fetchone()
        assert run["config_hash"] == test_db_cfg.config_hash and run["corpus_sha"] == pipe.corpus_sha
        assert run["model_info"]["llm"]["provider"] == "fake"
        stages = {x["stage"]: x["n"] for x in conn.execute(
            "SELECT stage, count(*) n FROM run_items WHERE run_id = %s GROUP BY stage", (r.run_id,))}
        assert stages["retrieved"] == len(r.retrieved) and stages["reranked"] == len(r.kept)
        n_checks = conn.execute(
            "SELECT count(*) n FROM check_results cr JOIN run_items ri ON ri.id = cr.run_item_id "
            "WHERE ri.run_id = %s", (r.run_id,)).fetchone()["n"]
        assert n_checks == sum(len(e.checks) for c in r.claims for e in c.evidence) > 0


def test_out_of_scope_question_abstains_end_to_end_and_is_logged(test_db_cfg):
    with connect(test_db_cfg) as conn:
        r = Pipeline.from_config(test_db_cfg, conn).run("What is the capital of France?", experiment="test")
        assert r.outcome == "abstained" and r.final.abstained and r.final.statements == []
        assert not r.final.llm_called  # abstention came from verification, not generator discretion
        assert r.claims == [] and r.retrieved  # retrieval still ran and was logged
        row = conn.execute("SELECT outcome, final_answer FROM runs WHERE id = %s", (r.run_id,)).fetchone()
        assert row["outcome"] == "abstained" and row["final_answer"]["abstained"] is True


def test_statute_evidence_carries_act_and_section(test_db_cfg):
    with connect(test_db_cfg) as conn:
        pipe = Pipeline.from_config(test_db_cfg, conn)
        r = pipe.run("security deposit exceeding months rent landlord", experiment="test")
    statutes = [e for e in r.retrieved if e.act_name]
    assert statutes and all(e.section_number and e.case_name is None for e in statutes)
