"""The three-arm comparison end to end against a real Postgres (fake generator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legalrag.eval.compare import run
from legalrag.ingest.corpus_view import CorpusView
from legalrag.pipeline.arms import ARMS, run_baseline, run_standard_rag, run_verified_rag
from legalrag.pipeline.orchestrate import Pipeline
from legalrag.settings import load_config
from legalrag.store.db import connect
from legalrag.store.repos import AuthorityRepo

pytestmark = pytest.mark.db

CORPUS = Path(__file__).parent / "fixtures" / "sample_corpus"


@pytest.fixture()
def demo_cfg(test_db_cfg):
    # test_db_cfg has built the throwaway database (and set LEGALRAG_DSN); offline_demo shares
    # its hash embedder, so it is compatible with that database.
    return load_config("offline_demo")


def test_authority_index_from_postgres_equals_the_one_from_the_files(test_db_cfg):
    view = CorpusView(CORPUS)
    with connect(test_db_cfg) as conn:
        db = AuthorityRepo(conn).load()
    key = lambda e: (e.case_name, e.citation, e.court, e.decided_on, e.in_corpus)  # noqa: E731
    assert sorted(map(key, db.registry)) == sorted(map(key, view.index.registry))
    vkey = lambda v: (v.act_name, v.section_number, v.valid_from, v.valid_until, v.text)  # noqa: E731
    assert sorted(map(vkey, db.sections)) == sorted(map(vkey, view.index.sections))


def test_repo_passage_lookups_return_the_stored_text(test_db_cfg):
    with connect(test_db_cfg) as conn:
        repo = AuthorityRepo(conn)
        assert any("thirty days" in t for t in repo.case_passages("2004 SYN 21", None))
        assert repo.case_passages(None, "Asha Devi v. Rajan Traders (SYNTHETIC)") == repo.case_passages("2004 SYN 21", None)
        assert len(repo.section_passages("Sample Tenancy Act", "7")) == 2  # both versions
        assert repo.case_passages("1999 SYN 1", None) == []


def test_each_arm_returns_the_common_result_and_logs_a_run(demo_cfg):
    q = "How much notice must a landlord give before seeking eviction under Section 12 of the Sample Tenancy Act?"
    with connect(demo_cfg) as conn:
        pipe = Pipeline.from_config(demo_cfg, conn)
        base = run_baseline(pipe, q, "t")
        rag = run_standard_rag(pipe, q, "t")
        ver = run_verified_rag(pipe, q, "t")
        assert base.retrieved == [] and base.evidence_pool == []  # no retrieval: nothing to ground on
        assert rag.retrieved and rag.evidence_pool and len(rag.evidence_pool) <= demo_cfg.rerank.keep
        assert ver.retrieved and ver.confidence_source in ("verification", "none")
        for r in (base, rag, ver):
            row = conn.execute("SELECT config_name, outcome FROM runs WHERE id = %s", (r.run_id,)).fetchone()
            assert row["outcome"] == r.outcome
        assert conn.execute("SELECT config_name FROM runs WHERE id = %s", (base.run_id,)).fetchone()["config_name"] == "offline_demo:baseline"


def test_verified_arm_answers_only_from_evidence_it_verified(demo_cfg):
    with connect(demo_cfg) as conn:
        pipe = Pipeline.from_config(demo_cfg, conn)
        ver = run_verified_rag(pipe, "Can a tenant sublet part of the premises without the landlord's written consent?", "t")
    if ver.outcome == "answered":
        assert ver.evidence_pool and all(e.status is not None for e in ver.evidence_pool)
        assert ver.citations  # the authorities behind the answer are shown, not implied
        assert ver.confidence is not None and 0.0 <= ver.confidence <= 1.0


def test_out_of_scope_question_is_abstained_by_every_arm(demo_cfg):
    with connect(demo_cfg) as conn:
        pipe = Pipeline.from_config(demo_cfg, conn)
        outcomes = {name: fn(pipe, "What is the capital of France?", "t").outcome
                    for name, fn in [("b", run_baseline), ("r", run_standard_rag), ("v", run_verified_rag)]}
    assert set(outcomes.values()) == {"abstained"}


def test_full_comparison_writes_a_report_with_every_metric(demo_cfg, tmp_path):
    j, m, summary = run(demo_cfg, list(ARMS), out_dir=tmp_path)
    payload = json.loads(j.read_text(encoding="utf-8"))
    assert payload["kind"] == "arm-comparison" and payload["config_hash"] == demo_cfg.config_hash
    assert set(payload["summary"]["arms"]) == set(ARMS)
    assert len(payload["records"]) == 18 * 3
    for arm, s in payload["summary"]["arms"].items():
        assert s["n_queries"] == 18
        for key in ("answer_accuracy", "cvr", "csr", "ega", "ucr", "hcer", "recall_doc@10"):
            assert key in s
    assert payload["summary"]["ava"]["ava"] == 1.0
    base, ver = summary["arms"]["baseline"], summary["arms"]["verified_rag"]
    assert base["recall_doc@10"] is None and ver["recall_doc@10"] is not None  # baseline retrieves nothing
    assert "PLUMBING RUN" in m.read_text(encoding="utf-8")  # a fake generator must never pass as a result
    md = m.read_text(encoding="utf-8")
    for heading in ("## Comparison", "## Retrieval", "## Error analysis", "## Authority Verification Accuracy", "## Per question"):
        assert heading in md
    # Everything the runs did is in Postgres under one experiment.
    with connect(demo_cfg) as conn:
        n = conn.execute("SELECT count(*) n FROM runs WHERE experiment = %s",
                         (f"compare-{demo_cfg.name}-{demo_cfg.config_hash}",)).fetchone()["n"]
    assert n == 18 * 3


def test_pipeline_abstains_before_retrieval_when_the_question_names_an_ungroundable_authority(demo_cfg):
    with connect(demo_cfg) as conn:
        pipe = Pipeline.from_config(demo_cfg, conn)
        r = pipe.run("What did the court hold in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8?", experiment="t")
        assert r.outcome == "abstained" and r.final.abstained and not r.final.llm_called
        assert "text is not in the corpus" in r.trace["unavailable_authorities"][0]
        assert r.retrieved == [] and "retrieval" not in r.latency_ms  # stopped before spending anything else
        row = conn.execute("SELECT outcome, trace FROM runs WHERE id = %s", (r.run_id,)).fetchone()
        assert row["outcome"] == "abstained" and row["trace"]["unavailable_authorities"]
        # the same question a standard RAG arm gets has no such guard: that is the comparison
        std = run_standard_rag(pipe, "What did the court hold in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8?", "t")
        assert std.retrieved  # it retrieved neighbours and went on to answer from them


def test_guard_can_be_switched_off_for_ablation(demo_cfg):
    cfg = demo_cfg.model_copy(deep=True)
    cfg.pipeline.abstain_on_unavailable_authority = False
    with connect(cfg) as conn:
        r = Pipeline.from_config(cfg, conn).run("What did the court hold in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8?", "t")
    assert "unavailable_authorities" not in r.trace and r.retrieved
