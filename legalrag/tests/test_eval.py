"""Benchmark integrity, authority probes and every metric, with no database or model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legalrag.eval.benchmark import BenchmarkItem, load_benchmark, validate_against_corpus
from legalrag.eval.metrics import (
    ScoringContext, aggregate, answer_correct, extract_cited, hallucination_reduction, judge_agreement,
    retrieval_scores, score_item,
)
from legalrag.eval.probes import load_probes, run_probes
from legalrag.ingest.corpus_view import CorpusView
from legalrag.pipeline.arms import ArmResult, split_statements
from legalrag.settings import load_config
from legalrag.verify.claim_support import LexicalSupportJudge

CORPUS = Path(__file__).parent / "fixtures" / "sample_corpus"
DATA = Path(__file__).parent.parent / "eval" / "data"


@pytest.fixture(scope="module")
def view() -> CorpusView:
    return CorpusView(CORPUS)


@pytest.fixture(scope="module")
def items() -> dict[str, BenchmarkItem]:
    return {i.id: i for i in load_benchmark(DATA / "benchmark_synthetic.jsonl")}


@pytest.fixture()
def ctx(view: CorpusView) -> ScoringContext:
    return ScoringContext(
        index=view.index, judge=LexicalSupportJudge(),
        case_passages=lambda cit, name: view.passages_for_case(cit or name or ""),
        section_passages=view.passages_for_section, ks=[1, 3, 10])


def arm(name="verified_rag", outcome="answered", text="", citations=(), conf=None, pool=(), retrieved=()) -> ArmResult:
    return ArmResult(name, "q", outcome, answer_text=text, statements=split_statements(text),
                     citations=list(citations), confidence=conf, retrieved=list(retrieved),
                     evidence_pool=list(pool), confidence_source="verification" if conf is not None else "none")


# -- benchmark ----------------------------------------------------------------------
def test_shipped_benchmark_matches_the_corpus_verbatim(view, items):
    assert validate_against_corpus(list(items.values()), view) == []
    assert sum(not i.answerable for i in items.values()) >= 3 and sum(i.answerable for i in items.values()) >= 12


def test_validation_catches_a_passage_that_is_not_in_the_corpus(view, items):
    bad = items["q01"].model_copy(deep=True)
    bad.passages[1].text = "a sentence that no judgment contains"
    assert any("not found verbatim" in p for p in validate_against_corpus([bad], view))
    ghost = items["q01"].model_copy(deep=True)
    ghost.gold_authorities[1].citation = "1999 SYN 1"
    assert any("not in corpus" in p for p in validate_against_corpus([ghost], view))


def test_answerable_items_must_carry_their_evidence():
    with pytest.raises(ValueError, match="needs gold_authorities"):
        BenchmarkItem(id="x", question="q")
    with pytest.raises(ValueError, match="must not list"):
        BenchmarkItem(id="x", question="q", answerable=False, must_include=[["a"]])


def test_loader_reports_the_line_and_rejects_duplicates(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"id": "a", "question": "q", "answerable": false}\n{"bad": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"b.jsonl:2"):
        load_benchmark(p)
    p.write_text('{"id": "a", "question": "q", "answerable": false}\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(p)


# -- authority probes / AVA ---------------------------------------------------------
def test_authority_probes_separate_valid_from_invalid(view):
    res = run_probes(load_probes(DATA / "authority_probes_synthetic.jsonl"), view.index, view)
    wrong = [r for r in res["rows"] if not r["correct"]]
    assert wrong == [] and res["ava"] == 1.0 and res["n"] >= 12
    assert res["confusion"]["valid_wrongly_rejected"] == 0 and res["confusion"]["invalid_missed"] == 0
    kinds = set(res["by_kind"])
    assert {"fabricated_case", "wrong_metadata", "wrong_section", "outside_corpus"} <= kinds


def test_probe_scoring_counts_misses(view):
    from legalrag.eval.probes import Probe, ProbeEvidence

    p = Probe(id="x", claim="Notice must be written.", evidence=ProbeEvidence(citation="2004 SYN 21"),
              label="invalid", kind="real")  # mislabelled on purpose: the checks will pass it
    res = run_probes([p], view.index, view)
    assert res["ava"] == 0.0 and res["confusion"]["invalid_missed"] == 1


# -- answer accuracy ----------------------------------------------------------------
def test_answer_correct_uses_groups_and_forbidden_phrases(items):
    q06 = items["q06"]
    good = arm(text="The cap is three months' rent, raised from two months by the 2012 amendment.")
    stale = arm(text="A landlord may take at most two months' rent, but also three months' rent.")
    only_old = arm(text="The maximum deposit is two months' rent.")
    assert answer_correct(q06, good) == (True, [])
    assert answer_correct(q06, only_old) == (False, ["outdated_authority"])
    # naming both caps without saying the old one was superseded is still the stale answer
    assert answer_correct(q06, stale) == (False, ["outdated_authority"])
    assert answer_correct(q06, arm(outcome="abstained"))[0] is False
    assert answer_correct(q06, arm(outcome="error"))[0] is False


def test_abstaining_is_the_correct_answer_to_an_unanswerable_question(items):
    assert answer_correct(items["q15"], arm(outcome="abstained"))[0] is True
    assert answer_correct(items["q15"], arm(text="Paris."))[0] is False


# -- retrieval ----------------------------------------------------------------------
def test_retrieval_scores_are_doc_and_span_level_and_undefined_for_the_baseline(view, items):
    q02 = items["q02"]  # gold: 2004 SYN 21 (headnote snippet) + Sample Tenancy Act s12
    headnote = view.judgment_evidence("2004 SYN 21", "headnote")
    reasoning = view.judgment_evidence("2004 SYN 21", "reasoning")
    s12 = view.statute_evidence("Sample Tenancy Act", "12")
    other = view.judgment_evidence("2019 SYN 27", "holding")
    r = retrieval_scores(q02, arm(retrieved=[other, reasoning, headnote, s12]), [1, 3, 10])
    assert r["recall_doc@1"] == 0.0
    assert r["recall_doc@3"] == 0.5  # 2004 SYN 21 present, statute not yet
    assert r["recall_doc@10"] == 1.0
    assert r["recall_span@3"] == 0.5 and r["recall_span@10"] == 1.0  # the reasoning chunk does not hold the snippet
    assert r["precision_span@10"] == 0.5
    assert retrieval_scores(q02, arm(name="baseline"), [1]) == {}
    assert retrieval_scores(items["q15"], arm(retrieved=[other]), [1]) == {}


# -- citations: CVR / CSR -----------------------------------------------------------
def test_cited_authorities_are_classified(ctx):
    a = arm(text="Asha Devi v. Rajan Traders (2004 SYN 21) and Ramesh Kulkarni v. Union Housing Board (2015 SYN 99) "
                 "apply Section 12 of the Sample Tenancy Act and Section 99 of the Sample Tenancy Act.",
            citations=["Mohan Lal v. Sunita Estates, 2004 SYN 21", "the landlord tenant case", "Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8"])
    by = {c.label: c for c in extract_cited(ctx, a)}
    assert by["2004 SYN 21"].valid and by["2004 SYN 21"].in_corpus
    assert by["2015 SYN 99"].category == "fabricated_citation"
    assert by["Section 12 of Sample Tenancy Act"].valid
    assert by["Section 99 of Sample Tenancy Act"].category == "wrong_section"
    assert by["the landlord tenant case"].kind == "unparsed" and not by["the landlord tenant case"].valid
    assert by["2002 SYN 8"].valid and not by["2002 SYN 8"].in_corpus  # real, but text not held


def test_real_citation_with_the_wrong_name_is_incorrect_metadata(ctx):
    a = arm(text="See Mohan Lal v. Sunita Estates, 2004 SYN 21 on notice.")
    (c,) = extract_cited(ctx, a)
    assert c.category == "incorrect_metadata" and not c.valid


def test_score_item_grounded_answer_versus_hallucinated_one(ctx, view, items):
    q01 = items["q01"]
    s12, holding = view.statute_evidence("Sample Tenancy Act", "12"), view.judgment_evidence("2004 SYN 21")
    grounded = arm(text="A landlord must give thirty days' written notice before eviction proceedings.",
                   citations=["Section 12 of Sample Tenancy Act", "Asha Devi v. Rajan Traders (2004 SYN 21)"],
                   conf=0.9, pool=[s12, holding], retrieved=[s12, holding])
    g = score_item(ctx, q01, grounded)
    assert g["correct"] and g["cited_valid"] == g["cited_total"] == 2
    assert g["csr_supported"] == g["csr_total"] == 2
    assert g["stmts_supported"] == g["stmts_total"] == 1 and g["error_categories"] == {}
    assert g["retrieval"]["recall_doc@1"] >= 0.5

    hallu = arm(name="baseline", text="A landlord must give sixty days' notice, as held in Ramesh Kulkarni v. Union Housing Board, 2015 SYN 99.",
                conf=0.95)
    h = score_item(ctx, q01, hallu)
    assert not h["correct"] and h["cited_valid"] == 0 and h["stmts_supported"] == 0  # empty pool: nothing grounded
    assert {"fabricated_citation", "unsupported_reasoning", "wrong_conclusion"} <= set(h["error_categories"])
    assert h["high_confidence_error"] is True and h["retrieval"] == {}


def test_valid_citation_that_supports_nothing_is_the_wrong_proposition_error(ctx, items):
    a = arm(text="A landlord must give sixty days' notice under the tenancy law.", citations=["Kavita Rao v. Greenfield Housing Society (2011 SYN 33)"])
    row = score_item(ctx, items["q01"], a)
    assert row["csr_total"] == 1 and row["csr_supported"] == 0
    assert "real_case_wrong_proposition" in row["error_categories"]


def test_unanswerable_question_answered_is_an_out_of_scope_error_with_no_high_confidence_when_unsure(ctx, items):
    row = score_item(ctx, items["q17"], arm(text="It held that the eviction failed.", conf=0.4))
    assert not row["correct"] and "answered_out_of_scope" in row["error_categories"]
    assert row["high_confidence_error"] is False
    ok = score_item(ctx, items["q17"], arm(outcome="abstained"))
    assert ok["correct"] and ok["error_categories"] == {} and not ok["has_confidence"]


def test_missed_answer_when_abstaining_on_an_answerable_question(ctx, items):
    row = score_item(ctx, items["q01"], arm(outcome="abstained"))
    assert "missed_answer" in row["error_categories"] and row["high_confidence_error"] is False


# -- aggregation --------------------------------------------------------------------
def test_aggregate_sums_counts_rather_than_averaging_ratios(ctx, items):
    rows = [
        score_item(ctx, items["q15"], arm(outcome="abstained")),
        score_item(ctx, items["q17"], arm(text="Something happened in that case decided long ago.", conf=0.9)),
        score_item(ctx, items["q01"], arm(text="Thirty days' written notice is required by Section 12.", conf=0.9,
                                          citations=["Section 12 of Sample Tenancy Act"],
                                          pool=[CorpusView(CORPUS).statute_evidence("Sample Tenancy Act", "12")])),
    ]
    s = aggregate(rows, [1, 3, 10])
    assert s["n_queries"] == 3 and s["answered"] == 2 and s["abstained"] == 1
    assert s["answer_accuracy"] == round(2 / 3, 4)
    assert s["abstention_accuracy_unanswerable"] == 0.5
    assert s["hcer"] == round(1 / 3, 4) and s["hcer_over_answered"] == 0.5
    assert s["cvr"] == 1.0 and s["n_cited"] == 1
    assert s["ega"] == 0.5 and s["ucr"] == 0.5  # 1 of 2 statements grounded


def test_hallucination_reduction_is_relative_to_the_baseline():
    base = {"invalid_citation_rate": 0.5, "ucr": 1.0, "error_categories": {"fabricated_citation": 4}}
    other = {"invalid_citation_rate": 0.0, "ucr": 0.25, "error_categories": {}}
    red = hallucination_reduction(base, other)
    assert red["invalid_citation_rate"] == 1.0 and red["ucr"] == 0.75 and red["fabricated_citation"] == 1.0
    assert red["wrong_section"] is None  # baseline had none to reduce


def test_audit_judge_agreement_against_gold_labels(ctx, items):
    n, agree = judge_agreement(ctx, items["q06"])
    assert n == 2 and agree == 2  # supported cap-of-three vs contradicted cap-of-two, told apart by the number guard


def test_split_statements_is_sentence_level_and_ignores_fragments():
    assert split_statements("Notice is required. It must be written. Ok.") == ["Notice is required.", "It must be written."]


# -- config -------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["offline_demo", "local_ollama"])
def test_shipped_configs_load_and_pin_as_of(name):
    cfg = load_config(name)
    assert cfg.verify.kind == "real" and cfg.verify.as_of is not None  # an unpinned date is not reproducible
    json.loads(cfg.canonical_json())
    assert cfg.config_hash == load_config(name).config_hash


def test_wrong_answers_that_carry_a_caveat_are_counted_as_disclosed(ctx, items):
    silent = score_item(ctx, items["q06"], arm(text="The maximum deposit is two months' rent.", conf=0.9))
    flagged = arm(text="The maximum deposit is two months' rent.", conf=0.25)
    flagged.caveated = True
    disclosed = score_item(ctx, items["q06"], flagged)
    right = score_item(ctx, items["q06"], arm(text="It is three months' rent.", conf=0.9))
    s = aggregate([silent, disclosed, right], [1])
    assert s["n_wrong_answers"] == 2 and s["wrong_answers_disclosed"] == 0.5
    assert disclosed["high_confidence_error"] is False and silent["high_confidence_error"] is True
