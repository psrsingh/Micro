"""Unit tests for pipeline stages that need no database."""

from __future__ import annotations

import json

from legalrag.evidence import Claim, Evidence, Outcome, Status
from legalrag.llm import LLMRequest
from legalrag.llm.fake import FakeLLM
from legalrag.pipeline.decompose import decompose
from legalrag.pipeline.generate import final_answer, label_evidence
from legalrag.pipeline.query_analysis import analyze_query
from legalrag.pipeline.rerank import LexicalOverlapScorer, boosts, rerank
from legalrag.pipeline.retrieval import rrf_fuse
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import RerankConfig, VerifyConfig
from legalrag.verify.fake import build_checks, phase0_status
from legalrag.verify.verifier import Verifier


def ev(cid: int, text: str, label: str = "holding", **kw) -> Evidence:
    return Evidence(chunk_id=cid, document_id=cid, text=text, span=(0, len(text)),
                    section_label=label, **kw)


def test_rrf_rewards_agreement_and_is_deterministic():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]  # both lists rank 1 and 3 highly
    assert fused == rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_query_analysis_falls_back_to_raw_query_on_bad_output():
    llm = FakeLLM(responder=lambda r: "I cannot produce JSON")
    res = analyze_query(llm, "what is section 12?")
    assert res.analysis.fallback and res.analysis.rewritten_query == "what is section 12?"
    assert res.fallback_reason


def test_query_analysis_accepts_valid_output():
    res = analyze_query(FakeLLM(), "Notice under Section 12?")
    assert not res.analysis.fallback and res.fallback_reason is None
    assert res.analysis.section_numbers == ["12"]


def test_rerank_boosts_holding_over_arguments_and_matches_sections():
    cfg = RerankConfig()
    qa = QueryAnalysis(issue="i", rewritten_query="notice eviction", section_numbers=["12"],
                       case_names=["Asha Devi v. Rajan Traders"])
    holding = ev(1, "Section 12 requires notice for eviction.", "holding",
                 case_name="Asha Devi v. Rajan Traders (SYNTHETIC)")
    args = ev(2, "Section 12 requires notice for eviction.", "arguments")
    assert set(boosts(holding, qa, cfg)) == {"section_match", "case_name_match", "holding_or_reasoning"}
    assert "arguments_penalty" in boosts(args, qa, cfg)
    ranked, kept = rerank([args, holding], qa, LexicalOverlapScorer(), RerankConfig(keep=1))
    assert [e.chunk_id for e in ranked] == [1, 2] and [e.chunk_id for e in kept] == [1]
    assert all(e.rerank_score is not None for e in ranked)  # every candidate is scored


def test_decompose_keeps_unresolved_citations_and_isolates_evidence_copies():
    labelled = label_evidence([ev(10, "alpha"), ev(11, "beta")])
    draft = "Claim one. [E1]\nClaim two. [E1][E9]"
    res = decompose(FakeLLM(), draft, labelled, max_claims=10)
    c1, c2 = res.claims
    assert c2.unresolved_citations == ["E9"]  # a citation to nothing we retrieved is not dropped
    assert c1.evidence[0] is not c2.evidence[0]  # same chunk, separate objects
    c1.evidence[0].status = Status.REJECTED
    assert c2.evidence[0].status is None


def test_decompose_falls_back_to_lines_on_bad_output():
    llm = FakeLLM(responder=lambda r: "nope")
    res = decompose(llm, "A. [E1]\nB. [E1]", label_evidence([ev(1, "x")]), max_claims=10)
    assert res.fallback_reason and [c.text for c in res.claims] == ["A.", "B."]


def test_no_surviving_claims_abstains_without_calling_the_model():
    llm = FakeLLM()
    res = final_answer(llm, "q", [])
    assert res.abstained and not res.llm_called and llm.calls == []


def test_final_answer_drops_statements_citing_unknown_claims():
    surviving = Claim(idx=1, text="Notice is required.", evidence=[])
    out = {"abstain": False, "statements": [
        {"text": "ok", "claim_ids": [1]}, {"text": "invented", "claim_ids": [7]}, {"text": "uncited", "claim_ids": []}]}
    res = final_answer(FakeLLM(responder=lambda r: json.dumps(out)), "q", [surviving])
    assert [s["text"] for s in res.statements] == ["ok"]
    assert "dropped 2" in (res.used_fallback or "")


def test_final_answer_parse_failure_falls_back_to_verified_claims():
    surviving = Claim(idx=1, text="Notice is required.")
    res = final_answer(FakeLLM(responder=lambda r: "garbage"), "q", [surviving])
    assert not res.abstained and res.statements[0]["claim_ids"] == [1] and res.used_fallback


def test_verifier_statuses_and_survival():
    strong = ev(1, "notice must be written before eviction proceedings", "holding")
    weak = ev(2, "completely unrelated words here", "holding")
    args = ev(3, "notice must be written before eviction proceedings", "arguments")
    claim = Claim(idx=1, text="Notice must be written before eviction proceedings",
                  evidence=[strong, weak, args])
    Verifier(build_checks(VerifyConfig()), phase0_status, 1).verify([claim])
    assert [e.status for e in claim.evidence] == [Status.VERIFIED, Status.REJECTED, Status.FLAGGED]
    assert claim.survives is True
    # checks that cannot run say so rather than staying silent
    assert strong.checks["authority"].outcome == Outcome.NOT_APPLICABLE
    assert "not implemented" in strong.checks["authority"].explanation


def test_claim_with_only_flagged_or_no_evidence_does_not_survive():
    args = ev(3, "notice must be written before eviction proceedings", "arguments")
    claims = [Claim(1, "Notice must be written before eviction proceedings", [args]),
              Claim(2, "Cites something not retrieved", [], unresolved_citations=["E9"])]
    Verifier(build_checks(VerifyConfig()), phase0_status, 1).verify(claims)
    assert [c.survives for c in claims] == [False, False]


def test_disabled_checks_are_not_run():
    checks = build_checks(VerifyConfig(enabled_checks=["claim_support"]))
    assert [c.name for c in checks] == ["claim_support"]


def test_fake_llm_records_calls():
    llm = FakeLLM()
    llm.complete(LLMRequest(task="draft", prompt="x"))
    assert len(llm.calls) == 1
