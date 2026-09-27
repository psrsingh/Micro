"""The real verification layer, exercised against the fixture corpus with no database."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from legalrag.authority import norm_name
from legalrag.evidence import CheckResult, Claim, Evidence, Outcome, Status
from legalrag.ingest.corpus_view import CorpusView
from legalrag.llm.fake import FakeLLM
from legalrag.pipeline.generate import final_answer
from legalrag.settings import VerifyConfig
from legalrag.verify.build import build_verifier
from legalrag.verify.claim_support import (
    ClaimSupportCheck, LexicalSupportJudge, LLMSupportJudge, locate_quote, strip_references,
)
from legalrag.verify.extract import (
    find_case_refs, find_courts, find_section_refs, stated_decision_year,
)
from legalrag.verify.rules import make_status_fn

CORPUS = Path(__file__).parent / "fixtures" / "sample_corpus"
AS_OF = date(2026, 9, 20)


@pytest.fixture(scope="module")
def view() -> CorpusView:
    return CorpusView(CORPUS)


@pytest.fixture(scope="module")
def cfg() -> VerifyConfig:
    return VerifyConfig(kind="real", as_of=AS_OF)


def verify(view: CorpusView, cfg: VerifyConfig, text: str, ev: Evidence, **kw) -> Claim:
    claim = Claim(1, text, [ev])
    build_verifier(cfg, view.index, kw.get("llm")).verify([claim])
    return claim


def outcomes(claim: Claim) -> dict[str, Outcome]:
    return {n: r.outcome for n, r in claim.evidence[0].checks.items()}


# -- extraction ---------------------------------------------------------------------
def test_case_refs_pair_a_citation_and_survive_a_lead_in():
    (ref,) = find_case_refs("In Asha Devi v. Rajan Traders (SYNTHETIC), 2004 SYN 21, the court held X.")
    assert ref.name == "Asha Devi v. Rajan Traders" and ref.citation == "2004 SYN 21"


def test_adjacent_cases_are_not_fused():
    refs = find_case_refs("See Kavita Rao v. Greenfield Housing Society and Mohan Lal v. Sunita Estates (2008 SYN 5).")
    assert [r.name for r in refs] == ["Kavita Rao v. Greenfield Housing Society", "Mohan Lal v. Sunita Estates"]
    assert refs[1].citation == "2008 SYN 5"


@pytest.mark.parametrize("cite", ["2004 SYN 21", "(2004) 3 SYN 101", "AIR 1973 SC 1461", "2023 INSC 123"])
def test_citation_formats(cite: str):
    assert [r.citation for r in find_case_refs(f"as reported at {cite} in the volume")] == [cite]


def test_section_refs_take_the_act_from_either_side():
    a, b = find_section_refs("Section 12 of the Sample Tenancy Act, 2001 and Sample Consumer Protection Act, 2010, Section 3")
    assert (a.number, a.act.name, a.act.year) == ("12", "Sample Tenancy Act", 2001)
    assert (b.number, b.act.name) == ("3", "Sample Consumer Protection Act")
    (c,) = find_section_refs("notice under Section 15(2) is required")
    assert (c.number, c.subsection, c.act) == ("15", "2", None)


def test_decision_year_is_only_read_when_stated_about_the_case():
    t = "Asha Devi v. Rajan Traders was decided in 2010 under the Act, 2001."
    (ref,) = find_case_refs(t)
    assert stated_decision_year(t, ref.end) == 2010
    t2 = "Asha Devi v. Rajan Traders applied the Sample Tenancy Act, 2001."
    (ref2,) = find_case_refs(t2)
    assert stated_decision_year(t2, ref2.end) is None  # an Act's year is not a decision year
    assert find_courts("The High Court and the Supreme Court") == ["high court", "supreme court"]


def test_norm_name_ignores_case_punctuation_and_synthetic_tag():
    assert norm_name("The Asha Devi v. Rajan Traders (SYNTHETIC)") == norm_name("asha devi v rajan traders")


# -- authority ----------------------------------------------------------------------
def test_fabricated_citation_is_rejected(view, cfg):
    c = verify(view, cfg, "In Ramesh Kulkarni v. Union Housing Board, 2015 SYN 99, notice must be written.",
               view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["authority"] == Outcome.FAIL
    assert c.evidence[0].status == Status.REJECTED and c.survives is False


def test_fabricated_name_without_citation_is_rejected(view, cfg):
    c = verify(view, cfg, "In Sanjay Verma v. Delta Estates the court required written notice.",
               view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["authority"] == Outcome.FAIL


def test_real_case_outside_the_corpus_is_flagged_not_rejected(view, cfg):
    c = verify(view, cfg, "Written notice is required, as in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8.",
               view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["authority"] == Outcome.WARN
    assert c.evidence[0].status != Status.REJECTED


# -- citation metadata --------------------------------------------------------------
@pytest.mark.parametrize("text, why", [
    ("In Asha Devi v. Rajan Traders, 2008 SYN 5, written notice is required.", "belongs to"),
    ("Asha Devi v. Rajan Traders (2004 SYN 21) was decided in 2010 and requires written notice.", "year"),
    ("The High Court in Asha Devi v. Rajan Traders held that written notice is required.", "high court"),
])
def test_wrong_metadata_is_rejected(view, cfg, text, why):
    c = verify(view, cfg, text, view.judgment_evidence("2004 SYN 21"))
    r = c.evidence[0].checks["citation_metadata"]
    assert r.outcome == Outcome.FAIL and why in r.explanation
    assert c.evidence[0].status == Status.REJECTED


def test_correct_metadata_passes(view, cfg):
    c = verify(view, cfg, "Asha Devi v. Rajan Traders (2004 SYN 21), decided in 2004 by the Supreme Court, "
                          "requires thirty days' written notice.", view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["citation_metadata"] == Outcome.PASS


# -- statute ------------------------------------------------------------------------
def test_wrong_section_and_unknown_act_are_rejected(view, cfg):
    # Fresh evidence per claim: the verifier writes check results onto the Evidence it is given.
    wrong = verify(view, cfg, "Under Section 99 of the Sample Tenancy Act notice must be written.",
                   view.judgment_evidence("2004 SYN 21"))
    ghost = verify(view, cfg, "Under Section 12 of the Sample Housing Regulation Act notice must be written.",
                   view.judgment_evidence("2004 SYN 21"))
    assert outcomes(wrong)["statute"] == Outcome.FAIL and "Section 99" in wrong.evidence[0].checks["statute"].explanation
    assert outcomes(ghost)["statute"] == Outcome.FAIL and "does not exist" in ghost.evidence[0].checks["statute"].explanation


def test_unnamed_act_resolves_when_the_section_number_is_unique(view, cfg):
    c = verify(view, cfg, "Notice must be written under Section 12.", view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["statute"] == Outcome.PASS


def test_claim_without_a_section_is_not_applicable(view, cfg):
    c = verify(view, cfg, "Notice must be written.", view.judgment_evidence("2004 SYN 21"))
    assert outcomes(c)["statute"] == Outcome.NOT_APPLICABLE


# -- temporal -----------------------------------------------------------------------
def test_superseded_statute_version_is_flagged_and_current_one_is_not(view, cfg):
    old = verify(view, cfg, "A landlord shall not demand a security deposit exceeding two months' rent.",
                 view.statute_evidence("Sample Tenancy Act", "7", date(2010, 1, 1)))
    new = verify(view, cfg, "A landlord shall not demand a security deposit exceeding three months' rent.",
                 view.statute_evidence("Sample Tenancy Act", "7"))
    assert outcomes(old)["temporal"] == Outcome.WARN
    assert "2012-06-30" in old.evidence[0].checks["temporal"].explanation
    assert old.evidence[0].status == Status.FLAGGED
    assert outcomes(new)["temporal"] == Outcome.PASS and new.evidence[0].status == Status.VERIFIED


def test_judgment_predating_an_amendment_is_flagged(view, cfg):
    # 2011 judgment discussing Section 7; Section 7 was substituted in 2012.
    c = verify(view, cfg, "The excess over two months' rent must be refunded.",
               view.judgment_evidence("2011 SYN 33", "reasoning"))
    r = c.evidence[0].checks["temporal"]
    assert r.outcome == Outcome.WARN and "2012-07-01" in r.explanation
    # 2013 judgment discussing the amended Section 7 is not.
    c2 = verify(view, cfg, "The amendment is prospective.", view.judgment_evidence("2013 SYN 18", "reasoning"))
    assert c2.evidence[0].checks["temporal"].outcome != Outcome.WARN


def test_conflicting_versions_of_one_section_are_flagged(view, cfg):
    v1 = view.statute_evidence("Sample Tenancy Act", "7", date(2010, 1, 1))
    v2 = view.statute_evidence("Sample Tenancy Act", "7")
    claim = Claim(1, "The deposit cap is stated in Section 7.", [v1, v2])
    build_verifier(cfg, view.index).verify([claim])
    assert "conflicting versions" in claim.evidence[1].checks["temporal"].explanation


# -- claim support ------------------------------------------------------------------
def test_lexical_judge_ignores_references_and_catches_a_wrong_number():
    j = LexicalSupportJudge()
    passage = "A landlord shall not demand or accept a security deposit exceeding two months' rent."
    assert j.judge("A landlord shall not demand a security deposit exceeding two months' rent.", passage).label == "supported"
    off = j.judge("A landlord shall not demand a security deposit exceeding three months' rent.", passage)
    assert off.label == "partial" and "three" in off.reason
    noisy = "As held in Asha Devi v. Rajan Traders, 2004 SYN 21, under Section 12 of the Sample Tenancy Act, notice is written."
    assert strip_references(noisy) == "As held in , under , notice is written."


def test_arguments_sections_cannot_support_a_claim(view, cfg):
    ev = view.judgment_evidence("2004 SYN 21", "arguments")
    c = verify(view, cfg, "The respondent argued that Section 12 admits of no exception.", ev)
    assert c.evidence[0].checks["claim_support"].outcome == Outcome.WARN
    assert c.evidence[0].status == Status.FLAGGED


def test_llm_judge_requires_a_quote_that_really_occurs(view):
    ev = view.judgment_evidence("2004 SYN 21")
    ok = LLMSupportJudge(FakeLLM(responder=lambda r: json.dumps(
        {"verdict": "supported", "quote": "not maintainable", "reason": "r"})))
    lie = LLMSupportJudge(FakeLLM(responder=lambda r: json.dumps(
        {"verdict": "supported", "quote": "a sentence the passage never says", "reason": "r"})))
    junk = LLMSupportJudge(FakeLLM(responder=lambda r: "no json here"))
    assert ok.judge("claim", ev.text).label == "supported"
    assert lie.judge("claim", ev.text).label == "partial"  # invented quote: downgraded
    assert junk.judge("claim", ev.text).label == "error"
    check = ClaimSupportCheck(ok).run(Claim(1, "claim", [ev]), ev)
    start = ev.text.index("not maintainable")
    assert check.evidence_pointer == f"chunk:{ev.chunk_id}[{ev.span[0] + start}:{ev.span[0] + start + 16}]"
    assert ClaimSupportCheck(junk).run(Claim(1, "claim", [ev]), ev).outcome == Outcome.WARN  # an error is never a PASS


def test_locate_quote_tolerates_whitespace_and_case():
    assert locate_quote("THIRTY   days", "notice of thirty\ndays is required") == (10, 21)
    assert locate_quote("", "x") is None


def test_fake_llm_judge_path_runs_through_the_llm_judge(view):
    ev = view.judgment_evidence("2004 SYN 21")
    v = LLMSupportJudge(FakeLLM()).judge("Eviction proceedings without thirty days' written notice are not maintainable.", ev.text)
    assert v.label == "supported" and v.quote_span is not None


# -- rules and the verifier ---------------------------------------------------------
def test_status_rules_reject_hard_failures_and_flag_soft_ones(view, cfg):
    fn = make_status_fn(cfg)
    ev = view.judgment_evidence("2004 SYN 21")

    def with_checks(**outcome_by_name: Outcome) -> Evidence:
        ev.checks = {n: CheckResult(n, o, 1.0 if o == Outcome.PASS else 0.0) for n, o in outcome_by_name.items()}
        return ev

    assert fn(with_checks(claim_support=Outcome.PASS, temporal=Outcome.WARN))[0] == Status.FLAGGED
    assert fn(with_checks(claim_support=Outcome.PASS, statute=Outcome.FAIL))[0] == Status.REJECTED
    assert fn(with_checks(claim_support=Outcome.FAIL))[0] == Status.REJECTED
    status, why = fn(with_checks(authority=Outcome.PASS))
    assert status == Status.FLAGGED and "claim_support" in why  # cannot be VERIFIED on silence
    assert fn(with_checks(authority=Outcome.PASS, claim_support=Outcome.PASS))[0] == Status.VERIFIED


def test_include_flagged_keeps_caveated_claims_and_attaches_the_caveat(view):
    strict = VerifyConfig(kind="real", as_of=AS_OF)
    lenient = VerifyConfig(kind="real", as_of=AS_OF, include_flagged=True)
    text = "A landlord shall not demand a security deposit exceeding two months' rent."
    for cfg, survives in ((strict, False), (lenient, True)):
        claim = Claim(1, text, [view.statute_evidence("Sample Tenancy Act", "7", date(2010, 1, 1))])
        build_verifier(cfg, view.index).verify([claim])
        assert claim.survives is survives
    assert claim.support == "flagged" and "2012-06-30" in claim.caveats[0]

    res = final_answer(FakeLLM(), "deposit cap?", [claim])
    st = res.statements[0]
    assert st["status"] == "flagged" and st["caveats"] == claim.caveats
    assert st["sources"] == ["Section 7 of Sample Tenancy Act"]
    assert 0 < res.confidence < 0.5  # a caveated answer never looks as certain as a clean one


def test_verified_claim_carries_full_confidence_when_the_judge_is_certain(view):
    cfg = VerifyConfig(kind="real", as_of=AS_OF, support_judge="llm")
    llm = FakeLLM(responder=lambda r: json.dumps({"verdict": "supported", "quote": "not maintainable", "reason": "r"}))
    claim = Claim(1, "Eviction without notice is not maintainable.", [view.judgment_evidence("2004 SYN 21")])
    build_verifier(cfg, view.index, llm).verify([claim])
    assert claim.support == "verified"
    assert final_answer(FakeLLM(), "q", [claim]).confidence == 1.0


def test_real_verifier_needs_an_index():
    with pytest.raises(ValueError):
        build_verifier(VerifyConfig(kind="real"))


def test_phase0_fake_verifier_is_untouched():
    assert build_verifier(VerifyConfig()).checks[0].name == "authority"


# -- coverage guard -----------------------------------------------------------------
from legalrag.verify.query_check import unavailable_authorities  # noqa: E402


@pytest.mark.parametrize("query, expect", [
    ("What is the punishment for murder under Section 302 of the Indian Penal Code?", "no Act in the corpus has such a section"),
    ("What did the court hold in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8?", "text is not in the corpus"),
    ("Summarise the holding in Ramesh Kulkarni v. Union Housing Board, 2015 SYN 99.", "not in the citation registry"),
    ("What does Section 99 of the Sample Tenancy Act say?", "no such section"),
    ("What does Section 3 of the Sample Housing Regulation Act say?", "the Act is not in the corpus"),
])
def test_questions_naming_ungroundable_authorities_are_caught(view, query, expect):
    (reason,) = unavailable_authorities(view.index, query)
    assert expect in reason


@pytest.mark.parametrize("query", [
    "How much notice must a landlord give under Section 12 of the Sample Tenancy Act?",
    "What did Asha Devi v. Rajan Traders (2004 SYN 21) hold about oral notice?",
    "Can a tenant sublet part of the premises without the landlord's written consent?",
    "What is the capital of France?",  # out of scope, but names no authority: left to retrieval/verification
])
def test_questions_about_held_or_no_authorities_are_not_blocked(view, query):
    assert unavailable_authorities(view.index, query) == []


def test_extra_case_names_from_query_analysis_are_checked_too(view):
    assert unavailable_authorities(view.index, "what happened next?", ["Sanjay Verma v. Delta Estates"])
