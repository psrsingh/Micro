"""Verification-side behaviour added for the real corpus: 2023-code names, overlap-cannot-verify,
direct-vs-inferred support, typed pinpoint sources, the disclaimer, and config-hash stability."""

from __future__ import annotations

import json
from datetime import date

import pytest

from legalrag.authority import AuthorityIndex, RegistryEntry, SectionVersion
from legalrag.evidence import Claim, Evidence, Outcome, Status
from legalrag.llm.fake import FakeLLM
from legalrag.pipeline.generate import DISCLAIMER, final_answer, pinpoint, source_kind
from legalrag.pipeline.prompt_loader import render
from legalrag.settings import VerifyConfig, load_config
from legalrag.store.repos import MetadataFilter
from legalrag.verify.build import build_verifier
from legalrag.verify.claim_support import LexicalSupportJudge, LLMSupportJudge
from legalrag.verify.extract import find_act_mentions, find_section_refs
from legalrag.verify.statute import StatuteCheck

S103 = ("103. (1) Whoever commits murder shall be punished with death or imprisonment for life, "
        "and shall also be liable to fine.")
AS_OF = date(2026, 9, 21)


@pytest.fixture(scope="module")
def index() -> AuthorityIndex:
    return AuthorityIndex(
        [RegistryEntry("Central Bureau of Investigation v. Ashok Sirpal", "2024 INSC 819",
                       "Supreme Court of India", date(2024, 10, 24), True)],
        [SectionVersion("Bharatiya Nyaya Sanhita", 2023, "103", "", S103, date(2024, 7, 1), None)])


def bns_evidence(**kw) -> Evidence:
    base = dict(chunk_id=1, document_id=1, text=S103, span=(0, len(S103)), act_name="Bharatiya Nyaya Sanhita",
                section_number="103", section_label="statute_section", page_start=34, page_end=34,
                heading="Punishment for murder.", verification_status="official_verified",
                source_authority="official", doc_uid="ACT-BNS-2023")
    base.update(kw)
    return Evidence(**base)


# ------------------------------------------------------------------ the 2023 codes are recognised

@pytest.mark.parametrize("text,name,year", [
    ("under the Bharatiya Nyaya Sanhita, 2023 it is", "Bharatiya Nyaya Sanhita", 2023),
    ("the Bharatiya Nagarik Suraksha Sanhita, 2023 provides", "Bharatiya Nagarik Suraksha Sanhita", 2023),
    ("the Bharatiya Sakshya Adhiniyam, 2023 provides", "Bharatiya Sakshya Adhiniyam", 2023),
    ("the Sample Tenancy Act, 2001 provides", "Sample Tenancy Act", 2001),   # existing behaviour unchanged
])
def test_act_mentions_include_sanhita_and_adhiniyam(text, name, year):
    (m,) = find_act_mentions(text)
    assert (m.name, m.year) == (name, year)


def test_a_section_is_paired_with_a_sanhita_named_next_to_it():
    (r,) = find_section_refs("Section 103 of the Bharatiya Nyaya Sanhita, 2023 punishes murder")
    assert r.number == "103" and r.act.name == "Bharatiya Nyaya Sanhita"


# ------------------------------------------------------------------ the Constitution's Articles

def test_act_mentions_include_the_constitution():
    (m,) = find_act_mentions("Article 21 of the Constitution of India guarantees")
    assert (m.name, m.year) == ("Constitution of India", None)


@pytest.mark.parametrize("text,number", [
    ("Article 21 of the Constitution of India", "21"),
    ("Article 239AA of the Constitution of India", "239AA"),    # two-letter amendment insertion
    ("Article 371A of the Constitution of India", "371A"),
])
def test_article_references_are_recognised_like_sections(text, number):
    (r,) = find_section_refs(text)
    assert r.number == number and r.act.name == "Constitution of India"


def test_statute_check_passes_a_real_article_and_fails_a_nonexistent_one():
    idx = AuthorityIndex(
        [], [SectionVersion("Constitution of India", 1950, "21", "", "No person shall be deprived of life...",
                            date(1950, 1, 26), None)])
    ev = Evidence(chunk_id=1, document_id=1, text="No person shall be deprived of life", span=(0, 10),
                 act_name="Constitution of India", section_number="21", section_label="statute_section")
    ok = StatuteCheck(idx).run(Claim(1, "Article 21 of the Constitution of India protects life."), ev)
    bad = StatuteCheck(idx).run(Claim(1, "Article 999 of the Constitution of India protects life."), ev)
    assert ok.outcome == Outcome.PASS and bad.outcome == Outcome.FAIL


def test_statute_check_passes_real_section_and_fails_a_nonexistent_one(index):
    ok = StatuteCheck(index).run(Claim(1, "Section 103 of the Bharatiya Nyaya Sanhita, 2023 punishes murder."), bns_evidence())
    bad = StatuteCheck(index).run(Claim(1, "Section 999 of the Bharatiya Nyaya Sanhita, 2023 punishes murder."), bns_evidence())
    assert ok.outcome == Outcome.PASS and bad.outcome == Outcome.FAIL
    fake_act = StatuteCheck(index).run(Claim(1, "Section 103 of the Bharatiya Foo Sanhita, 2023 punishes murder."), bns_evidence())
    assert fake_act.outcome == Outcome.FAIL


# ------------------------------------------------------------------ similarity is not entailment

CLAIM = ("Section 103 of the Bharatiya Nyaya Sanhita, 2023 provides that whoever commits murder shall be punished "
         "with death or imprisonment for life.")


def _verify(index, cfg, claim_text=CLAIM, ev=None) -> Claim:
    c = Claim(1, claim_text, [ev or bns_evidence()])
    build_verifier(cfg, index).verify([c])
    return c


def test_default_lexical_judge_still_verifies_on_overlap_so_old_results_are_unchanged(index):
    c = _verify(index, VerifyConfig(kind="real", as_of=AS_OF))
    assert c.evidence[0].status == Status.VERIFIED


def test_with_lexical_can_verify_off_overlap_alone_can_only_flag(index):
    c = _verify(index, VerifyConfig(kind="real", as_of=AS_OF, lexical_can_verify=False, include_flagged=True))
    e = c.evidence[0]
    assert e.status == Status.FLAGGED and e.checks["claim_support"].outcome == Outcome.WARN
    assert "cannot verify" in e.checks["claim_support"].explanation
    assert c.survives and c.support == "flagged" and c.caveats     # surfaced, with its caveat, never as verified


def test_lexical_judge_capped_verdict_is_partial():
    j = LexicalSupportJudge(can_verify=False).judge(CLAIM, S103)
    assert j.label == "partial"
    assert LexicalSupportJudge(can_verify=True).judge(CLAIM, S103).label == "supported"


def test_overlap_cannot_catch_a_contradiction_so_it_must_not_verify_one(index):
    wrong = ("Section 103 of the Bharatiya Nyaya Sanhita, 2023 provides that whoever commits murder shall be punished "
             "with imprisonment for life and shall not be liable to fine.")
    c = _verify(index, VerifyConfig(kind="real", as_of=AS_OF, lexical_can_verify=False), wrong)
    assert c.evidence[0].status != Status.VERIFIED


# ------------------------------------------------------------------ direct text vs inferred interpretation

def _llm_judge(payload: dict, version: str = "v2") -> LLMSupportJudge:
    return LLMSupportJudge(FakeLLM(responder=lambda r: json.dumps(payload)), version)


def test_v2_judge_downgrades_inferred_support_to_partial():
    quote = "shall be punished with death or imprisonment for life"
    direct = _llm_judge({"verdict": "supported", "support_kind": "direct", "quote": quote, "reason": "r"})
    inferred = _llm_judge({"verdict": "supported", "support_kind": "inferred", "quote": quote, "reason": "r"})
    assert direct.judge("c", S103).label == "supported"
    v = inferred.judge("c", S103)
    assert v.label == "partial" and v.kind == "inferred" and "inferred" in v.reason


def test_v1_judge_output_without_support_kind_behaves_as_before():
    v1 = _llm_judge({"verdict": "supported", "quote": "shall be punished", "reason": "r"}, "v1")
    assert v1.judge("c", S103).label == "supported"


def test_inferred_support_is_flagged_not_verified_end_to_end(index):
    quote = "shall be punished with death or imprisonment for life"
    llm = FakeLLM(responder=lambda r: json.dumps(
        {"verdict": "supported", "support_kind": "inferred", "quote": quote, "reason": "applies it"}))
    cfg = VerifyConfig(kind="real", as_of=AS_OF, support_judge="llm", judge_prompt="v2", include_flagged=True)
    c = Claim(1, CLAIM, [bns_evidence()])
    build_verifier(cfg, index, llm).verify([c])
    assert c.evidence[0].status == Status.FLAGGED and c.support == "flagged"


def test_judge_v2_prompt_renders_and_asks_for_support_kind():
    text = render("judge@v2", claim="C", passage="P")
    assert "support_kind" in text and '"inferred"' in text and "C" in text and "P" in text


# ------------------------------------------------------------------ typed, pinpointed sources + disclaimer

def test_source_kinds_and_pinpoints():
    assert source_kind(bns_evidence()) == "statutory_text"
    for label, kind in [("headnote", "reporter_headnote"), ("arguments", "party_arguments"),
                        ("facts", "factual_or_procedural_background"), ("reasoning", "court_reasoning"),
                        ("holding", "court_holding"), ("judgment_text", "judgment_text_unclassified")]:
        assert source_kind(Evidence(chunk_id=1, document_id=1, text="x", span=(0, 1), section_label=label)) == kind
    assert source_kind(Evidence(chunk_id=1, document_id=1, text="x", span=(0, 1))) == "unclassified"
    assert pinpoint(bns_evidence()) == "s. 103, p. 34"
    assert pinpoint(bns_evidence(subsection="2", page_end=35)) == "s. 103(2), p. 34-35"
    j = Evidence(chunk_id=2, document_id=2, text="x", span=(0, 1), section_label="holding", paragraph_number="10", page_start=9)
    assert pinpoint(j) == "para 10, p. 9"
    assert pinpoint(Evidence(chunk_id=3, document_id=3, text="x", span=(0, 1))) == ""    # fixtures: nothing to point at


def test_final_answer_carries_typed_sources_origin_and_disclaimer(index):
    c = _verify(index, VerifyConfig(kind="real", as_of=AS_OF))
    res = final_answer(FakeLLM(), "punishment for murder?", [c])
    st = res.statements[0]
    (d,) = st["source_details"]
    assert d["kind"] == "statutory_text" and d["pinpoint"] == "s. 103, p. 34"
    assert d["verification_status"] == "official_verified" and d["doc_uid"] == "ACT-BNS-2023"
    assert st["source_types"] == ["statutory_text"] and st["origin"] == "model_generated_from_verified_claims"
    assert st["sources"] == ["Section 103 of Bharatiya Nyaya Sanhita"]     # legacy field unchanged
    assert res.disclaimer == DISCLAIMER and "not legal advice" in res.disclaimer.lower()


def test_an_abstention_also_carries_the_disclaimer():
    res = final_answer(FakeLLM(), "q", [])
    assert res.abstained and "not legal advice" in res.disclaimer.lower()


# ------------------------------------------------------------------ retrieval trust filter and config hashes

def test_metadata_filter_can_restrict_to_trusted_statuses():
    sql, params = MetadataFilter(verification_statuses=["official_verified", "supplementary_source"]).sql()
    assert "verification_status = ANY" in sql and params["f_status"] == ["official_verified", "supplementary_source"]
    assert MetadataFilter().sql() == ("", {})


def test_existing_config_hashes_are_unchanged_by_the_new_fields():
    # recorded in results/*.json and used for run comparison: must not drift when fields are added
    assert load_config("offline_demo").config_hash == "8cd7d7ee7dd36864"
    assert load_config("local_ollama").config_hash == "f53f12ace10e26f6"


def test_real_config_is_strict_and_acquisition_settings_do_not_change_the_hash():
    cfg = load_config("real_corpus")
    assert cfg.retrieval.allowed_verification_statuses == ["official_verified", "supplementary_source"]
    assert cfg.verify.lexical_can_verify is False and cfg.verify.judge_prompt == "v2"
    assert "legalrag_real" in cfg.db.dsn and "127.0.0.1" in cfg.db.dsn
    assert cfg.config_hash != load_config("offline_demo").config_hash
    other = cfg.model_copy(deep=True)
    other.corpus.download_min_interval_s = 9.0
    other.corpus.corpus_dir = "somewhere/else"
    assert other.config_hash == cfg.config_hash
    stricter = cfg.model_copy(deep=True)
    stricter.verify.lexical_can_verify = True
    assert stricter.config_hash != cfg.config_hash
