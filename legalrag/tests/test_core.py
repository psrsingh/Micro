"""Unit tests that need no database: config, evidence, parsing, chunking, LLM adapters."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from legalrag.evidence import CheckResult, Evidence, Outcome, Status
from legalrag.ingest.chunk import chunk_span
from legalrag.ingest.parse import ParseError, parse_document
from legalrag.llm import LLMRequest, parse_structured
from legalrag.llm.anthropic_api import AnthropicAdapter, accepts_sampling, supports_effort
from legalrag.llm.base import extract_json
from legalrag.llm.fake import FakeLLM
from legalrag.llm.ollama import OllamaAdapter
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import Config, LLMConfig, load_config

CORPUS = Path(__file__).parent / "fixtures" / "sample_corpus"


# -- config ------------------------------------------------------------------------
def test_config_hash_is_stable_and_sensitive():
    a, b = load_config("phase0"), load_config("phase0")
    assert a.config_hash == b.config_hash
    changed = a.model_copy(update={"seed": 999})
    assert changed.config_hash != a.config_hash


def test_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        Config.model_validate({"name": "x", "bogus": 1})


# -- evidence ----------------------------------------------------------------------
def test_evidence_roundtrip_preserves_everything():
    ev = Evidence(chunk_id=3, document_id=1, text="t", span=(10, 20), case_name="A v. B",
                  section_label="holding", bm25_score=0.1, fused_score=0.03, status=Status.FLAGGED,
                  status_reason="why")
    ev.checks["authority"] = CheckResult("authority", Outcome.WARN, 0.5, "chunk:3[10:20]", "outside corpus")
    back = Evidence.from_dict(json.loads(json.dumps(ev.to_dict())))
    assert back == ev


def test_check_score_must_be_unit_interval():
    with pytest.raises(ValueError):
        CheckResult("x", Outcome.PASS, 1.5)


# -- parsing and chunking ----------------------------------------------------------
def test_all_fixture_documents_parse_with_exact_offsets():
    files = sorted(CORPUS.glob("*.txt"))
    assert len(files) == 10
    for f in files:
        doc = parse_document(f)
        assert doc.sections, f.name
        for s in doc.sections:
            body = doc.raw_text[s.start:s.end]
            assert body and body == body.strip()
            assert not body.startswith("##")


def test_judgment_sections_are_labelled_structurally():
    doc = parse_document(CORPUS / "j01_asha_devi_v_rajan_traders.txt")
    assert [s.label for s in doc.sections] == ["headnote", "facts", "arguments", "reasoning", "holding"]


def test_statute_versions_link_and_dates_parse():
    doc = parse_document(CORPUS / "s01_sample_tenancy_act_2001.txt")
    s7 = [s.statute for s in doc.sections if s.statute.section_number == "7"]
    assert [v.valid_until for v in s7] == [date(2012, 6, 30), None]
    assert "Amendment" in (s7[1].amendment_note or "")


def test_overlapping_statute_versions_are_rejected(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text(
        "---\ndoc_type: statute\ntitle: T\nact_name: A\nact_year: 2000\n---\n"
        "## SECTION 1 [valid 2000-01-01 to 2010-01-01]\nold\n"
        "## SECTION 1 [valid 2009-01-01 to present]\nnew\n", encoding="utf-8")
    with pytest.raises(ParseError, match="overlap"):
        parse_document(p)


def test_chunker_never_exceeds_max_and_offsets_are_verbatim():
    text = " ".join(f"w{i}" for i in range(1000))
    spans = chunk_span(text, 0, len(text), target=200, max_tokens=400, overlap=40)
    assert len(spans) > 1
    assert all(s.n_tokens <= 400 for s in spans)
    for a, b in zip(spans, spans[1:]):
        assert b.start < a.end  # overlap
        assert b.start > a.start
    assert spans[0].start == 0 and spans[-1].end == len(text)
    for s in spans:
        assert len(text[s.start:s.end].split()) == s.n_tokens


def test_short_section_is_a_single_chunk():
    text = "one two three"
    assert len(chunk_span(text, 0, len(text), target=200, max_tokens=400, overlap=40)) == 1


def test_chunker_respects_section_bounds():
    text = "aaa bbb ccc ddd eee"
    (s,) = chunk_span(text, 4, 11, target=10, max_tokens=10, overlap=1)
    assert text[s.start:s.end] == "bbb ccc"


# -- prompts -----------------------------------------------------------------------
def test_every_prompt_renders_and_rejects_bad_variables():
    assert "hello" in render(PROMPT_IDS["query_analysis"], query="hello")
    with pytest.raises(KeyError):
        render(PROMPT_IDS["query_analysis"], nope="x")
    with pytest.raises(KeyError):
        render(PROMPT_IDS["draft"], query="q")  # evidence placeholder left unfilled


# -- structured output -------------------------------------------------------------
@pytest.mark.parametrize("text", [
    '{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go: {"a": 1} hope that helps'])
def test_extract_json_tolerates_fences_and_prose(text: str):
    assert extract_json(text) == {"a": 1}


def test_parse_structured_reports_errors_instead_of_raising():
    val, err = parse_structured("not json", QueryAnalysis)
    assert val is None and err
    val, err = parse_structured('{"issue": "x"}', QueryAnalysis)  # missing rewritten_query
    assert val is None and err


# -- adapters ----------------------------------------------------------------------
def test_ollama_adapter_sends_temperature_zero_and_seed_and_reads_digest():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:1", "digest": "sha256:abc"}]})
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"model": "m:1", "message": {"content": "hi"},
                                         "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 1})

    a = OllamaAdapter(LLMConfig(provider="ollama", model="m:1", seed=7),
                      transport=httpx.MockTransport(handler))
    r = a.complete(LLMRequest(task="draft", prompt="p", json_output=True))
    assert r.text == "hi" and r.output_tokens == 1
    assert seen["options"]["temperature"] == 0.0 and seen["options"]["seed"] == 7
    assert seen["format"] == "json" and seen["stream"] is False
    assert a.describe()["digest"] == "sha256:abc"


class _StubAnthropic:
    def __init__(self):
        self.kwargs: dict = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="ok")],
            model=kw["model"], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=5, output_tokens=2), _request_id="req_1")


def test_anthropic_adapter_drops_sampling_params_on_current_models():
    stub = _StubAnthropic()
    a = AnthropicAdapter(LLMConfig(provider="anthropic", model="claude-opus-5", effort="low"), client=stub)
    r = a.complete(LLMRequest(task="draft", prompt="p", system="s"))
    assert "temperature" not in stub.kwargs and "top_p" not in stub.kwargs
    assert stub.kwargs["output_config"] == {"effort": "low"}
    assert r.text == "ok" and r.request_id == "req_1"
    assert "temperature" in r.params_dropped and "seed" in r.params_dropped
    assert a.describe()["deterministic_sampling"] is False


def test_anthropic_adapter_sends_temperature_where_accepted_and_no_effort_on_haiku():
    stub = _StubAnthropic()
    a = AnthropicAdapter(LLMConfig(provider="anthropic", model="claude-haiku-4-5", effort="low"), client=stub)
    a.complete(LLMRequest(task="draft", prompt="p"))
    assert stub.kwargs["temperature"] == 0.0
    assert "output_config" not in stub.kwargs


@pytest.mark.parametrize("model,sampling,effort", [
    ("claude-opus-5", False, True), ("claude-sonnet-5", False, True), ("claude-fable-5-1", False, True),
    ("claude-opus-4-8", False, True), ("claude-haiku-4-5", True, False),
    ("claude-sonnet-4-6", True, True), ("claude-opus-4-6", True, True)])
def test_model_capability_table(model: str, sampling: bool, effort: bool):
    assert accepts_sampling(model) is sampling
    assert supports_effort(model) is effort


def test_fake_llm_pipeline_tasks_are_wired():
    llm = FakeLLM()
    qa = json.loads(llm.complete(LLMRequest(task="query_analysis", prompt="QUESTION:\nSection 12 notice?")).text)
    assert qa["section_numbers"] == ["12"]
    passages = "[E1] A v. B; 2000 SYN 1; holding :: Notice must be written. More text.\n"
    draft = llm.complete(LLMRequest(task="draft", prompt=f"{passages}QUESTION:\nIs notice needed?")).text
    assert draft == "Notice must be written. [E1]"
    off_topic = llm.complete(LLMRequest(task="draft", prompt=f"{passages}QUESTION:\nWhat is the capital of France?"))
    assert off_topic.text == ""  # passages that do not touch the question yield no statements
    claims = json.loads(llm.complete(LLMRequest(task="decompose", prompt=f"DRAFT:\n{draft}")).text)
    assert claims["claims"] == [{"text": "Notice must be written.", "citations": ["E1"]}]
