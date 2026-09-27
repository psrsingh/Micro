"""The web UI: serialisers (no database) and the HTTP surface (real server, throwaway Postgres)."""

from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timezone

import pytest

from legalrag.ui.server import ApiError, Engine, get_result, make_server, run_from_rows

QUERY = "How much notice must a landlord give before seeking eviction under Section 12 of the Sample Tenancy Act?"


# -- serialisers: no database ----------------------------------------------------------
def _row(**kw):
    base = {"id": 7, "query": "q", "outcome": "answered", "config_name": "offline_demo",
            "created_at": datetime(2026, 9, 21, tzinfo=timezone.utc), "latency_ms": {"total": 1.0},
            "trace": {}, "final_answer": None}
    return {**base, **kw}


def test_stored_verified_run_is_rebuilt_without_the_hidden_draft():
    ev = {"chunk_id": 1, "document_id": 1, "text": "t", "span": [0, 1], "case_name": "A v. B", "citation": "2001 SYN 1",
          "status": "VERIFIED", "checks": {}}
    items = [{"stage": "retrieved", "claim_idx": None, "rank": 1, "chunk_id": 1, "evidence": ev},
             {"stage": "reranked", "claim_idx": None, "rank": 1, "chunk_id": 1, "evidence": ev},
             {"stage": "verified", "claim_idx": 1, "rank": 1, "chunk_id": 1, "evidence": ev}]
    trace = {"draft": "SECRET DRAFT", "final": {"note": None}, "claims": [
        {"idx": 1, "text": "c", "survives": True, "evidence_chunk_ids": [1], "unresolved_citations": []}]}
    out = run_from_rows(_row(trace=trace, final_answer={"abstained": False, "confidence": 0.9, "statements": [
        {"text": "s", "claim_ids": [1], "status": "verified", "caveats": [], "sources": ["A v. B (2001 SYN 1)"]}]}), items)
    assert out["arm"] == "verified_rag" and out["confidence_source"] == "verification"
    assert "draft" not in out["stages"] and "SECRET" not in json.dumps(out)
    assert out["claims"][0]["evidence"][0]["source"] == "A v. B (2001 SYN 1)"
    assert out["retrieved"][0]["kept"] is True


def test_stored_arm_run_keeps_its_arm_and_unchecked_citations():
    out = run_from_rows(_row(config_name="offline_demo:standard_rag", trace={"arm": "standard_rag"}, final_answer={
        "abstained": False, "statements": [{"text": "One.", "claim_ids": []}, {"text": "Two.", "claim_ids": []}],
        "citations": ["X v. Y"], "confidence": 0.6, "confidence_source": "model_stated"}), [])
    assert out["arm"] == "standard_rag" and out["answer_text"] == "One. Two."
    assert out["citations"] == ["X v. Y"] and out["confidence_source"] == "model_stated"


def test_abstained_stored_run_has_no_confidence():
    out = run_from_rows(_row(outcome="abstained"), [])
    assert out["confidence"] is None and out["confidence_source"] == "none"


@pytest.mark.parametrize("name", ["../pyproject.toml", "..\\x.json", "a/b.json", "nope.json", "x.txt"])
def test_results_names_cannot_escape_the_results_dir(name):
    with pytest.raises(ApiError) as e:
        get_result(name)
    assert e.value.status == 404


# -- HTTP: real server, throwaway database ---------------------------------------------
@pytest.fixture()
def server(test_db_cfg):
    engine = Engine("offline_demo")  # same hash embedder as the throwaway database
    srv = make_server(engine, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], engine
    srv.shutdown()
    srv.server_close()
    engine.close()


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    hdrs = {"Content-Type": "application/json", **(headers or {})} if method == "POST" else dict(headers or {})
    c.request(method, path, body=None if body is None else json.dumps(body), headers=hdrs)
    r = c.getresponse()
    raw = r.read()
    c.close()
    ctype = r.getheader("Content-Type", "")
    return r.status, (json.loads(raw) if ctype.startswith("application/json") else raw.decode())


@pytest.mark.db
def test_serves_the_page_and_meta(server):
    port, _ = server
    status, html = call(port, "GET", "/")
    assert status == 200 and "<title>" in html
    assert call(port, "GET", "/app.js")[0] == 200
    assert call(port, "GET", "/static/../pyproject.toml")[0] == 404
    status, meta = call(port, "GET", "/api/meta")
    assert status == 200 and meta["config"]["name"] == "offline_demo" and meta["config"]["demo"] is True
    assert meta["stats"]["documents"] > 0 and meta["arms"] == ["baseline", "standard_rag", "verified_rag"]
    # The example questions carry no gold answers or passages.
    assert meta["samples"] and all(set(s) == {"id", "question", "tags", "answerable"} for s in meta["samples"])


@pytest.mark.db
def test_ask_each_arm_and_verified_trace(server):
    port, _ = server
    for arm in ("baseline", "standard_rag"):
        status, r = call(port, "POST", "/api/ask", {"query": QUERY, "arm": arm})
        assert status == 200 and r["arm"] == arm and r["outcome"] in ("answered", "abstained")
    status, r = call(port, "POST", "/api/ask", {"query": QUERY, "arm": "verified_rag"})
    assert status == 200 and r["outcome"] == "answered" and r["confidence_source"] == "verification"
    assert r["statements"] and r["claims"] and r["claims"][0]["evidence"][0]["checks"]
    assert "draft" not in r["stages"] and "total" in r["latency_ms"]
    assert 0 < len(r["retrieved"]) and any(p["kept"] for p in r["retrieved"])


@pytest.mark.db
def test_coverage_guard_abstention_is_explained(server):
    port, _ = server
    _, r = call(port, "POST", "/api/ask", {"query": "What did the court hold in Bhaskar Iyer v. Coastal Lettings, 2002 SYN 8?",
                                           "arm": "verified_rag"})
    assert r["outcome"] == "abstained" and "cannot ground" in r["note"] and r["stages"]["unavailable_authorities"]


@pytest.mark.db
def test_history_reopens_what_was_asked(server):
    port, _ = server
    _, live = call(port, "POST", "/api/ask", {"query": QUERY, "arm": "verified_rag"})
    _, runs = call(port, "GET", "/api/runs?scope=ui")
    assert runs[0]["id"] == live["run_id"] and runs[0]["arm"] == "verified_rag"
    status, stored = call(port, "GET", f"/api/runs/{live['run_id']}")
    assert status == 200 and len(stored["claims"]) == len(live["claims"])
    assert [s["text"] for s in stored["statements"]] == [s["text"] for s in live["statements"]]
    assert stored["claims"][0]["evidence"][0]["checks"].keys() == live["claims"][0]["evidence"][0]["checks"].keys()
    assert call(port, "GET", "/api/runs/999999999")[0] == 404


@pytest.mark.db
def test_corpus_endpoints(server):
    port, _ = server
    _, corpus = call(port, "GET", "/api/corpus")
    statute = next(d for d in corpus["documents"] if d["doc_type"] == "statute")
    _, doc = call(port, "GET", f"/api/corpus/{statute['id']}")
    assert doc["chunks"] and doc["versions"]
    assert call(port, "GET", "/api/corpus/999999999")[0] == 404


@pytest.mark.db
@pytest.mark.parametrize("body,fragment", [
    ({"query": "", "arm": "baseline"}, "non-empty"), ({"query": "x", "arm": "nope"}, "arm must be"),
    ({"query": "x" * 2001, "arm": "baseline"}, "longer than"), ({"arm": "baseline"}, "non-empty")])
def test_ask_validation(server, body, fragment):
    status, r = call(server[0], "POST", "/api/ask", body)
    assert status == 400 and fragment in r["error"]


@pytest.mark.db
def test_cross_site_requests_are_refused(server):
    port, _ = server
    assert call(port, "GET", "/api/meta", headers={"Host": "evil.example.com"})[0] == 403  # DNS rebinding
    status, _ = call(port, "POST", "/api/ask", headers={"Content-Type": "text/plain"}, body={"query": "x", "arm": "baseline"})
    assert status == 415  # what a cross-site <form> or no-cors fetch can send


@pytest.mark.db
def test_failed_config_switch_keeps_the_running_one(server):
    port, engine = server
    # local_ollama needs a different embedding model than this database was built with.
    status, r = call(port, "POST", "/api/config", {"name": "local_ollama"})
    assert status == 409 and "embedding" in r["error"]
    assert engine.cfg.name == "offline_demo"
    assert call(port, "POST", "/api/ask", {"query": QUERY, "arm": "baseline"})[0] == 200
    assert call(port, "POST", "/api/config", {"name": "../x"})[0] == 400
