"""Local web UI: ask a question, compare the three arms, inspect every verification step,
and browse past runs, the corpus and saved benchmark results.

    python -m legalrag.ui.server --config offline_demo        # http://127.0.0.1:8765

Standard library only (nothing extra to install) and bound to 127.0.0.1. One database
connection and one loaded model stack sit behind a lock: this is a single-user tool for a
small machine, not a service. The frontend is plain HTML/CSS/JS in `static/`.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import psycopg

from legalrag.eval.benchmark import load_benchmark
from legalrag.eval.results import RESULTS_DIR
from legalrag.evidence import Evidence
from legalrag.pipeline.arms import ARMS, RUNNERS, ArmResult
from legalrag.pipeline.generate import source_label
from legalrag.pipeline.orchestrate import Pipeline, PipelineResult
from legalrag.settings import CONFIG_DIR, Config, load_config
from legalrag.store.db import connect

STATIC_DIR = Path(__file__).parent / "static"
STATIC_TYPES = {"index.html": "text/html; charset=utf-8", "app.css": "text/css; charset=utf-8",
                "app.js": "text/javascript; charset=utf-8"}
UI_EXPERIMENT = "ui"  # `runs.experiment` for everything asked through this UI
MAX_QUERY_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024
_RESULT_NAME = re.compile(r"^[\w.-]+\.json$")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# -- serialisation ---------------------------------------------------------------------
# Pure functions from pipeline objects (or stored rows) to JSON-ready dicts, so they can be
# tested without a server or a database.

def _snippet(text: str, n: int = 260) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1].rstrip() + "…"


def _evidence(e: Evidence) -> dict[str, Any]:
    return {**e.to_dict(), "source": source_label(e)}


def _candidate(e: dict[str, Any], kept: bool) -> dict[str, Any]:
    """One retrieved passage, without the full text (up to 50 of them per query)."""
    who = e.get("case_name") or (f"{e['act_name']} s.{e['section_number']}" if e.get("act_name") else "unknown source")
    return {"chunk_id": e["chunk_id"], "title": who, "citation": e.get("citation"), "court": e.get("court"),
            "decided_on": e.get("decided_on"), "section_label": e.get("section_label"),
            "snippet": _snippet(e.get("text", "")), "kept": kept,
            "scores": {k: e.get(f"{k}_score") for k in ("bm25", "dense", "fused", "rerank")}}


def _stages(trace: dict[str, Any]) -> dict[str, Any]:
    # The draft is logged for diagnosis and, by design, never shown to the user.
    return {k: v for k, v in trace.items() if k not in ("draft", "claims")}


def serialize_pipeline(r: PipelineResult) -> dict[str, Any]:
    kept_ids = {e.chunk_id for e in r.kept}
    f = r.final
    answered = f is not None and not f.abstained
    return {
        "arm": "verified_rag", "query": r.query, "outcome": r.outcome, "error": r.error, "run_id": r.run_id,
        "statements": f.statements if answered else [],
        "confidence": f.confidence if answered else None,
        "confidence_source": "verification" if answered and f.confidence is not None else "none",
        "note": f.used_fallback if f else None,  # why it abstained, or what was dropped
        "disclaimer": f.disclaimer if f else None,
        "caveated": answered and any(s.get("status") == "flagged" for s in f.statements),
        "claims": [{**c.to_dict(), "evidence": [_evidence(e) for e in c.evidence]} for c in r.claims],
        "stages": _stages(r.trace), "latency_ms": r.latency_ms,
        "retrieved": [_candidate(e.to_dict(), e.chunk_id in kept_ids) for e in r.retrieved],
    }


def serialize_arm(r: ArmResult) -> dict[str, Any]:
    pool_ids = {e.chunk_id for e in r.evidence_pool}
    return {
        "arm": r.arm, "query": r.query, "outcome": r.outcome, "error": r.error, "run_id": r.run_id,
        "answer_text": r.answer_text, "citations": r.citations, "confidence": r.confidence,
        "confidence_source": r.confidence_source, "note": r.notes.get("parse_error"),
        "retrieved": [_candidate(e.to_dict(), e.chunk_id in pool_ids) for e in r.retrieved],
        "latency_ms": r.latency_ms,
    }


def run_from_rows(run: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the live-result shape for a stored run. `run_items` keep the Evidence (with its
    checks); claim-level `support`/`caveats` were not logged, only the statements' copies."""
    trace, final = run["trace"] or {}, run["final_answer"]
    arm = trace.get("arm") or "verified_rag"
    retrieved = [i["evidence"] for i in items if i["stage"] == "retrieved"]
    kept_ids = {i["chunk_id"] for i in items if i["stage"] == "reranked"}
    out: dict[str, Any] = {
        "arm": arm, "query": run["query"], "outcome": run["outcome"], "error": trace.get("error"),
        "run_id": run["id"], "created_at": run["created_at"].isoformat(), "config_name": run["config_name"],
        "confidence": (final or {}).get("confidence"),
        "confidence_source": (final or {}).get("confidence_source") or ("verification" if final else "none"),
        "latency_ms": run["latency_ms"],
        "retrieved": [_candidate(e, e["chunk_id"] in kept_ids) for e in retrieved],
    }
    if arm != "verified_rag":
        out.update(answer_text=" ".join(s["text"] for s in (final or {}).get("statements", [])),
                   citations=(final or {}).get("citations", []), note=trace.get("parse_error"))
        return out
    by_claim: dict[int, list[dict[str, Any]]] = {}
    for i in items:
        if i["stage"] == "verified":
            by_claim.setdefault(i["claim_idx"], []).append({**i["evidence"], "source": _row_source(i["evidence"])})
    stages = _stages(trace)
    out.update(
        statements=(final or {}).get("statements", []), caveated=any(
            s.get("status") == "flagged" for s in (final or {}).get("statements", [])),
        note=(stages.get("final") or {}).get("note"), stages=stages,
        claims=[{"idx": c["idx"], "text": c["text"], "survives": c["survives"], "support": None, "caveats": [],
                 "unresolved_citations": c.get("unresolved_citations", []),
                 "evidence": by_claim.get(c["idx"], [])} for c in trace.get("claims", [])])
    return out


def _row_source(ev: dict[str, Any]) -> str:
    return source_label(Evidence.from_dict(ev))


# -- engine ----------------------------------------------------------------------------
class Engine:
    """The loaded pipeline plus its one connection. Every DB touch goes through `_locked`."""

    def __init__(self, config_name: str):
        self._lock = threading.RLock()
        self._stack: ExitStack | None = None
        self.cfg: Config
        self.conn: psycopg.Connection
        self.pipe: Pipeline
        self.load(config_name)

    def load(self, name: str) -> None:
        """Build the new stack first so a failed switch (missing model, mismatched database)
        leaves the current one running."""
        with self._lock:
            cfg = load_config(name)
            stack = ExitStack()
            try:
                conn = stack.enter_context(connect(cfg))
                self._check_store(conn, cfg)
                pipe = Pipeline.from_config(cfg, conn)
            except BaseException:
                stack.close()
                raise
            self.close()
            self._stack, self.cfg, self.conn, self.pipe = stack, cfg, conn, pipe

    @staticmethod
    def _check_store(conn: psycopg.Connection, cfg: Config) -> None:
        have = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM store_meta")}
        if not have:
            raise ApiError(409, "the database is empty: run `python -m legalrag.ingest.build --config "
                                f"{cfg.name}` first")
        if have.get("embedding_model") != cfg.embedding.model:
            raise ApiError(409, f"the database was built with embedding model {have.get('embedding_model')!r} but "
                                f"config {cfg.name!r} uses {cfg.embedding.model!r}; the embedding model is pinned "
                                "into the database, so that config needs a fresh database")
        conn.rollback()

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def _locked(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            try:
                return fn()
            except psycopg.OperationalError:
                self.load(self.cfg.name)  # Postgres restarted: reconnect once and retry
                return fn()
            except psycopg.Error:
                self.conn.rollback()
                raise

    def read(self, fn: Callable[[psycopg.Connection], Any]) -> Any:
        def once() -> Any:
            try:
                return fn(self.conn)
            finally:
                self.conn.rollback()  # end the read-only transaction

        return self._locked(once)

    def ask(self, query: str, arm: str) -> dict[str, Any]:
        if arm == "verified_rag":  # not RUNNERS: ArmResult would drop the claims and checks
            return self._locked(lambda: serialize_pipeline(self.pipe.run(query, UI_EXPERIMENT)))
        return self._locked(lambda: serialize_arm(RUNNERS[arm](self.pipe, query, UI_EXPERIMENT)))

    # -- reads -------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        cfg = self.cfg
        stats = self.read(lambda c: c.execute(
            "SELECT (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM chunks) AS chunks, "
            "(SELECT count(*) FROM statute_sections) AS statute_versions, "
            "(SELECT count(*) FROM citation_registry) AS registry").fetchone())
        try:
            items = load_benchmark(cfg.resolve_path(cfg.eval.benchmark_path))
            samples = [{"id": i.id, "question": i.question, "tags": i.tags, "answerable": i.answerable}
                       for i in items]
        except (OSError, ValueError):
            samples = []
        return {
            "config": {"name": cfg.name, "hash": cfg.config_hash, "llm": self.pipe.model_info["llm"],
                       "embedding": self.pipe.model_info["embedding"], "reranker": self.pipe.model_info["reranker"],
                       "verifier": self.pipe.model_info["verifier"], "high_confidence": cfg.eval.high_confidence,
                       "demo": cfg.llm.provider == "fake",
                       "real_corpus": cfg.retrieval.allowed_verification_statuses is not None},
            "configs": sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
            "arms": list(ARMS), "stats": stats, "samples": samples,
        }

    def runs(self, scope: str, limit: int) -> list[dict[str, Any]]:
        where, args = ("WHERE experiment = %s", (UI_EXPERIMENT, limit)) if scope == "ui" else ("", (limit,))
        rows = self.read(lambda c: c.execute(
            "SELECT id, experiment, config_name, query, outcome, created_at, final_answer->>'confidence' AS confidence, "
            f"latency_ms->>'total' AS total_ms FROM runs {where} ORDER BY id DESC LIMIT %s", args).fetchall())
        return [{"id": r["id"], "experiment": r["experiment"], "query": r["query"], "outcome": r["outcome"],
                 "config": r["config_name"].split(":")[0],
                 "arm": r["config_name"].split(":")[1] if ":" in r["config_name"] else "verified_rag",
                 "created_at": r["created_at"].isoformat(),
                 "confidence": None if r["confidence"] is None else float(r["confidence"]),
                 "total_ms": None if r["total_ms"] is None else float(r["total_ms"])} for r in rows]

    def run(self, run_id: int) -> dict[str, Any]:
        def q(c: psycopg.Connection) -> dict[str, Any] | None:
            run = c.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
            if run is None:
                return None
            items = c.execute("SELECT stage, claim_idx, rank, chunk_id, evidence FROM run_items WHERE run_id = %s "
                              "ORDER BY stage, claim_idx, rank", (run_id,)).fetchall()
            return run_from_rows(run, items)

        out = self.read(q)
        if out is None:
            raise ApiError(404, f"no run {run_id}")
        return out

    def corpus(self) -> dict[str, Any]:
        def q(c: psycopg.Connection) -> dict[str, Any]:
            docs = c.execute(
                "SELECT d.id, d.doc_type, d.title, d.court, d.decided_on, d.neutral_citation, count(k.id) AS n_chunks "
                "FROM documents d LEFT JOIN chunks k ON k.document_id = d.id GROUP BY d.id "
                "ORDER BY d.doc_type DESC, d.title").fetchall()
            reg = c.execute("SELECT case_name, citation, court, decided_on, in_corpus FROM citation_registry "
                            "ORDER BY in_corpus DESC, case_name").fetchall()
            return {"documents": docs, "registry": reg}

        return self.read(q)

    def document(self, doc_id: int) -> dict[str, Any]:
        def q(c: psycopg.Connection) -> dict[str, Any] | None:
            doc = c.execute("SELECT id, doc_type, title, court, decided_on, neutral_citation, reporter_citations "
                            "FROM documents WHERE id = %s", (doc_id,)).fetchone()
            if doc is None:
                return None
            doc["chunks"] = c.execute(
                "SELECT id, ordinal, section_label, text FROM chunks WHERE document_id = %s ORDER BY ordinal",
                (doc_id,)).fetchall()
            doc["versions"] = c.execute(
                "SELECT id, act_name, act_year, section_number, valid_from, valid_until, superseded_by_id, "
                "amendment_note, text FROM statute_sections WHERE document_id = %s "
                "ORDER BY section_number, valid_from", (doc_id,)).fetchall()
            return doc

        out = self.read(q)
        if out is None:
            raise ApiError(404, f"no document {doc_id}")
        return out


def list_results() -> list[dict[str, Any]]:
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("kind") != "arm-comparison":  # e.g. the phase-0 smoke run has no arms to compare
            continue
        out.append({"name": p.name, "timestamp": d["timestamp"], "config_name": d["config_name"],
                    "plumbing": any(n.startswith("PLUMBING RUN") for n in d.get("notes", [])),
                    "n_questions": len({r["id"] for r in d.get("records", [])}), "arms": list(d["summary"]["arms"])})
    return out


def get_result(name: str) -> dict[str, Any]:
    p = RESULTS_DIR / name
    if not _RESULT_NAME.match(name) or not p.is_file():
        raise ApiError(404, f"no results file {name!r}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k: d[k] for k in ("kind", "timestamp", "config_name", "model_info", "notes", "summary", "records")}


# -- http ------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "legalrag-ui"
    engine: Engine
    port: int

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - silence per-request logging
        pass

    # A page on another site can POST to 127.0.0.1 too, and an ask can spend API credits. Two
    # cheap guards: only accept our own Host header (DNS rebinding), and require a JSON
    # content type on POST (a cross-site form or fetch cannot send one without a preflight,
    # which this server never answers).
    def _check_host(self) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost"):
            raise ApiError(403, "unexpected Host header")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload, default=str).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            raise ApiError(415, "expected application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY_BYTES:
            raise ApiError(400, "missing or oversized request body")
        try:
            body = json.loads(self.rfile.read(n))
        except ValueError:
            raise ApiError(400, "body is not valid JSON") from None
        if not isinstance(body, dict):
            raise ApiError(400, "body must be a JSON object")
        return body

    def _dispatch(self, routes: list[tuple[str, Callable[..., Any]]]) -> None:
        try:
            self._check_host()
            url = urlsplit(self.path)
            for pattern, fn in routes:
                m = re.fullmatch(pattern, url.path)
                if m:
                    result = fn(self, *m.groups(), parse_qs(url.query))
                    if result is not None:
                        self._json(result)
                    return
            raise ApiError(404, "not found")
        except ApiError as e:
            self._json({"error": e.message}, e.status)
        except Exception as e:  # noqa: BLE001 - the UI shows the message instead of a dropped connection
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(GET_ROUTES)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(POST_ROUTES)

    # -- route handlers ----------------------------------------------------------------
    def _static(self, name: str | None, _q: dict) -> None:
        name = name or "index.html"
        if name not in STATIC_TYPES:
            raise ApiError(404, "not found")
        self._send(200, (STATIC_DIR / name).read_bytes(), STATIC_TYPES[name])

    def _meta(self, _q: dict) -> Any:
        return self.engine.meta()

    def _ask(self, _q: dict) -> Any:
        body = self._body()
        query, arm = body.get("query"), body.get("arm")
        if not isinstance(query, str) or not query.strip():
            raise ApiError(400, "query must be a non-empty string")
        if len(query) > MAX_QUERY_CHARS:
            raise ApiError(400, f"query is longer than {MAX_QUERY_CHARS} characters")
        if arm not in ARMS:
            raise ApiError(400, f"arm must be one of {list(ARMS)}")
        return self.engine.ask(query.strip(), arm)

    def _config(self, _q: dict) -> Any:
        name = self._body().get("name")
        if name not in {p.stem for p in CONFIG_DIR.glob("*.yaml")}:
            raise ApiError(400, "unknown config")
        try:
            self.engine.load(name)
        except ApiError:
            raise
        except Exception as e:  # noqa: BLE001 - missing extras, no Ollama, ...: report, keep the old config
            raise ApiError(409, f"could not switch to {name!r}: {type(e).__name__}: {e}") from e
        return self.engine.meta()

    def _runs(self, q: dict) -> Any:
        scope = q.get("scope", ["ui"])[0]
        if scope not in ("ui", "all"):
            raise ApiError(400, "scope must be ui or all")
        return self.engine.runs(scope, 100)

    def _run(self, run_id: str, _q: dict) -> Any:
        return self.engine.run(int(run_id))

    def _corpus(self, _q: dict) -> Any:
        return self.engine.corpus()

    def _document(self, doc_id: str, _q: dict) -> Any:
        return self.engine.document(int(doc_id))

    def _results(self, _q: dict) -> Any:
        return list_results()

    def _result(self, name: str, _q: dict) -> Any:
        return get_result(name)


GET_ROUTES: list[tuple[str, Callable[..., Any]]] = [
    (r"/(?:static/)?(index\.html|app\.css|app\.js)?", Handler._static),
    (r"/api/meta", Handler._meta),
    (r"/api/runs", Handler._runs),
    (r"/api/runs/(\d+)", Handler._run),
    (r"/api/corpus", Handler._corpus),
    (r"/api/corpus/(\d+)", Handler._document),
    (r"/api/results", Handler._results),
    (r"/api/results/([\w.-]+\.json)", Handler._result),
]
POST_ROUTES: list[tuple[str, Callable[..., Any]]] = [
    (r"/api/ask", Handler._ask),
    (r"/api/config", Handler._config),
]


def make_server(engine: Engine, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"engine": engine, "port": port})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="offline_demo", help="config to load first (switchable in the UI)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser once the server is up")
    args = ap.parse_args()
    try:
        engine = Engine(args.config)
    except ApiError as e:
        raise SystemExit(f"cannot start: {e.message}") from None
    except (psycopg.OperationalError, FileNotFoundError) as e:
        raise SystemExit(f"cannot start: {e}\n(is Postgres up? `docker compose up -d db`)") from None
    server = make_server(engine, args.port)
    url = f"http://127.0.0.1:{args.port}/"  # 127.0.0.1, not localhost: see README
    print(f"legalrag UI on {url}  (config: {args.config}; Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
