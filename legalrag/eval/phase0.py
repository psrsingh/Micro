"""Phase 0 smoke run: a handful of queries through the whole pipeline on the ten sample
documents, writing a results file. This proves plumbing; the figures are not metrics.

    python -m legalrag.eval.phase0 --config phase0
"""

from __future__ import annotations

import argparse
import statistics

from legalrag.eval.results import write_results
from legalrag.pipeline.orchestrate import Pipeline
from legalrag.settings import load_config
from legalrag.store.db import connect

# (query, fragment of the title of the document a sensible system should surface, or None)
SMOKE_QUERIES: list[tuple[str, str | None]] = [
    ("How much notice must a landlord give before seeking eviction under Section 12 of the Sample Tenancy Act?", "Asha Devi"),
    ("Can a tenant sublet part of the premises without the landlord's written consent?", "Mohan Lal"),
    ("What is the maximum security deposit a landlord may take from a tenant?", "Kavita Rao"),
    ("Is a manufacturer bound to repair a defective product during the warranty period?", "Imran Sheikh"),
    ("Within what period must a consumer complaint be filed?", "Prakash Nair"),
    ("What is the capital of France?", None),  # out of scope: should abstain
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    records = []
    with connect(cfg) as conn:
        pipe = Pipeline.from_config(cfg, conn)
        for q, expect in SMOKE_QUERIES:
            r = pipe.run(q, experiment=f"phase0-smoke-{cfg.config_hash}")
            titles = [e.case_name or f"{e.act_name} s.{e.section_number}" for e in r.kept]
            records.append({
                "run_id": r.run_id, "query": q, "outcome": r.outcome, "error": r.error,
                "n_candidates": len(r.retrieved), "n_claims": len(r.claims),
                "n_claims_surviving": sum(bool(c.survives) for c in r.claims),
                "expected_title_fragment": expect,
                "expected_in_top10": None if expect is None else any(expect in (t or "") for t in titles),
                "top3": titles[:3], "answer": r.final.statements if r.final else None,
                "latency_ms": r.latency_ms,
            })
        corpus_sha, model_info = pipe.corpus_sha, pipe.model_info

    stages = sorted({s for rec in records for s in rec["latency_ms"]})
    mean_latency = {s: round(statistics.mean(rec["latency_ms"].get(s, 0.0) for rec in records), 2) for s in stages}
    summary = {
        "n_queries": len(records),
        "answered": sum(r["outcome"] == "answered" for r in records),
        "abstained": sum(r["outcome"] == "abstained" for r in records),
        "errors": sum(r["outcome"] == "error" for r in records),
        "mean_latency_ms": mean_latency,
    }
    notes = [
        "PHASE 0 PLUMBING RUN. Fake LLM, hashed embeddings, lexical stand-in reranker and fake "
        "verifier. Nothing here measures retrieval or verification quality.",
        "Document and case names in this corpus are synthetic fixtures, not real Indian law.",
    ]
    rows = "\n".join(
        f"| {r['query'][:70]} | {r['outcome']} | {r['n_claims_surviving']}/{r['n_claims']} | "
        f"{r['expected_in_top10']} | {r['latency_ms']['total']} |" for r in records)
    body = (f"## Summary\n\n```\n{summary}\n```\n\n## Queries\n\n"
            "| query | outcome | claims surviving | expected doc in top-10 | total ms |\n|---|---|---|---|---|\n"
            f"{rows}\n")
    j, m = write_results(cfg, kind="phase0-smoke", corpus_sha=corpus_sha, model_info=model_info,
                         records=records, summary=summary, notes=notes, markdown_body=body)
    print(f"wrote {j}\nwrote {m}\n{summary}")


if __name__ == "__main__":
    main()
