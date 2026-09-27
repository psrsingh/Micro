"""The synopsis's experiment: baseline LLM vs standard RAG vs verification-enhanced RAG on
one benchmark, scored with the same metrics.

    python -m legalrag.eval.compare --config offline_demo
    python -m legalrag.eval.compare --config local_ollama --arms baseline standard_rag verified_rag

Writes results/<timestamp>-<config>.{json,md}. Every arm's runs are also logged to the
`runs` table under one experiment name.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from legalrag.authority import AuthorityIndex
from legalrag.eval.benchmark import BenchmarkItem, load_benchmark, validate_against_corpus
from legalrag.eval.metrics import ScoringContext, aggregate, hallucination_reduction, judge_agreement, score_item
from legalrag.eval.probes import load_probes, run_probes
from legalrag.eval.results import RESULTS_DIR, write_results
from legalrag.ingest.corpus_view import CorpusView
from legalrag.pipeline.arms import ARMS, RUNNERS
from legalrag.pipeline.orchestrate import Pipeline
from legalrag.settings import Config, load_config
from legalrag.store.db import connect
from legalrag.store.repos import AuthorityRepo
from legalrag.verify.build import build_support_judge


@dataclass
class _DbBenchmarkView:
    """Just enough of `CorpusView`'s interface for `validate_against_corpus`, backed by the live
    database instead of the synthetic front-matter `.txt` files. Used when `eval.corpus_source ==
    'db'` (a benchmark written against a real, DB-ingested corpus): reuses `AuthorityRepo` and
    `AuthorityIndex` wholesale, so there is exactly one implementation of 'does this authority
    exist' and 'what text backs it', file-based or DB-based."""

    index: AuthorityIndex
    repo: AuthorityRepo

    def judgment(self, citation_or_name: str) -> object | None:
        return self.passages_for_case(citation_or_name) or None

    def passages_for_section(self, act_name: str, section: str) -> list[str]:
        return self.repo.section_passages(act_name, section)

    def passages_for_case(self, citation_or_name: str) -> list[str]:
        return self.repo.case_passages(citation_or_name, None) or self.repo.case_passages(None, citation_or_name)

_LABELS = {
    "answer_accuracy": "Answer accuracy", "answer_accuracy_answerable": "  on answerable questions",
    "abstention_accuracy_unanswerable": "  abstained when it should", "cvr": "Citation Validity Rate (CVR)",
    "csr": "Citation Support Rate (CSR)", "ega": "Evidence Grounding Accuracy (EGA)",
    "ucr": "Unsupported Claim Rate (UCR) - lower is better", "hcer": "High-Confidence Error Rate (HCER) - lower is better",
    "hcer_over_answered": "  HCER over answered queries only",
    "wrong_answers_disclosed": "Wrong answers disclosed as flagged (higher is better)",
}


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _report(cfg: Config, arms: list[str], summary: dict, rows: list[dict], ava: dict | None,
            agreement: tuple[int, int]) -> str:
    s = summary["arms"]
    parts = ["## Comparison\n", _table(["metric"] + arms, [
        [label] + [s[a][key] for a in arms] for key, label in _LABELS.items()]),
        "\nCounts behind the ratios: " + ", ".join(
            f"{a}: {s[a]['n_cited']} cited / {s[a]['n_csr']} checkable / {s[a]['n_statements']} statements / "
            f"{s[a]['n_with_confidence']} with confidence / {s[a]['n_wrong_answers']} wrong answers" for a in arms)]
    parts += ["\n## Retrieval (over answerable questions; the baseline retrieves nothing)\n",
              _table(["metric"] + arms, [[m] + [s[a].get(m) for a in arms] for m in
                                         [f"{p}@{k}" for k in cfg.eval.recall_ks
                                          for p in ("recall_doc", "recall_span", "precision_span")]])]
    cats = sorted({c for a in arms for c in s[a]["error_categories"]})
    parts += ["\n## Error analysis (questions affected per category)\n",
              _table(["category"] + arms, [[c] + [s[a]["error_categories"].get(c, 0) for a in arms] for c in cats])
              if cats else "No errors recorded."]
    if "baseline" in arms and len(arms) > 1:
        red = summary["hallucination_reduction_vs_baseline"]
        keys = sorted({k for a in red.values() for k in a})
        parts += ["\n## Reduction vs baseline (positive = fewer errors; n/a = baseline had none)\n",
                  _table(["error type"] + [a for a in arms if a != "baseline"],
                         [[k] + [red[a].get(k) for a in arms if a != "baseline"] for k in keys])]
    if ava:
        parts += [f"\n## Authority Verification Accuracy (AVA)\n\nAVA = {_fmt(ava['ava'])} over {ava['n']} constructed probes "
                  f"({ava['confusion']}).\n\n" + _table(
                      ["probe kind", "n", "accuracy"], [[k, v["n"], v["accuracy"]] for k, v in ava["by_kind"].items()])]
    n, agree = agreement
    parts.append(f"\n## Audit judge check\n\nThe `{cfg.eval.audit_judge}` audit judge agrees with the gold claim labels on "
                 f"{agree}/{n} labelled claims. EGA, UCR and CSR are only as trustworthy as this.")
    parts.append("\n## Per question\n\n" + _table(
        ["id"] + [f"{a}: outcome / correct / conf" for a in arms],
        [[qid] + [f"{r['outcome']} / {'yes' if r['correct'] else 'NO'} / {_fmt(r['confidence'])}"
                  for a in arms for r in rows if r["id"] == qid and r["arm"] == a]
         for qid in dict.fromkeys(r["id"] for r in rows)]))
    return "\n".join(parts) + "\n"


def run(cfg: Config, arms: list[str], limit: int | None = None,
        out_dir: Path = RESULTS_DIR) -> tuple[Path, Path, dict]:
    items: list[BenchmarkItem] = load_benchmark(cfg.resolve_path(cfg.eval.benchmark_path))[:limit]
    # AVA is a fixed regression suite over constructed cases (see EvalConfig.corpus_source): it
    # always reads the synthetic fixtures, whatever corpus the benchmark itself is checked against.
    probes_view = CorpusView(cfg.resolve_path(cfg.ingest.corpus_dir))

    experiment = f"compare-{cfg.name}-{cfg.config_hash}"
    rows: list[dict] = []
    with connect(cfg) as conn:
        pipe = Pipeline.from_config(cfg, conn)
        repo = AuthorityRepo(conn)
        index = repo.load()
        bench_view = _DbBenchmarkView(index, repo) if cfg.eval.corpus_source == "db" else probes_view
        problems = validate_against_corpus(items, bench_view)
        if problems:
            raise ValueError("benchmark does not match the corpus:\n  " + "\n  ".join(problems))
        ctx = ScoringContext(
            index=index, judge=build_support_judge(cfg.verify, cfg.eval.audit_judge, pipe.llm),
            case_passages=repo.case_passages, section_passages=repo.section_passages,
            ks=cfg.eval.recall_ks, high_confidence=cfg.eval.high_confidence)
        for item in items:
            for arm in arms:
                rows.append(score_item(ctx, item, RUNNERS[arm](pipe, item.question, experiment)))
        agreement = tuple(map(sum, zip(*(judge_agreement(ctx, i) for i in items)))) if items else (0, 0)
        ava = None
        if "verified_rag" in arms and cfg.verify.kind == "real":
            ava = run_probes(load_probes(cfg.resolve_path(cfg.eval.probes_path)), probes_view.index, probes_view)
        corpus_sha, model_info = pipe.corpus_sha, {**pipe.model_info, "audit_judge": cfg.eval.audit_judge}

    per_arm = {a: aggregate([r for r in rows if r["arm"] == a], cfg.eval.recall_ks) for a in arms}
    summary: dict = {"arms": per_arm, "ava": None if ava is None else {k: v for k, v in ava.items() if k != "rows"},
                     "audit_judge_agreement": {"n": agreement[0], "agree": agreement[1]}}
    if "baseline" in per_arm:
        summary["hallucination_reduction_vs_baseline"] = {
            a: hallucination_reduction(per_arm["baseline"], per_arm[a]) for a in arms if a != "baseline"}

    notes = [
        "Answer accuracy is graded by phrase rules in the benchmark (must_include / must_not_include), not by a "
        "model: it is transparent but coarse.",
        f"HCER threshold: confidence >= {cfg.eval.high_confidence}. Confidence is verification-derived for "
        "verified_rag and model-stated for the other arms, so HCER compares two different kinds of confidence.",
        "The authority-verification probes were written together with the checks: AVA is a regression "
        "suite over constructed cases, not a held-out estimate" +
        (" (it always runs over the synthetic fixtures, independently of the corpus below)."
         if cfg.eval.corpus_source == "db" else "."),
        "Document and case names in this corpus are synthetic fixtures, not real Indian law."
        if cfg.eval.corpus_source == "files" else
        "This benchmark is checked against the real, DB-ingested corpus (eval.corpus_source: db): "
        "see the corpus report for source authority and verification status of every document.",
    ]
    if cfg.llm.provider == "fake":
        notes.insert(0, "PLUMBING RUN. The generator is a deterministic fake; nothing here measures model quality.")
    if cfg.llm.provider == "anthropic":
        notes.append("The API-backed generator cannot be run at temperature 0 (see llm.anthropic_api): "
                     "repeat runs may differ.")
    if cfg.eval.audit_judge == "lexical":
        notes.append("The audit judge is lexical word-overlap, which cannot detect a contradiction: "
                     "EGA/UCR/CSR are approximate. See the audit-judge check below.")
    body = _report(cfg, arms, summary, rows, ava, agreement)
    j, m = write_results(cfg, kind="arm-comparison", corpus_sha=corpus_sha, model_info=model_info,
                         records=rows, summary=summary, notes=notes, markdown_body=body, out_dir=out_dir)
    return j, m, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N benchmark items")
    args = ap.parse_args()
    cfg = load_config(args.config)
    j, m, summary = run(cfg, args.arms or list(cfg.eval.arms), args.limit)
    print(f"wrote {j}\nwrote {m}")
    for arm, s in summary["arms"].items():
        print(f"{arm:14s} accuracy={_fmt(s['answer_accuracy'])} cvr={_fmt(s['cvr'])} csr={_fmt(s['csr'])} "
              f"ega={_fmt(s['ega'])} ucr={_fmt(s['ucr'])} hcer={_fmt(s['hcer'])}")


if __name__ == "__main__":
    main()
