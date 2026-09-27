"""Per-item scoring and aggregation for the synopsis's metrics.

Definitions (each is a ratio of counts, so aggregation is a sum over items, not a mean of
means):

  Answer accuracy   answer satisfies the item's must_include groups and no forbidden phrase;
                    for unanswerable items, abstaining is the correct answer.
  Recall@K          share of gold authorities (doc) / gold passages (span) present in the top K
                    reranked chunks. Undefined for the baseline (it retrieves nothing).
  CVR               cited authorities that resolve to a real authority / all cited authorities.
  CSR               valid, in-corpus cited authorities whose text supports at least one
                    statement of the answer / valid, in-corpus cited authorities.
  EGA / UCR         statements supported by the arm's own evidence pool / all statements, and
                    its complement. The baseline has an empty pool, so EGA = 0 by construction.
  AVA               see legalrag.eval.probes.
  Disclosed errors  wrong answers whose statements carry a FLAGGED caveat / wrong answers.
  HCER              incorrect answers stated with confidence >= threshold / all queries (also
                    reported over answered queries). Confidence is verification-derived for
                    verified_rag and model-stated for the others; abstentions carry none.

The auditor (`ctx.judge`) is configured separately from the pipeline's own verifier.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from legalrag.authority import AuthorityIndex, norm_citation, norm_name, norm_ws
from legalrag.eval.benchmark import BenchmarkItem
from legalrag.evidence import Evidence
from legalrag.pipeline.arms import ArmResult
from legalrag.verify.claim_support import SupportJudge
from legalrag.verify.extract import find_case_refs, find_section_refs

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class ScoringContext:
    index: AuthorityIndex
    judge: SupportJudge
    case_passages: Callable[[str | None, str | None], list[str]]  # (citation, name) -> chunk texts
    section_passages: Callable[[str, str], list[str]]  # (act, section) -> version texts
    ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    high_confidence: float = 0.8
    prefilter: int = 2  # passages per (statement, authority) actually sent to the judge


# -- helpers ---------------------------------------------------------------------------
def _overlap_rank(statement: str, passages: list[str], n: int) -> list[str]:
    s = set(_WORD.findall(statement.lower()))
    return sorted(passages, key=lambda p: -len(s & set(_WORD.findall(p.lower()))))[:n]


def statement_supported(ctx: ScoringContext, statement: str, passages: list[str]) -> bool:
    return any(ctx.judge.judge(statement, p).label == "supported"
               for p in _overlap_rank(statement, passages, ctx.prefilter))


def _has(text: str, phrase: str) -> bool:
    return norm_ws(phrase).lower() in norm_ws(text).lower()


# -- answer accuracy -------------------------------------------------------------------
def content_satisfied(item: BenchmarkItem, text: str) -> bool:
    return all(any(_has(text, alt) for alt in group) for group in item.must_include)


def answer_correct(item: BenchmarkItem, arm: ArmResult) -> tuple[bool, list[str]]:
    """Returns (correct, triggered forbidden categories)."""
    if arm.outcome == "error":
        return False, []
    if not item.answerable:
        return arm.outcome == "abstained", []
    if arm.outcome != "answered":
        return False, []
    text = arm.answer_text
    triggered = [f.category for f in item.must_not_include
                 if _has(text, f.phrase) and not any(_has(text, u) for u in f.unless_any)]
    return content_satisfied(item, text) and not triggered, triggered


# -- retrieval -------------------------------------------------------------------------
def _evidence_authority_key(e: Evidence) -> str | None:
    if e.case_name and e.citation:
        return norm_citation(e.citation)
    if e.act_name and e.section_number:
        return f"{norm_name(e.act_name)}|{e.section_number.upper()}"
    return None


def retrieval_scores(item: BenchmarkItem, arm: ArmResult, ks: list[int]) -> dict[str, float | None]:
    """Doc- and span-level recall/precision at each K over the reranked candidates."""
    out: dict[str, float | None] = {}
    if not item.answerable or arm.arm == "baseline":
        return out
    gold_keys = {g.key for g in item.gold_authorities}
    gold_pass = [(norm_ws(p.text), (norm_citation(p.authority) if "|" not in p.authority else
                                    f"{norm_name(p.authority.partition('|')[0])}|{p.authority.partition('|')[2].upper()}"))
                 for p in item.passages]

    def span_hit(e: Evidence) -> set[int]:
        key, text = _evidence_authority_key(e), norm_ws(e.text)
        return {i for i, (snip, gkey) in enumerate(gold_pass) if key == gkey and (snip in text or text in snip)}

    for k in ks:
        top = arm.retrieved[:k]
        found_docs = {key for e in top if (key := _evidence_authority_key(e)) in gold_keys}
        hit_spans = set().union(*(span_hit(e) for e in top)) if top else set()
        out[f"recall_doc@{k}"] = len(found_docs) / len(gold_keys)
        out[f"recall_span@{k}"] = len(hit_spans) / len(gold_pass)
        out[f"precision_span@{k}"] = (sum(bool(span_hit(e)) for e in top) / len(top)) if top else 0.0
    return out


# -- citations -------------------------------------------------------------------------
@dataclass
class Cited:
    label: str
    valid: bool
    category: str | None  # fabricated_citation | incorrect_metadata | wrong_section
    kind: str  # case | section | unparsed
    in_corpus: bool = False
    case_citation: str | None = None
    case_name: str | None = None
    act: str | None = None
    section: str | None = None


def extract_cited(ctx: ScoringContext, arm: ArmResult) -> list[Cited]:
    """Every authority the answer names, in its text or its citation list, resolved against
    the authority index. A citation string that parses to nothing counts as invalid: an
    answer that cites something we cannot even identify has not cited a real authority."""
    out: dict[str, Cited] = {}
    for text, is_list_entry in [(arm.answer_text, False)] + [(c, True) for c in arm.citations]:
        cases, secs = find_case_refs(text), find_section_refs(text)
        if is_list_entry and not cases and not secs:
            out.setdefault(f"unparsed:{text}", Cited(text, False, "fabricated_citation", "unparsed"))
        for r in cases:
            label = r.citation or r.name or "?"
            if label in out:
                continue
            by_cit = ctx.index.find_citation(r.citation) if r.citation else []
            by_name = ctx.index.find_case(r.name) if r.name else []
            entry = (by_cit or by_name or [None])[0]
            if entry is None:
                out[label] = Cited(label, False, "fabricated_citation", "case")
            elif r.citation and r.name and not (norm_name(entry.case_name) in norm_name(r.name)
                                                or norm_name(r.name) in norm_name(entry.case_name)):
                out[label] = Cited(label, False, "incorrect_metadata", "case")
            else:
                out[label] = Cited(label, True, None, "case", entry.in_corpus, entry.citation, entry.case_name)
        for r in secs:
            acts = [ctx.index.find_act(r.act.name)[0]] if r.act and ctx.index.find_act(r.act.name) else []
            if not acts and r.act is None:
                acts = ctx.index.acts_with_section(r.number)
            label = f"Section {r.number} of {acts[0] if len(acts) == 1 else (r.act.name if r.act else '?')}"
            if label in out:
                continue
            if acts and ctx.index.versions(acts[0], r.number):
                out[label] = Cited(label, True, None, "section", True, act=acts[0], section=r.number.upper())
            else:  # the Act does not exist, or has no such section
                out[label] = Cited(label, False, "wrong_section", "section")
    return list(out.values())


# -- the per-item score ----------------------------------------------------------------
def score_item(ctx: ScoringContext, item: BenchmarkItem, arm: ArmResult) -> dict:
    correct, forbidden = answer_correct(item, arm)
    cited = extract_cited(ctx, arm)

    stmt_supported = [statement_supported(ctx, s, [e.text for e in arm.evidence_pool]) for s in arm.statements]

    csr_total = csr_ok = 0
    wrong_proposition = False
    for c in cited:
        if not (c.valid and c.in_corpus):
            continue
        passages = (ctx.case_passages(c.case_citation, c.case_name) if c.kind == "case"
                    else ctx.section_passages(c.act or "", c.section or ""))
        csr_total += 1
        if any(statement_supported(ctx, s, passages) for s in arm.statements):
            csr_ok += 1
        else:
            wrong_proposition = True

    cats: Counter[str] = Counter(c.category for c in cited if c.category)
    cats.update(forbidden)
    if wrong_proposition:
        cats["real_case_wrong_proposition"] += 1
    if arm.statements and not all(stmt_supported):
        cats["unsupported_reasoning"] += 1
    if item.answerable and arm.outcome == "abstained":
        cats["missed_answer"] += 1
    elif not item.answerable and arm.outcome == "answered":
        cats["answered_out_of_scope"] += 1
    elif item.answerable and arm.outcome == "answered" and not content_satisfied(item, arm.answer_text):
        cats["wrong_conclusion"] += 1  # independent of any citation error the answer also has

    confident = arm.outcome == "answered" and arm.confidence is not None and arm.confidence >= ctx.high_confidence
    return {
        "id": item.id, "arm": arm.arm, "outcome": arm.outcome, "correct": correct,
        "answerable": item.answerable,
        "confidence": arm.confidence, "confidence_source": arm.confidence_source,
        "high_confidence_error": bool(confident and not correct),
        "has_confidence": arm.outcome == "answered" and arm.confidence is not None,
        "caveated": arm.caveated,
        "retrieval": retrieval_scores(item, arm, ctx.ks),
        "cited": [c.__dict__ for c in cited],
        "cited_total": len(cited), "cited_valid": sum(c.valid for c in cited),
        "csr_total": csr_total, "csr_supported": csr_ok,
        "stmts_total": len(arm.statements), "stmts_supported": sum(stmt_supported),
        "error_categories": dict(cats),
        "latency_ms": arm.latency_ms.get("total"),
    }


# -- aggregation -----------------------------------------------------------------------
def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(rows: list[dict], ks: list[int]) -> dict:
    n = len(rows)
    tot = lambda key: sum(r[key] for r in rows)  # noqa: E731
    answered = [r for r in rows if r["outcome"] == "answered"]
    wrong = [r for r in answered if not r["correct"]]
    out: dict = {
        "n_queries": n, "answered": len(answered),
        "abstained": sum(r["outcome"] == "abstained" for r in rows),
        "errors": sum(r["outcome"] == "error" for r in rows),
        "answer_accuracy": _ratio(sum(r["correct"] for r in rows), n),
        "answer_accuracy_answerable": _ratio(sum(r["correct"] for r in rows if r["answerable"]),
                                             sum(r["answerable"] for r in rows)),
        "abstention_accuracy_unanswerable": _ratio(
            sum(r["correct"] for r in rows if not r["answerable"]), sum(not r["answerable"] for r in rows)),
        "cvr": _ratio(tot("cited_valid"), tot("cited_total")),
        "invalid_citation_rate": _ratio(tot("cited_total") - tot("cited_valid"), tot("cited_total")),
        "n_cited": tot("cited_total"),
        "csr": _ratio(tot("csr_supported"), tot("csr_total")), "n_csr": tot("csr_total"),
        "ega": _ratio(tot("stmts_supported"), tot("stmts_total")),
        "ucr": _ratio(tot("stmts_total") - tot("stmts_supported"), tot("stmts_total")),
        "n_statements": tot("stmts_total"),
        "hcer": _ratio(sum(r["high_confidence_error"] for r in rows), n),
        "hcer_over_answered": _ratio(sum(r["high_confidence_error"] for r in rows), len(answered)),
        "n_with_confidence": sum(r["has_confidence"] for r in rows),
        # Wrong answers the system itself marked as flagged: errors that were disclosed to the
        # user, which plain answer accuracy scores the same as silent ones.
        "wrong_answers_disclosed": _ratio(
            sum(r["caveated"] for r in wrong), len(wrong)),
        "n_wrong_answers": len(wrong),
        "mean_latency_ms": round(sum(r["latency_ms"] or 0 for r in rows) / n, 2) if n else None,
    }
    for k in ks:
        for m in ("recall_doc", "recall_span", "precision_span"):
            vals = [r["retrieval"][f"{m}@{k}"] for r in rows if f"{m}@{k}" in r["retrieval"]]
            out[f"{m}@{k}"] = round(sum(vals) / len(vals), 4) if vals else None
    cats: Counter[str] = Counter()
    for r in rows:
        cats.update({c: 1 for c in r["error_categories"]})  # incidence: items affected
    out["error_categories"] = dict(sorted(cats.items()))
    return out


def hallucination_reduction(baseline: dict, other: dict) -> dict[str, float | None]:
    """Relative reduction vs the baseline in the rates the synopsis calls hallucination:
    invalid citations, unsupported claims, and the fabricated/incorrect/wrong-section error
    categories. Positive = fewer than baseline; None = baseline had none to reduce."""
    def red(b, a):
        return None if b in (None, 0) or a is None else round((b - a) / b, 4)

    out = {"invalid_citation_rate": red(baseline["invalid_citation_rate"], other["invalid_citation_rate"]),
           "ucr": red(baseline["ucr"], other["ucr"])}
    for cat in ("fabricated_citation", "incorrect_metadata", "wrong_section", "real_case_wrong_proposition"):
        out[cat] = red(baseline["error_categories"].get(cat, 0), other["error_categories"].get(cat, 0))
    return out


def judge_agreement(ctx: ScoringContext, item: BenchmarkItem) -> tuple[int, int]:
    """(n labelled claims, n where the audit judge agrees with the gold label). Gold is
    binary here: 'supported' vs anything else, judged against the item's gold passages. This
    measures how far the auditor can be trusted before its scores are read."""
    gold_texts = [p.text for p in item.passages]
    agree = 0
    for cl in item.claim_labels:
        agree += statement_supported(ctx, cl.text, gold_texts) == (cl.label == "supported")
    return len(item.claim_labels), agree
