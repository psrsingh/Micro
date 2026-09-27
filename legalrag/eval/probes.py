"""Authority Verification Accuracy (AVA): how well do the authority-existence checks tell
valid authorities from invalid ones?

A probe pairs a claim with a real evidence chunk and a label saying whether the authority the
claim invokes is valid. The verifier's three authority checks (authority, citation_metadata,
statute) run on it; the probe is predicted INVALID when any of them FAILs. This isolates the
verification layer from retrieval and generation: the probes are constructed, not generated.

    {"id": "p01", "claim": "...", "evidence": {"citation": "2004 SYN 21", "label": "holding"},
     "label": "valid" | "invalid", "kind": "real | fabricated_case | wrong_metadata | ..."}
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from legalrag.authority import AuthorityIndex
from legalrag.evidence import Claim, Outcome
from legalrag.ingest.corpus_view import CorpusView
from legalrag.verify.authority import AuthorityCheck
from legalrag.verify.citation_metadata import CitationMetadataCheck
from legalrag.verify.statute import StatuteCheck

AUTHORITY_CHECKS = ("authority", "citation_metadata", "statute")


class ProbeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str | None = None  # judgment evidence
    act: str | None = None  # statute evidence
    section: str | None = None
    label: str = "holding"
    on: date | None = None


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    claim: str
    evidence: ProbeEvidence
    label: Literal["valid", "invalid"]
    kind: str = "real"
    notes: str = Field(default="")


def load_probes(path: Path) -> list[Probe]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip() and not line.lstrip().startswith("#"):
                try:
                    out.append(Probe.model_validate(json.loads(line)))
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"{path}:{n}: {e}") from e
    return out


def run_probes(probes: list[Probe], index: AuthorityIndex, view: CorpusView) -> dict[str, Any]:
    checks = [AuthorityCheck(index), CitationMetadataCheck(index), StatuteCheck(index)]
    rows, tp, tn, fp, fn = [], 0, 0, 0, 0
    by_kind: dict[str, list[bool]] = {}
    for p in probes:
        ev = (view.judgment_evidence(p.evidence.citation, p.evidence.label) if p.evidence.citation
              else view.statute_evidence(p.evidence.act or "", p.evidence.section or "", p.evidence.on))
        claim = Claim(1, p.claim, [ev])
        results = {c.name: c.run(claim, ev) for c in checks}
        failed = [n for n, r in results.items() if r.outcome == Outcome.FAIL]
        predicted = "invalid" if failed else "valid"
        ok = predicted == p.label
        tp += p.label == "invalid" and predicted == "invalid"
        tn += p.label == "valid" and predicted == "valid"
        fp += p.label == "valid" and predicted == "invalid"  # a valid authority wrongly rejected
        fn += p.label == "invalid" and predicted == "valid"  # an invalid authority let through
        by_kind.setdefault(p.kind, []).append(ok)
        rows.append({"id": p.id, "kind": p.kind, "label": p.label, "predicted": predicted, "correct": ok,
                     "failed_checks": failed,
                     "explanations": {n: r.explanation for n, r in results.items() if r.outcome != Outcome.PASS}})
    n = len(probes)
    return {
        "ava": round((tp + tn) / n, 4) if n else None, "n": n,
        "confusion": {"invalid_caught": tp, "valid_passed": tn,
                      "valid_wrongly_rejected": fp, "invalid_missed": fn},
        "by_kind": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(by_kind.items())},
        "rows": rows,
    }
