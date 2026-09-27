"""Phase 0 fake verifier: proves the wiring, logging and status plumbing, nothing more.

Only `claim_support` does anything (word overlap plus a structural-section penalty) so
that PASS / WARN / FAIL all occur. The other four checks return NOT_APPLICABLE with an
honest reason until Phases 3 and 5 replace them. The status rule below is a placeholder
for the config-driven rule engine.
"""

from __future__ import annotations

import re

from legalrag.evidence import CheckResult, Claim, Evidence, Outcome, Status
from legalrag.settings import CHECK_NAMES, VerifyConfig
from legalrag.verify.base import Check

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split())


def _terms(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP}


class FakeCheck:
    def __init__(self, name: str):
        if name not in CHECK_NAMES:
            raise ValueError(f"unknown check {name!r}")
        self.name = name

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        pointer = f"chunk:{evidence.chunk_id}[{evidence.span[0]}:{evidence.span[1]}]"
        if self.name != "claim_support":
            return CheckResult(self.name, Outcome.NOT_APPLICABLE, 0.0, pointer,
                               "fake verifier: real check not implemented until a later phase")
        c, e = _terms(claim.text), _terms(evidence.text)
        overlap = len(c & e) / len(c) if c else 0.0
        if evidence.section_label == "arguments":
            return CheckResult(self.name, Outcome.WARN, min(overlap, 0.4), pointer,
                               f"overlap {overlap:.2f}; source is an 'arguments' section, not the court's view")
        outcome = Outcome.PASS if overlap >= 0.5 else Outcome.WARN if overlap >= 0.3 else Outcome.FAIL
        return CheckResult(self.name, outcome, overlap, pointer, f"word overlap {overlap:.2f}")


def build_checks(cfg: VerifyConfig) -> list[Check]:
    unknown = set(cfg.enabled_checks) - set(CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown checks in config: {sorted(unknown)}")
    return [FakeCheck(n) for n in CHECK_NAMES if n in cfg.enabled_checks]


def phase0_status(evidence: Evidence) -> tuple[Status, str]:
    """FAIL anywhere -> REJECTED; WARN anywhere -> FLAGGED; otherwise VERIFIED."""
    fails = [n for n, r in evidence.checks.items() if r.outcome == Outcome.FAIL]
    warns = [n for n, r in evidence.checks.items() if r.outcome == Outcome.WARN]
    if fails:
        return Status.REJECTED, f"failed: {', '.join(fails)}"
    if warns:
        return Status.FLAGGED, f"warned: {', '.join(warns)}"
    if not any(r.outcome == Outcome.PASS for r in evidence.checks.values()):
        return Status.FLAGGED, "no check produced a PASS"
    return Status.VERIFIED, "no check failed or warned"
