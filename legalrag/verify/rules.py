"""Config-driven status rules: turns an evidence item's check results into VERIFIED /
FLAGGED / REJECTED, replacing the Phase 0 placeholder rule.

    FAIL in a `reject_on_fail` check          -> REJECTED  (does not exist / wrong section /
                                                 wrong metadata / does not support the claim)
    any other FAIL, or any WARN               -> FLAGGED   (superseded, conflicting, partial...)
    a `require_pass` check that did not PASS  -> FLAGGED   (can't be VERIFIED on silence)
    otherwise                                 -> VERIFIED
"""

from __future__ import annotations

from legalrag.evidence import Evidence, Outcome, Status
from legalrag.settings import VerifyConfig
from legalrag.verify.verifier import StatusFn


def make_status_fn(cfg: VerifyConfig) -> StatusFn:
    reject_on = set(cfg.reject_on_fail)
    require = [c for c in cfg.require_pass if c in cfg.enabled_checks]

    def status(ev: Evidence) -> tuple[Status, str]:
        by = lambda o: [n for n, r in ev.checks.items() if r.outcome == o]  # noqa: E731
        fails, warns = by(Outcome.FAIL), by(Outcome.WARN)
        hard = [n for n in fails if n in reject_on]
        if hard:
            return Status.REJECTED, "; ".join(f"{n}: {ev.checks[n].explanation}" for n in hard)
        soft = [n for n in fails + warns]
        if soft:
            return Status.FLAGGED, "; ".join(f"{n}: {ev.checks[n].explanation}" for n in soft)
        unmet = [n for n in require if n not in ev.checks or ev.checks[n].outcome != Outcome.PASS]
        if unmet:
            return Status.FLAGGED, f"required check(s) did not pass: {', '.join(unmet)}"
        return Status.VERIFIED, "all applicable checks passed"

    return status
