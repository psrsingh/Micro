"""Runs enabled checks over every (claim, evidence) pair, assigns status, marks survivors."""

from __future__ import annotations

from typing import Callable

from legalrag.evidence import Claim, Evidence, Status
from legalrag.verify.base import Check

StatusFn = Callable[[Evidence], tuple[Status, str]]


class Verifier:
    def __init__(self, checks: list[Check], status_fn: StatusFn, min_verified: int = 1,
                 include_flagged: bool = False):
        self.checks, self.status_fn, self.min_verified = checks, status_fn, min_verified
        self.include_flagged = include_flagged

    def verify(self, claims: list[Claim]) -> None:
        for claim in claims:
            for ev in claim.evidence:
                for check in self.checks:
                    ev.checks[check.name] = check.run(claim, ev)
                ev.status, ev.status_reason = self.status_fn(ev)
            # A claim with no verified support is removed. A citation we could not
            # resolve to retrieved evidence is not support either.
            n_verified = sum(e.status == Status.VERIFIED for e in claim.evidence)
            flagged = [e for e in claim.evidence if e.status == Status.FLAGGED]
            claim.caveats = []
            if n_verified >= self.min_verified:
                claim.survives, claim.support = True, "verified"
            elif self.include_flagged and flagged:
                # Kept, but never silently: the answer must carry these caveats.
                claim.survives, claim.support = True, "flagged"
                claim.caveats = list(dict.fromkeys(e.status_reason for e in flagged if e.status_reason))
            else:
                claim.survives, claim.support = False, None
