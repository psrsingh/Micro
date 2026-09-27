"""Shared interface for all verification checks. Each check is one module implementing
`Check`, so any of them can be disabled from config without touching pipeline code."""

from __future__ import annotations

from typing import Protocol

from legalrag.evidence import CheckResult, Claim, Evidence


class Check(Protocol):
    name: str

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        """Judge one (claim, evidence) pair. If the check cannot run, return
        NOT_APPLICABLE with the reason in `explanation`; never raise for a data gap."""
        ...
