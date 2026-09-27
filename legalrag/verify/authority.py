"""Check `authority`: does every authority involved actually exist?

Two things are checked per (claim, evidence) pair:
  * the evidence's own authority (a judgment must be in the citation registry, a statute's
    Act must be a known Act);
  * every case the claim itself names or cites. A name/citation that resolves to nothing is
    a fabricated authority (FAIL). One that exists but whose text we do not hold is WARN,
    because the claim rests on something we cannot look inside.
"""

from __future__ import annotations

from legalrag.authority import AuthorityIndex
from legalrag.evidence import CheckResult, Claim, Evidence, Outcome
from legalrag.verify.extract import find_case_refs


def pointer(ev: Evidence) -> str:
    return f"chunk:{ev.chunk_id}[{ev.span[0]}:{ev.span[1]}]"


def combine(name: str, ev: Evidence, fails: list[str], warns: list[str], oks: list[str],
            *, na_reason: str = "nothing to check") -> CheckResult:
    """Worst outcome wins; the explanation lists every finding."""
    if fails:
        return CheckResult(name, Outcome.FAIL, 0.0, pointer(ev), "; ".join(fails + warns))
    if warns:
        return CheckResult(name, Outcome.WARN, 0.5, pointer(ev), "; ".join(warns))
    if oks:
        return CheckResult(name, Outcome.PASS, 1.0, pointer(ev), "; ".join(oks))
    return CheckResult(name, Outcome.NOT_APPLICABLE, 0.0, pointer(ev), na_reason)


class AuthorityCheck:
    name = "authority"

    def __init__(self, index: AuthorityIndex):
        self.index = index

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        fails: list[str] = []
        warns: list[str] = []
        oks: list[str] = []

        if evidence.case_name:
            found = (self.index.find_citation(evidence.citation) if evidence.citation
                     else self.index.find_case(evidence.case_name))
            if found:
                oks.append(f"evidence authority {evidence.citation or evidence.case_name!r} is in the registry")
            else:
                fails.append(f"evidence authority {evidence.citation or evidence.case_name!r} is not in the registry")
        elif evidence.act_name:
            if self.index.find_act(evidence.act_name):
                oks.append(f"Act {evidence.act_name!r} exists")
            else:
                fails.append(f"Act {evidence.act_name!r} is unknown")

        own = {c.upper() for c in [evidence.citation] if c}
        for ref in find_case_refs(claim.text):
            label = ref.citation or ref.name or "?"
            by_cit = self.index.find_citation(ref.citation) if ref.citation else []
            by_name = self.index.find_case(ref.name) if ref.name else []
            entries = by_cit or by_name
            if not entries:
                fails.append(f"claim cites {label!r}, which is not in the citation registry (possible fabricated authority)")
            elif not any(e.in_corpus for e in entries) and not (ref.citation and ref.citation.upper() in own):
                warns.append(f"claim cites {label!r}: it exists but its text is not in the corpus, so it cannot be checked")
            else:
                oks.append(f"claim cites {label!r}: found in registry")
        return combine(self.name, evidence, fails, warns, oks)
