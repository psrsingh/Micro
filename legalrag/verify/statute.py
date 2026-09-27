"""Check `statute`: do the Act and section a claim invokes exist?

For each 'Section N' in the claim, the Act is taken from (in order) an Act named beside the
section, the evidence's own Act, an Act named in the evidence text, or the only Act in the
corpus that has such a section. An Act that does not exist, or a section number missing from
an Act that does, is FAIL (wrong section reference). If the Act cannot be determined and
several Acts have that section number, the result is WARN rather than a guess.
"""

from __future__ import annotations

from legalrag.authority import AuthorityIndex
from legalrag.evidence import CheckResult, Claim, Evidence
from legalrag.verify.authority import combine
from legalrag.verify.extract import find_section_refs, known_acts_in


def resolve_act(index: AuthorityIndex, ref_act: str | None, evidence: Evidence,
                section: str) -> tuple[str | None, str | None]:
    """Returns (act_name, problem). Exactly one of the two is set."""
    if ref_act:
        found = index.find_act(ref_act)
        return (found[0], None) if found else (None, f"Act {ref_act!r} does not exist")
    if evidence.act_name:
        return evidence.act_name, None
    in_text = known_acts_in(evidence.text, index)
    if len(in_text) == 1:
        return in_text[0][0], None
    candidates = index.acts_with_section(section)
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, f"no Act in the corpus has a Section {section}"
    return None, f"Section {section} exists in several Acts ({', '.join(candidates)}) and the claim names none"


class StatuteCheck:
    name = "statute"

    def __init__(self, index: AuthorityIndex):
        self.index = index

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        fails: list[str] = []
        warns: list[str] = []
        oks: list[str] = []
        for ref in find_section_refs(claim.text):
            act, problem = resolve_act(self.index, ref.act.name if ref.act else None, evidence, ref.number)
            if act is None:
                assert problem is not None
                (warns if "several Acts" in problem else fails).append(problem)
                continue
            if self.index.versions(act, ref.number):
                oks.append(f"Section {ref.number} of {act} exists")
            else:
                fails.append(f"Section {ref.number} does not exist in {act} (wrong section reference)")
        return combine(self.name, evidence, fails, warns, oks, na_reason="claim references no statutory section")
