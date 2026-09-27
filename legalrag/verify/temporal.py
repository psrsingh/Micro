"""Check `temporal`: does this authority need a currency or conflict warning?

Flags (WARN, never FAIL: the synopsis scopes temporal checking to "flag for review", not to
deciding what current law is):
  * a statute chunk that is a superseded (or not yet effective) version as of `as_of`;
  * a judgment that discusses a section which was amended AFTER the judgment was decided, so
    the decision applies text that has since changed;
  * conflicting versions: the same claim rests on two different versions of one section.

Which Act a judgment's section belongs to is resolved as in the statute check; when it
cannot be determined that section is skipped rather than guessed.
"""

from __future__ import annotations

from datetime import date

from legalrag.authority import AuthorityIndex
from legalrag.evidence import CheckResult, Claim, Evidence
from legalrag.verify.authority import combine
from legalrag.verify.extract import find_section_refs
from legalrag.verify.statute import resolve_act


class TemporalCheck:
    name = "temporal"

    def __init__(self, index: AuthorityIndex, as_of: date | None = None):
        self.index = index
        self.as_of = as_of or date.today()

    def _statute_version(self, ev: Evidence):
        if ev.act_name and ev.section_number:
            return self.index.version_for_text(ev.act_name, ev.section_number, ev.text)
        return None

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        warns: list[str] = []
        oks: list[str] = []

        if evidence.act_name and evidence.section_number:
            v = self._statute_version(evidence)
            label = f"Section {evidence.section_number} of {evidence.act_name}"
            if v is None:
                warns.append(f"{label}: cannot tell which version this passage is")
            elif v.valid_from > self.as_of:
                warns.append(f"{label}: this version takes effect {v.valid_from}, after {self.as_of}")
            elif v.valid_until is not None and v.valid_until < self.as_of:
                later = [x for x in self.index.versions(evidence.act_name, evidence.section_number)
                         if x.valid_from > v.valid_from]
                warns.append(f"{label}: this version was valid only until {v.valid_until}"
                             + (f"; replaced by the version effective {later[0].valid_from}" if later else "")
                             + f" (as of {self.as_of})")
            else:
                oks.append(f"{label}: version in force as of {self.as_of}")
            for other in claim.evidence:
                if other is evidence or other.chunk_id == evidence.chunk_id:
                    continue
                if (other.act_name, other.section_number) == (evidence.act_name, evidence.section_number) \
                        and other.text != evidence.text:
                    warns.append(f"{label}: claim also cites a different version of the same section (conflicting versions)")
                    break

        elif evidence.case_name:
            if evidence.decided_on is None:
                warns.append("judgment has no decision date, so later amendments cannot be assessed")
            else:
                decided = date.fromisoformat(evidence.decided_on)
                seen: set[tuple[str, str]] = set()
                for ref in find_section_refs(evidence.text):
                    act, _ = resolve_act(self.index, ref.act.name if ref.act else None, evidence, ref.number)
                    if act is None or (act, ref.number) in seen:
                        continue
                    seen.add((act, ref.number))
                    later = [v for v in self.index.versions(act, ref.number) if v.valid_from > decided]
                    if later:
                        warns.append(f"Section {ref.number} of {act} was amended on {later[0].valid_from}, after this "
                                     f"judgment ({decided}); the decision applies the earlier text")
                    else:
                        oks.append(f"Section {ref.number} of {act} not amended since this judgment")
        return combine(self.name, evidence, [], warns, oks,
                       na_reason="no statutory version or judgment date to assess")
