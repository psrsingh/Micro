"""Check `citation_metadata`: are the case name, citation, court and year stated for an
authority the ones on record?

  * The evidence's own metadata must agree with its registry record (guards against a
    corrupted or mis-ingested document).
  * When the claim names a case or gives a citation, that name/citation pair must match
    the registry, and any year or court written next to it must match too. A real citation
    attached to the wrong case name is exactly the "incorrect metadata" error class.
"""

from __future__ import annotations

from legalrag.authority import AuthorityIndex, norm_name
from legalrag.evidence import CheckResult, Claim, Evidence
from legalrag.verify.authority import combine
from legalrag.verify.extract import find_case_refs, find_courts, stated_decision_year


class CitationMetadataCheck:
    name = "citation_metadata"

    def __init__(self, index: AuthorityIndex):
        self.index = index

    def run(self, claim: Claim, evidence: Evidence) -> CheckResult:
        fails: list[str] = []
        warns: list[str] = []
        oks: list[str] = []

        if evidence.case_name and evidence.citation:
            for e in self.index.find_citation(evidence.citation)[:1]:
                bad = []
                if norm_name(e.case_name) != norm_name(evidence.case_name):
                    bad.append(f"name {evidence.case_name!r} != registry {e.case_name!r}")
                if evidence.court and e.court and evidence.court != e.court:
                    bad.append(f"court {evidence.court!r} != registry {e.court!r}")
                if evidence.decided_on and e.decided_on and evidence.decided_on != e.decided_on.isoformat():
                    bad.append(f"date {evidence.decided_on} != registry {e.decided_on}")
                (fails if bad else oks).append(
                    f"evidence metadata disagrees with registry: {', '.join(bad)}" if bad
                    else "evidence metadata matches registry")
        elif evidence.case_name and not evidence.citation:
            warns.append("evidence has no citation on record, so its metadata cannot be confirmed")

        text = claim.text
        for ref in find_case_refs(text):
            entry = None
            if ref.citation:
                hits = self.index.find_citation(ref.citation)
                if not hits:
                    continue  # nonexistence is the authority check's finding, not this one's
                entry = hits[0]
                if ref.name and norm_name(entry.case_name) not in norm_name(ref.name) \
                        and norm_name(ref.name) not in norm_name(entry.case_name):
                    fails.append(f"citation {ref.citation!r} belongs to {entry.case_name!r}, not {ref.name!r}")
                    continue
            elif ref.name:
                hits = self.index.find_case(ref.name)
                if not hits:
                    continue
                entry = hits[0]
            if entry is None:
                continue
            year = stated_decision_year(text, ref.end)
            if entry.decided_on and year and year != entry.decided_on.year:
                fails.append(f"{ref.name or ref.citation!r}: claim gives year {year}, "
                             f"registry says {entry.decided_on.year}")
                continue
            courts = find_courts(text[max(0, ref.start - 60): ref.end + 60])
            if courts and entry.court and not any(c in entry.court.lower() for c in courts):
                fails.append(f"{ref.name or ref.citation!r}: claim says {courts[0]}, registry says {entry.court!r}")
                continue
            oks.append(f"{ref.name or ref.citation!r}: name/citation/year/court consistent with registry")

        return combine(self.name, evidence, fails, warns, oks,
                       na_reason="evidence is not a judgment and the claim names no case")
