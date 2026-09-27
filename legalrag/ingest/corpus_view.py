"""A database-free view of a corpus directory: the same `AuthorityIndex` the pipeline builds
from Postgres, plus builders for `Evidence` and passage lookups. Used by the offline tests,
by benchmark validation (gold passages must exist verbatim) and by the authority probes.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from legalrag.authority import AuthorityIndex, RegistryEntry, SectionVersion, norm_citation, norm_name
from legalrag.evidence import Evidence
from legalrag.ingest.parse import ParsedDocument, parse_document


class CorpusView:
    def __init__(self, corpus_dir: Path):
        self.dir = Path(corpus_dir)
        self.docs: list[ParsedDocument] = [parse_document(p) for p in sorted(self.dir.rglob("*.txt"))]
        self._next_id = -1
        registry: list[RegistryEntry] = []
        held = {norm_citation(d.neutral_citation) for d in self.docs if d.neutral_citation}
        for d in self.docs:
            if d.doc_type == "judgment" and d.neutral_citation:
                registry.append(RegistryEntry(d.title, d.neutral_citation, d.court, d.decided_on, True))
        for listing in sorted(self.dir.rglob("registry*.csv")):
            with listing.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    cit = row["citation"].strip()
                    if norm_citation(cit) in held:
                        continue
                    registry.append(RegistryEntry(
                        row["case_name"].strip(), cit, row.get("court") or None,
                        date.fromisoformat(row["decided_on"]) if row.get("decided_on") else None, False))
        sections: list[SectionVersion] = []
        for d in self.docs:
            if d.doc_type != "statute":
                continue
            for s in d.sections:
                v = s.statute
                assert v is not None and d.act_name and d.act_year is not None
                sections.append(SectionVersion(d.act_name, d.act_year, v.section_number, v.subsection,
                                               d.raw_text[s.start:s.end].strip(), v.valid_from,
                                               v.valid_until, v.amendment_note))
        self.index = AuthorityIndex(registry, sections)

    # -- lookups -------------------------------------------------------------------
    def judgment(self, citation_or_name: str) -> ParsedDocument | None:
        key = norm_citation(citation_or_name)
        for d in self.docs:
            if d.doc_type != "judgment":
                continue
            if (d.neutral_citation and norm_citation(d.neutral_citation) == key) \
                    or norm_name(d.title) == norm_name(citation_or_name):
                return d
        return None

    def sections_of(self, doc: ParsedDocument, label: str | None = None):
        return [s for s in doc.sections if label is None or s.label == label]

    def passages_for_case(self, citation_or_name: str) -> list[str]:
        d = self.judgment(citation_or_name)
        return [d.raw_text[s.start:s.end] for s in d.sections] if d else []

    def passages_for_section(self, act_name: str, section: str) -> list[str]:
        return [v.text for v in self.index.versions(act_name, section)]

    # -- evidence builders -----------------------------------------------------------
    def _id(self) -> int:
        self._next_id -= 1
        return self._next_id

    def judgment_evidence(self, citation_or_name: str, label: str = "holding") -> Evidence:
        d = self.judgment(citation_or_name)
        if d is None:
            raise KeyError(f"no judgment {citation_or_name!r} in {self.dir}")
        s = self.sections_of(d, label)[0]
        return Evidence(chunk_id=self._id(), document_id=0, text=d.raw_text[s.start:s.end],
                        span=(s.start, s.end), case_name=d.title, citation=d.neutral_citation,
                        court=d.court, decided_on=d.decided_on.isoformat() if d.decided_on else None,
                        section_label=label)

    def statute_evidence(self, act_name: str, section: str, on: date | None = None) -> Evidence:
        """The version of the section in force on `on` (default: the latest)."""
        versions = self.index.versions(act_name, section)
        if not versions:
            raise KeyError(f"no Section {section} in {act_name!r}")
        v = next((x for x in versions if on and x.valid_from <= on and (x.valid_until is None or x.valid_until >= on)),
                 versions[-1])
        return Evidence(chunk_id=self._id(), document_id=0, text=v.text, span=(0, len(v.text)),
                        act_name=v.act_name, section_number=v.section_number,
                        section_label="statute_section")
