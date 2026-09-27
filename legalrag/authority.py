"""In-memory index of what authorities exist: the citation registry plus every statute
section version. Built from Postgres for real runs (`AuthorityRepo.load`) and from the
corpus files for offline tests (`legalrag.ingest.corpus_view`). The verification checks and
the evaluation metrics depend on this interface, never on SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_WS = re.compile(r"\s+")
_NAME_NOISE = re.compile(r"\(\s*synthetic\s*\)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def norm_name(s: str) -> str:
    """Case name / act name normal form: lower case, no punctuation, no '(SYNTHETIC)' tag,
    'v.' == 'v', no leading 'the'."""
    s = _NAME_NOISE.sub(" ", s.lower())
    s = _NON_ALNUM.sub(" ", s)
    s = norm_ws(s)
    return s[4:] if s.startswith("the ") else s


def norm_citation(s: str) -> str:
    return norm_ws(s).upper()


@dataclass(frozen=True)
class RegistryEntry:
    case_name: str
    citation: str
    court: str | None
    decided_on: date | None
    in_corpus: bool


@dataclass(frozen=True)
class SectionVersion:
    act_name: str
    act_year: int
    section_number: str
    subsection: str
    text: str
    valid_from: date
    valid_until: date | None  # None = still in force
    amendment_note: str | None = None


class AuthorityIndex:
    def __init__(self, registry: list[RegistryEntry], sections: list[SectionVersion]):
        self.registry = list(registry)
        self.sections = list(sections)
        self._by_citation: dict[str, list[RegistryEntry]] = {}
        for e in self.registry:
            self._by_citation.setdefault(norm_citation(e.citation), []).append(e)
        self._acts: dict[str, tuple[str, int]] = {}  # norm act name -> (act_name, act_year)
        self._versions: dict[tuple[str, str], list[SectionVersion]] = {}
        for v in self.sections:
            self._acts.setdefault(norm_name(v.act_name), (v.act_name, v.act_year))
            self._versions.setdefault((norm_name(v.act_name), v.section_number.upper()), []).append(v)
        for vs in self._versions.values():
            vs.sort(key=lambda v: v.valid_from)

    # -- cases ---------------------------------------------------------------------
    def find_citation(self, citation: str) -> list[RegistryEntry]:
        return list(self._by_citation.get(norm_citation(citation), []))

    def find_case(self, name: str) -> list[RegistryEntry]:
        """Registry entries whose name equals `name`, or contains it / is contained by it
        (a claim usually gives a short name). Needs >= 2 tokens either way so that a bare
        surname never matches."""
        n = norm_name(name)
        if len(n.split()) < 2:
            return []
        hits = []
        for e in self.registry:
            m = norm_name(e.case_name)
            if m == n or (len(m.split()) >= 2 and (m in n or n in m)):
                hits.append(e)
        return hits

    # -- statutes ------------------------------------------------------------------
    def acts(self) -> list[tuple[str, int]]:
        return list(self._acts.values())

    def find_act(self, name: str) -> tuple[str, int] | None:
        n = norm_name(name)
        if n in self._acts:
            return self._acts[n]
        for key, val in self._acts.items():  # "the Sample Tenancy Act, 2001" vs "Sample Tenancy Act"
            if key in n or n in key:
                return val
        return None

    def versions(self, act_name: str, section_number: str) -> list[SectionVersion]:
        return list(self._versions.get((norm_name(act_name), section_number.upper()), []))

    def acts_with_section(self, section_number: str) -> list[str]:
        return sorted({v.act_name for (a, s), vs in self._versions.items()
                       if s == section_number.upper() for v in vs[:1]})

    def version_for_text(self, act_name: str, section_number: str, text: str) -> SectionVersion | None:
        """Identify which version a retrieved chunk is, by exact (whitespace-normalised) text."""
        want = norm_ws(text)
        for v in self.versions(act_name, section_number):
            if norm_ws(v.text) == want:
                return v
        return None
