"""Coverage guard: does the question itself name an authority the corpus cannot ground?

If someone asks "what did X v. Y hold?" and we hold no text for X v. Y, every passage we
retrieve is about some other authority, and an answer built from them would read as if it
were about X v. Y. Verification cannot catch that (each claim is faithfully supported), so the
pipeline abstains instead. The same goes for a section of an Act the corpus does not contain.

Deliberately conservative in the other direction too: an authority mentioned only as context
also triggers it. Losing an answer is preferable to a confident answer about something we
could not check.
"""

from __future__ import annotations

from legalrag.authority import AuthorityIndex
from legalrag.verify.extract import find_case_refs, find_section_refs


def unavailable_authorities(index: AuthorityIndex, query: str, extra_case_names: list[str] | None = None) -> list[str]:
    """Human-readable reasons, one per named authority we cannot ground. Empty = covered."""
    out: list[str] = []
    seen: set[str] = set()

    def add(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            out.append(msg)

    refs = find_case_refs(query)
    for name in extra_case_names or []:
        refs += [r for r in find_case_refs(name) if r.name]
    for r in refs:
        entries = (index.find_citation(r.citation) if r.citation else []) or (index.find_case(r.name) if r.name else [])
        label = r.name or r.citation or "?"
        if r.citation and r.name:
            label = f"{r.name} ({r.citation})"
        if not entries:
            add(f"{label}: not in the citation registry")
        elif not any(e.in_corpus for e in entries):
            add(f"{label}: exists, but its text is not in the corpus")

    for s in find_section_refs(query):
        if s.act:
            found = index.find_act(s.act.name)
            if not found:
                add(f"Section {s.number} of {s.act.name}: the Act is not in the corpus")
            elif not index.versions(found[0], s.number):
                add(f"Section {s.number} of {found[0]}: no such section in the corpus")
        elif not index.acts_with_section(s.number):
            add(f"Section {s.number}: no Act in the corpus has such a section")
    return out
