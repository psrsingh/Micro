"""Pull authority references (cases, citations, sections, acts, years, courts) out of free
text. Regex-based and deliberately conservative: a reference we fail to extract is simply
not checked, whereas one we extract wrongly could wrongly reject a claim. Every pattern
here has a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legalrag.authority import AuthorityIndex

# 2004 SYN 21 / 2023 INSC 123, (2004) 3 SCC 101, AIR 1973 SC 1461
_CITATION = re.compile(
    r"\bAIR\s+\d{4}\s+[A-Z]{2,6}\s+\d+\b"
    r"|\(\d{4}\)\s+\d+\s+[A-Z]{2,6}\s+\d+\b"
    r"|\b\d{4}\s+[A-Z]{2,6}\s+\d+\b"
)
_CAP = r"[A-Z][\w.&'-]*"
_CONNECT = r"(?:of|the|&|for|in)"  # not "and": it would fuse "X v. Y and Z v. W"
_PARTY = rf"{_CAP}(?:\s+(?:{_CONNECT}\s+)?{_CAP})*"  # connectors only between capitalised words
_CASE = re.compile(rf"({_PARTY})\s+v(?:s)?\.?\s+({_PARTY})(?:\s+\(\s*SYNTHETIC\s*\))?")
# Function words that a sentence starts with and the party-name regex would swallow.
_LEAD = frozenset("in the see per as also held whether under following applying citing".split())
# "[A-Z]{0,2}": plain sections/articles ('103'), one amendment letter ('35A'), or the Constitution's
# two-letter amendment insertions ('239AA').
_SECTION = re.compile(r"\b(?:[Ss]ec(?:tion|\.)?|[Aa]rticle)\s*(\d+[A-Z]{0,2})(?:\((\w+)\))?")
# "Act", the 2023 codes' own names ("Bharatiya Nyaya Sanhita, 2023"), or "Constitution of India"
# (which, unlike an Act, is named by its FIRST word, not its last: no trailing "Act"/"Sanhita").
_ACT = re.compile(r"((?:[A-Z][\w()'-]*\s+){1,7}?(?:Act|Sanhita|Adhiniyam)|Constitution of India)(?:\s*,?\s*(\d{4}))?")
_COURT = re.compile(r"\b(Supreme Court|High Court)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CaseRef:
    name: str | None  # as written, '(SYNTHETIC)' tag kept off
    citation: str | None
    start: int
    end: int


@dataclass(frozen=True)
class ActMention:
    name: str  # as written, without a leading 'The'
    year: int | None
    start: int
    end: int


@dataclass(frozen=True)
class SectionRef:
    number: str
    subsection: str
    act: ActMention | None  # act named next to the section, if any
    start: int
    end: int


def find_citations(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _CITATION.finditer(text)]


def _strip_lead(name: str) -> str:
    words = name.split()
    while words and words[0].lower() in _LEAD:
        words.pop(0)
    return " ".join(words)


def find_case_refs(text: str) -> list[CaseRef]:
    """Case names ('X v. Y'), each paired with a citation written right after it
    (e.g. 'X v. Y, 2004 SYN 21' or 'X v. Y (2004 SYN 21)'). Bare citations with no case name
    next to them are returned with name=None."""
    cites = find_citations(text)
    used: set[int] = set()
    refs: list[CaseRef] = []
    for m in _CASE.finditer(text):
        name = _strip_lead(f"{m.group(1)} v. {m.group(2)}")
        if len(name.split()) < 3:
            continue
        cit, end = None, m.end()
        for i, (c, s, e) in enumerate(cites):
            if i not in used and 0 <= s - m.end() <= 6 and not text[m.end():s].strip(" ,(;"):
                cit, end, used = c, e, used | {i}  # the reference spans name and citation
                break
        refs.append(CaseRef(name, cit, m.start(), end))
    for i, (c, s, e) in enumerate(cites):
        if i not in used:
            refs.append(CaseRef(None, c, s, e))
    return sorted(refs, key=lambda r: r.start)


def find_act_mentions(text: str) -> list[ActMention]:
    out = []
    for m in _ACT.finditer(text):
        name = m.group(1)
        while name.split()[0].lower() == "the" and len(name.split()) > 2:
            name = " ".join(name.split()[1:])
        out.append(ActMention(name, int(m.group(2)) if m.group(2) else None, m.start(1), m.end()))
    return out


_AFTER = re.compile(r"\s*(?:of|under|in)\s+(?:the\s+)?", re.IGNORECASE)
_BEFORE_SEP = re.compile(r"[\s,;:\-–—]*$")


def find_section_refs(text: str) -> list[SectionRef]:
    """'Section 12' references, each paired with an Act named right after it ('of the X
    Act') or right before it ('X Act, 2001, Section 12'), else act=None."""
    acts = find_act_mentions(text)
    refs = []
    for m in _SECTION.finditer(text):
        act = None
        tail = _AFTER.match(text, m.end())
        if tail:
            act = next((a for a in acts if a.start == tail.end()), None)
        if act is None:
            for a in acts:
                if a.end <= m.start() and _BEFORE_SEP.fullmatch(text[a.end:m.start()]) and m.start() - a.end <= 8:
                    act = a
        refs.append(SectionRef(m.group(1), m.group(2) or "", act, m.start(), m.end()))
    return refs


_YEAR_PAREN = re.compile(r"\s*\(\s*(\d{4})\s*\)")
_YEAR_DECIDED = re.compile(r"\b(?:decided|delivered|pronounced|dated)\b[^.;]{0,30}?\b(\d{4})\b")


def stated_decision_year(text: str, end: int) -> int | None:
    """The decision year a claim states for the case ending at `end`: '(2004)' right after
    it, or 'decided in 2004' shortly after. A year that merely appears nearby (e.g. an Act's
    year) is not a statement about the case and is ignored."""
    tail = text[end:end + 70]
    m = _YEAR_PAREN.match(tail) or _YEAR_DECIDED.search(tail)
    return int(m.group(1)) if m else None


def find_courts(text: str) -> list[str]:
    return [m.group(1).lower() for m in _COURT.finditer(text)]


def known_acts_in(text: str, index: AuthorityIndex) -> list[tuple[str, int]]:
    """Acts from the index whose name appears in `text` (ignoring case/punctuation)."""
    from legalrag.authority import norm_name

    hay = norm_name(text)
    return [(n, y) for n, y in index.acts() if norm_name(n) in hay]
