"""Parse source files into structural sections with exact character offsets.

Phase 0 reads the fixture format (YAML front matter + `## SECTION` headings). Phase 1
adds real judgment/statute parsers behind the same `ParsedDocument` shape. Offsets always
index into the raw file text as read by `read_raw`, so the UI can show a passage in place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_HEADING = re.compile(r"^## (.+?)[ \t]*\r?$", re.MULTILINE)
_STATUTE_HEAD = re.compile(
    r"^SECTION\s+(?P<num>\d+[A-Z]?)(?:\((?P<sub>[^)]+)\))?\s*"
    r"\[valid (?P<from>\d{4}-\d{2}-\d{2}) to (?P<to>present|\d{4}-\d{2}-\d{2})\]"
    r"(?:\s*\((?P<note>.*)\))?$"  # greedy: notes may contain nested parentheses
)

JUDGMENT_LABELS = {"HEADNOTE": "headnote", "FACTS": "facts", "ARGUMENTS": "arguments",
                   "REASONING": "reasoning", "HOLDING": "holding"}


class ParseError(ValueError):
    pass


@dataclass
class Section:
    label: str
    start: int  # char offsets of the body (heading excluded) in the raw text
    end: int
    statute: StatuteVersion | None = None


@dataclass
class StatuteVersion:
    section_number: str
    subsection: str
    valid_from: date
    valid_until: date | None
    amendment_note: str | None


@dataclass
class ParsedDocument:
    doc_type: str
    title: str
    court: str | None
    decided_on: date | None
    neutral_citation: str | None
    reporter_citations: list[str]
    source_url: str | None
    act_name: str | None
    act_year: int | None
    raw_text: str
    sha256: str
    sections: list[Section] = field(default_factory=list)


def read_raw(path: Path) -> tuple[str, str]:
    """Return (text, sha256). Bytes are decoded without newline translation so that
    offsets stay valid against the file on disk."""
    data = path.read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def _date(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def parse_document(path: Path) -> ParsedDocument:
    text, sha = read_raw(path)
    m = _FRONT.match(text)
    if not m:
        raise ParseError(f"{path.name}: missing YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    body_start = m.end()

    doc_type = meta.get("doc_type")
    if doc_type not in ("judgment", "statute"):
        raise ParseError(f"{path.name}: doc_type must be judgment|statute, got {doc_type!r}")
    if not meta.get("title"):
        raise ParseError(f"{path.name}: title is required")

    doc = ParsedDocument(
        doc_type=doc_type, title=meta["title"], court=meta.get("court"),
        decided_on=_date(meta.get("decided_on")), neutral_citation=meta.get("neutral_citation"),
        reporter_citations=list(meta.get("reporter_citations") or []),
        source_url=meta.get("source_url"), act_name=meta.get("act_name"),
        act_year=meta.get("act_year"), raw_text=text, sha256=sha,
    )

    heads = [h for h in _HEADING.finditer(text) if h.start() >= body_start]
    if not heads:
        raise ParseError(f"{path.name}: no '## ' section headings found")
    for i, h in enumerate(heads):
        start = h.end() + 1 if h.end() < len(text) else h.end()  # skip the newline after heading
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        # trim trailing whitespace so chunk spans end on real characters
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            raise ParseError(f"{path.name}: empty section {h.group(1)!r}")
        title = h.group(1).strip()
        if doc_type == "judgment":
            label = JUDGMENT_LABELS.get(title.upper())
            if label is None:
                raise ParseError(f"{path.name}: unknown judgment section {title!r}")
            doc.sections.append(Section(label, start, end))
        else:
            sm = _STATUTE_HEAD.match(title)
            if not sm:
                raise ParseError(f"{path.name}: cannot parse statute heading {title!r}")
            until = None if sm["to"] == "present" else date.fromisoformat(sm["to"])
            ver = StatuteVersion(sm["num"], sm["sub"] or "", date.fromisoformat(sm["from"]),
                                 until, (sm["note"] or None))
            doc.sections.append(Section("statute_section", start, end, ver))

    if doc_type == "statute":
        if not doc.act_name or doc.act_year is None:
            raise ParseError(f"{path.name}: statute needs act_name and act_year")
        _check_version_chain(path, doc)
    return doc


def _check_version_chain(path: Path, doc: ParsedDocument) -> None:
    """Versions of one section must not overlap and must tile without gaps."""
    by_key: dict[tuple[str, str], list[StatuteVersion]] = {}
    for s in doc.sections:
        v = s.statute
        assert v is not None
        by_key.setdefault((v.section_number, v.subsection), []).append(v)
    for key, versions in by_key.items():
        versions.sort(key=lambda v: v.valid_from)
        for a, b in zip(versions, versions[1:]):
            if a.valid_until is None:
                raise ParseError(f"{path.name}: section {key[0]} has a version after an open-ended one")
            if a.valid_until >= b.valid_from:
                raise ParseError(f"{path.name}: section {key[0]} versions overlap")
