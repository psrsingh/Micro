"""Legal structure on top of extracted pages: Acts -> chapter / section / sub-section, and reported
judgments -> editorial front matter + numbered paragraphs.

Principles (each one exists because the alternative silently corrupts evidence):
  * A section or paragraph number is accepted only if it is the *next number in sequence*. A
    numbered list inside a section, or a quotation with its own numbering, is therefore not
    mistaken for a new provision, and a genuinely missing number is reported instead of skipped.
  * Every chunk is an exact slice of `ExtractedDoc.text` (`text[char_start:char_end] == chunk`),
    with the PDF pages it spans, so a retrieved passage can be located in the original.
  * Unrelated provisions are never merged: one section (or sub-section, when the section is long)
    per chunk. Long units are split by words; the pieces keep the same section metadata.
  * Editorial matter (headnotes, keywords) is labelled `headnote`, never mixed into judgment text.
  * Paragraph roles (arguments / facts / reasoning / holding) are keyword heuristics. They are
    labelled as such (`role_source`), and anything that matches nothing stays `judgment_text`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalrag.corpus.extract import ExtractedDoc, LineRef
from legalrag.ingest.chunk import chunk_span


@dataclass
class StructuredChunk:
    text: str
    char_start: int
    char_end: int
    section_label: str
    page_start: int
    page_end: int
    act_name: str | None = None
    chapter: str | None = None
    section_number: str | None = None
    subsection: str | None = None
    heading: str | None = None
    paragraph_number: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StatuteUnit:
    """One row of `statute_sections`: a whole section, or one sub-section of a long section."""
    section_number: str
    subsection: str
    chapter: str | None
    heading: str | None
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    chunks: list[StructuredChunk] = field(default_factory=list)


@dataclass
class ParsedStatute:
    units: list[StatuteUnit]
    n_sections: int
    problems: list[str] = field(default_factory=list)   # make the document unfit to ingest
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedJudgment:
    chunks: list[StructuredChunk]
    reporter_citation: str | None
    neutral_citation: str | None
    n_paragraphs: int
    list_of_acts: str | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _words(text: str) -> int:
    return len(text.split())


def _split_words(doc: ExtractedDoc, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    """Span(s) inside text[start:end] of at most ~max_words words (one span when it fits)."""
    spans = chunk_span(doc.text, start, end, target=max_words, max_tokens=max_words,
                       overlap=min(30, max_words // 4))
    return [(s.start, s.end) for s in spans]


# ======================================================================================= statutes

_CHAPTER = re.compile(r"^CHAPTER\s*([IVXLC]+[A-Z]?)\s*$")  # justified headings can lose the space: "CHAPTERI"
_PART = re.compile(r"^PART\s*([IVXLC]+)\s*$")
_SCHEDULE = re.compile(r"^(?:THE\s*)?(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s*)?SCHEDULE\s*$")
# A number, a full stop, then a letter/bracket/quote - with or without a space ("192.Whoever"). A digit after
# the stop ("1.1", "2.5 lakh") is not a section start.
_LEAD_CH = r"(?=[A-Za-z(“‘\"'�])"
_SEC_START = re.compile(r"^(\d{1,3})\.\s*" + _LEAD_CH)
_SUB_START = re.compile(r"^(?:\d{1,3}\.\s+)?\((\d{1,3})\)\s")
_XREF = re.compile(r"\b\d{1,3}\s+of\s+\d{4}\.?")  # marginal cross-references to other Acts, e.g. "2 of 1974."
_CAPS_TITLE = re.compile(r"^[A-Z][A-Z ,'()&\-–—/]+$")


def _is_centered_heading(ref: LineRef, body_left: float) -> bool:
    t = ref.text.strip()
    return (ref.line.x0 >= body_left + 60 and len(t) <= 80 and not t.endswith((".", ";", ":", ",", "—", "-"))
            and not _SEC_START.match(t))


def parse_statute(doc: ExtractedDoc, act_name: str, *, max_words: int) -> ParsedStatute:
    refs = doc.line_refs()
    out = ParsedStatute(units=[], n_sections=0)
    if not refs:
        out.problems.append("no text extracted")
        return out
    # the left edge of body text: the most common line start (section/indent lines are to its right)
    xs = sorted(round(r.line.x0) for r in refs)
    body_left = float(xs[len(xs) // 4])

    # -- pass 1: chapter headings and section starts, in order -----------------------------
    first = next((i for i, r in enumerate(refs) if _CHAPTER.match(r.text.strip())), None)
    if first is None:
        out.problems.append("no 'CHAPTER I' heading found: cannot locate the start of the Act")
        return out

    starts: list[dict] = []           # section starts
    chapters: list[tuple[int, str]] = []  # (line index, label) in order
    excluded: set[int] = set()        # heading lines that belong to no section
    expected, in_sections, end_idx = 1, False, len(refs)
    i = first
    while i < len(refs):
        t = refs[i].text.strip()
        cm, pm = _CHAPTER.match(t), _PART.match(t)
        if cm or pm:
            label = f"CHAPTER {cm.group(1)}" if cm else f"PART {pm.group(1)}"
            excluded.add(i)
            j = i + 1
            title: list[str] = []
            while j < len(refs) and _CAPS_TITLE.match(refs[j].text.strip()) and len(title) < 3 \
                    and not _CHAPTER.match(refs[j].text.strip()) and not _PART.match(refs[j].text.strip()) \
                    and not _SCHEDULE.match(refs[j].text.strip()):
                title.append(refs[j].text.strip())
                excluded.add(j)
                j += 1
            chapters.append((i, label + (" - " + " ".join(title) if title else "")))
            i = j
            continue
        if in_sections and _SCHEDULE.match(t):
            end_idx = i
            out.warnings.append(f"stopped at line {t!r}: Schedules are not indexed")
            break
        m = _SEC_START.match(t)
        if m and int(m.group(1)) == expected:
            starts.append({"num": expected, "idx": i})
            expected += 1
            in_sections = True
        i += 1
    if not starts:
        out.problems.append("no numbered sections found")
        return out
    out.n_sections = starts[-1]["num"]
    # A lost sequence must never look like a short Act: any line further on that looks like one of the
    # next few section numbers means we stopped recognising sections part-way.
    for r in refs[starts[-1]["idx"] + 1:end_idx]:
        m = re.match(r"^(\d{1,3})\.", r.text.strip())
        if m and expected <= int(m.group(1)) <= expected + 5:
            out.problems.append(f"section numbering stalled after {expected - 1}: line {r.text[:60]!r} (page {r.page}) "
                                f"looks like section {m.group(1)} but was not recognised")
            break

    # -- pass 2: per-section extents, headings, sub-sections, chunks ------------------------
    def chapter_at(idx: int) -> str | None:
        cur = None
        for ci, label in chapters:
            if ci <= idx:
                cur = label
        return cur

    for k, s in enumerate(starts):
        a = s["idx"]
        b = starts[k + 1]["idx"] if k + 1 < len(starts) else end_idx
        # exclude chapter headings and centred sub-headings that sit just before the next section
        while b - 1 > a and (b - 1 in excluded or _is_centered_heading(refs[b - 1], body_left)) and k + 1 < len(starts):
            b -= 1
        while b - 1 > a and (b - 1 in excluded):
            b -= 1
        lines = [(j, refs[j]) for j in range(a, b) if j not in excluded]
        if not lines:
            out.problems.append(f"section {s['num']} has no text")
            continue
        s.update(lines=lines)
        # marginal note = the section title, taken from the margin column beside the section start
        page = refs[a].page
        top0 = refs[a].line.top
        nxt_top = None
        if k + 1 < len(starts) and refs[starts[k + 1]["idx"]].page == page:
            nxt_top = refs[starts[k + 1]["idx"]].line.top
        pg = next((p for p in doc.pages if p.number == page), None)
        notes: list[str] = []
        if pg:
            # the note is a contiguous block of small-gap lines starting beside the section's first line;
            # anything after a larger gap is page furniture or another section's note
            last_top = None
            for mn in sorted(pg.margin, key=lambda m: m.top):
                if mn.top < top0 - 4 or (nxt_top is not None and mn.top >= nxt_top - 4):
                    continue
                if last_top is not None and mn.top - last_top > 16:
                    break
                notes.append(mn.text)
                last_top = mn.top
        heading = _XREF.sub(" ", " ".join(notes))
        heading = " ".join(heading.split()) or None
        if heading is None:
            out.warnings.append(f"section {s['num']}: no marginal note (title) found")
        s["heading"] = heading

        char_start, char_end = lines[0][1].start, lines[-1][1].end
        text = doc.text[char_start:char_end]
        chapter = chapter_at(a)

        out.units.extend(_build_units(doc, act_name, chapter, heading, str(s["num"]), text, lines, max_words))
    for u in out.units:
        if not u.text.strip():
            out.problems.append(f"section {u.section_number} is empty")
    return out


def _build_units(doc: ExtractedDoc, act_name: str | None, chapter: str | None, heading: str | None,
                 number: str, text: str, lines: list[tuple[int, LineRef]], max_words: int) -> list[StatuteUnit]:
    """One section/article -> one `StatuteUnit`, split at its own `(1)`, `(2)`, ... sub-sections
    (and those into overlapping word-chunks) only when it is too long for a single chunk. Shared
    by `parse_statute` and `parse_constitution`: a sub-section is a sub-section whether the parent
    provision is called a Section or an Article."""
    subs: list[tuple[str, int]] = []  # (label, line-list index)
    if _words(text) > max_words:
        want = 1
        for li, (_, r) in enumerate(lines):
            m = _SUB_START.match(r.text.strip())
            if m and int(m.group(1)) == want:
                subs.append((str(want), li))
                want += 1
    unit_specs: list[tuple[str, int, int]] = []  # (subsection, first line index, last line index+1)
    if len(subs) >= 2:
        for n, (label, li) in enumerate(subs):
            first_li = 0 if n == 0 else li  # text before "(1)" (the lead-in) belongs to sub-section 1
            last_li = subs[n + 1][1] if n + 1 < len(subs) else len(lines)
            unit_specs.append((label, first_li, last_li))
    else:
        unit_specs.append(("", 0, len(lines)))

    units = []
    for label, fl, ll in unit_specs:
        seg = lines[fl:ll]
        us, ue = seg[0][1].start, seg[-1][1].end
        unit = StatuteUnit(number, label, chapter, heading, doc.text[us:ue], us, ue,
                           seg[0][1].page, seg[-1][1].page)
        for cs, ce in _split_words(doc, us, ue, max_words):
            cp = doc.page_of(cs), doc.page_of(max(cs, ce - 1))
            unit.chunks.append(StructuredChunk(
                text=doc.text[cs:ce], char_start=cs, char_end=ce, section_label="statute_section",
                page_start=cp[0], page_end=cp[1], act_name=act_name, chapter=chapter,
                section_number=number, subsection=label or None, heading=heading))
        units.append(unit)
    return units


# ================================================================================= constitutions
# Constitution-style PDFs: PART -> (nested CHAPTER, sometimes) -> Article -> clause. Different from
# a Gazette-style Act in three ways that matter here:
#   * the document starts with "PART I", not "CHAPTER I", and Parts (not Chapters) are the top-level
#     division - `_PART`/`_CHAPTER` are reused, but either can start the document;
#   * a long front-matter table of contents repeats every Part/Article heading with dot-leaders
#     ("1. Name and territory .......... 2") before the real body: skipped by `_toc_like` (a
#     dot-leader within the next couple of lines - the leader can itself wrap onto a continuation
#     line), never by page position, since ToC length varies by edition;
#   * amendments insert articles with a letter suffix (239AA, 371A) that sit between an existing
#     number and the next one rather than extending the sequence - accepted as a "bonus" insertion
#     (nothing subsequent needs to renumber) rather than requiring strict integer succession.
# There is no marginal-note column for the heading (unlike the Gazette Acts): the title sits inline,
# between the article number and the dash that starts the article's own text.

_ART_START = re.compile(r"^(?:\d{1,3}\[)?(\d{1,3}[A-Z]{0,2})\.\s+(?=[A-Za-z\[])")
# Rare (a handful of times across 800 pages): justified text stranded the number on its own line,
# title wrapped onto the lines around it. The article is still recognised (so numbering doesn't
# stall); its heading is left unextracted rather than guessed from a title that may span both
# neighbouring lines.
_BARE_ART_NUM = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.$")
_DASH_SPLIT = re.compile(r"\.\s*[—–\-�]\s*")  # title '.', then an em/en-dash (mojibake or plain '-')


def _toc_like(refs: list[LineRef], idx: int, span: int = 3) -> bool:
    return any("..." in refs[k].text for k in range(idx, min(idx + span, len(refs))))


def parse_constitution(doc: ExtractedDoc, act_name: str, *, max_words: int) -> ParsedStatute:
    refs = doc.line_refs()
    out = ParsedStatute(units=[], n_sections=0)
    if not refs:
        out.problems.append("no text extracted")
        return out

    # -- pass 1: skip the table of contents, then walk Part/Chapter headings and article starts --
    first = None
    for i, r in enumerate(refs):
        if _CHAPTER.match(r.text.strip()) or _PART.match(r.text.strip()):
            if any(_ART_START.match(refs[j].text.strip()) and not _toc_like(refs, j)
                  for j in range(i + 1, min(i + 8, len(refs)))):
                first = i
                break
    if first is None:
        out.problems.append("no 'PART I' (or 'CHAPTER I') heading found outside the table of "
                            "contents: cannot locate the start of the Constitution")
        return out

    starts: list[dict] = []
    chapters: list[tuple[int, str]] = []
    excluded: set[int] = set()
    expected, in_articles, end_idx = 1, False, len(refs)
    i = first
    while i < len(refs):
        t = refs[i].text.strip()
        cm, pm = _CHAPTER.match(t), _PART.match(t)
        if cm or pm:
            label = f"CHAPTER {cm.group(1)}" if cm else f"PART {pm.group(1)}"
            excluded.add(i)
            j = i + 1
            title: list[str] = []
            while j < len(refs) and _CAPS_TITLE.match(refs[j].text.strip()) and len(title) < 3 \
                    and not _CHAPTER.match(refs[j].text.strip()) and not _PART.match(refs[j].text.strip()) \
                    and not _SCHEDULE.match(refs[j].text.strip()):
                title.append(refs[j].text.strip())
                excluded.add(j)
                j += 1
            chapters.append((i, label + (" - " + " ".join(title) if title else "")))
            i = j
            continue
        if in_articles and _SCHEDULE.match(t):
            end_idx = i
            out.warnings.append(f"stopped at line {t!r}: Schedules are not indexed")
            break
        m = _ART_START.match(t)
        bare = None if m else _BARE_ART_NUM.match(t)
        if (m or bare) and not _toc_like(refs, i):
            base = (m or bare).group(1)
            numpart = re.match(r"\d+", base).group()
            num, letters = int(numpart), base[len(numpart):]
            if num == expected:
                starts.append({"num": base, "idx": i, "bare": bare is not None})
                expected += 1
                in_articles = True
            elif letters and num == expected - 1:
                starts.append({"num": base, "idx": i, "bare": bare is not None})
        i += 1
    if not starts:
        out.problems.append("no numbered articles found")
        return out
    out.n_sections = len(starts)
    tail0 = starts[-1]["idx"] + 1
    for offset, r in enumerate(refs[tail0:end_idx]):
        idx, t = tail0 + offset, r.text.strip()
        m = re.match(r"^(\d{1,3})\.", t)
        if m and expected <= int(m.group(1)) <= expected + 5 and not _toc_like(refs, idx):
            out.problems.append(f"article numbering stalled after {starts[-1]['num']}: line {t[:60]!r} "
                                f"(page {r.page}) looks like article {m.group(1)} but was not recognised")
            break

    # -- pass 2: per-article extent, chapter, inline heading, sub-sections/chunks -----------------
    def chapter_at(idx: int) -> str | None:
        cur = None
        for ci, label in chapters:
            if ci <= idx:
                cur = label
        return cur

    for k, s in enumerate(starts):
        a = s["idx"]
        b = starts[k + 1]["idx"] if k + 1 < len(starts) else end_idx
        while b - 1 > a and b - 1 in excluded:
            b -= 1
        lines = [(j, refs[j]) for j in range(a, b) if j not in excluded]
        if not lines:
            out.problems.append(f"article {s['num']} has no text")
            continue
        char_start, char_end = lines[0][1].start, lines[-1][1].end
        text = doc.text[char_start:char_end]
        chapter = chapter_at(a)

        heading = None
        if not s["bare"]:
            m = _ART_START.match(refs[a].text.strip())
            dm = _DASH_SPLIT.search(refs[a].text.strip()[m.end():])
            if dm:
                heading = refs[a].text.strip()[m.end():][:dm.start()].strip()
        if heading is None:
            out.warnings.append(f"article {s['num']}: no inline heading found before the first dash")

        out.units.extend(_build_units(doc, act_name, chapter, heading, s["num"], text, lines, max_words))
    for u in out.units:
        if not u.text.strip():
            out.problems.append(f"article {u.section_number} is empty")
    return out


# ================================================================================ India Code Acts
# Consolidated Acts as published by India Code (e.g. the Indian Contract Act, 1872). Differs from
# the Gazette Acts in the same ways as the Constitution (inline title before a dash, lettered
# insertions such as 19A, footnote-marker prefixes such as "1[16."), and also:
#   * the front-matter "Arrangement of Sections" repeats every CHAPTER heading and numbered title,
#     without dot-leaders: the body is found as the first "1. Title.—text" line, since ToC entries
#     never carry the title's dash;
#   * the number and title may touch ("3.Communication", "76.[‘Goods’ defined.]");
#   * repealed provisions stay as one-line stubs: "76. [Title.] Rep. by ...", and whole repealed
#     chapters as "[CHAPTER VII.—SALES OF GOODS.] Rep. by ..." - indexed, so the corpus can say a
#     section was repealed rather than not knowing it;
#   * amendment footnotes (which look like section starts) are removed at extraction
#     (`strip_footnotes`), not here.

# up to four digits: a footnote marker can touch the number ("1151." = marker 1 + s.151)
_IC_SEC_START = re.compile(r"^(?:\d{1,3}\[)?(\d{1,4}[A-Z]{0,2})\.\s*(?=[A-Za-z\[“‘\"'])")
_IC_REPEALED_CHAPTER = re.compile(r"^\[?CHAPTER\s*([IVXLC]+)\s*\.?\s*[—–\-]\s*\[?([^\]]+?)\.?\]")
_IC_SCHEDULE_STUB = re.compile(r"^(?:THE\s*)?(?:[A-Z]+\s+)?SCHEDULE\s*[.—–\-]")
_IC_BRACKET_TITLE = re.compile(r"^\[([^\]]+?)\.?\]")
# a repealed group heading inside a repealed chapter: "[DELIVERY.]Rep. by s. 65, ibid."
_IC_REPEALED_SUBHEAD = re.compile(r"^\[[A-Z][A-Z’'\s,-]+\.?\]\s*Rep\.")


def parse_india_code(doc: ExtractedDoc, act_name: str, *, max_words: int) -> ParsedStatute:
    refs = doc.line_refs()
    out = ParsedStatute(units=[], n_sections=0)
    if not refs:
        out.problems.append("no text extracted")
        return out

    first = None
    for i, r in enumerate(refs):
        t = r.text.strip()
        m = _IC_SEC_START.match(t)
        if m and m.group(1) == "1" and _DASH_SPLIT.search(t[m.end():]):
            first = i
            break
    if first is None:
        out.problems.append("no '1. <title>.—' line found: cannot locate the start of the Act body")
        return out

    starts: list[dict] = []
    chapters: list[tuple[int, str]] = []
    excluded: set[int] = set()
    expected, end_idx = 1, len(refs)
    i = first
    while i < len(refs):
        t = refs[i].text.strip()
        cm, pm, rc = _CHAPTER.match(t), _PART.match(t), _IC_REPEALED_CHAPTER.match(t)
        if cm or pm:
            label = f"CHAPTER {cm.group(1)}" if cm else f"PART {pm.group(1)}"
            excluded.add(i)
            j = i + 1
            title: list[str] = []
            while j < len(refs) and _CAPS_TITLE.match(refs[j].text.strip()) and len(title) < 3 \
                    and not _CHAPTER.match(refs[j].text.strip()) and not _PART.match(refs[j].text.strip()):
                title.append(refs[j].text.strip())
                excluded.add(j)
                j += 1
            chapters.append((i, label + (" - " + " ".join(title) if title else "")))
            i = j
            continue
        if rc:
            chapters.append((i, f"CHAPTER {rc.group(1)} - {rc.group(2).strip()} (repealed)"))
            excluded.add(i)
            j = i + 1  # its "Rep. by ..." note can wrap onto following lines
            while j < len(refs) and not _IC_SEC_START.match(refs[j].text.strip())                     and not _CHAPTER.match(refs[j].text.strip()):
                excluded.add(j)
                j += 1
            i = j
            continue
        if _IC_REPEALED_SUBHEAD.match(t):
            excluded.add(i)
            i += 1
            continue
        if starts and (_SCHEDULE.match(t) or _IC_SCHEDULE_STUB.match(t)):
            end_idx = i
            out.warnings.append(f"stopped at line {t[:60]!r}: Schedules are not indexed")
            break
        m = _IC_SEC_START.match(t)
        if m:
            base = m.group(1)
            numpart = re.match(r"\d+", base).group()
            num, letters = int(numpart), base[len(numpart):]
            # a footnote marker can touch the number with no bracket: "1151." is footnote 1 on s.151
            glued = numpart != str(expected) and numpart.endswith(str(expected)) \
                and len(numpart) == len(str(expected)) + 1 and not letters
            if (num == expected or glued) and not letters:
                if glued:
                    out.warnings.append(f"section {expected}: read {numpart!r} as footnote marker + {expected}")
                    base = str(expected)
                starts.append({"num": base, "idx": i})
                expected += 1
            elif letters and num == expected - 1:
                starts.append({"num": base, "idx": i})
        i += 1
    if not starts:
        out.problems.append("no numbered sections found")
        return out
    out.n_sections = len(starts)
    tail0 = starts[-1]["idx"] + 1
    for r in refs[tail0:end_idx]:
        m = re.match(r"^(?:\d{1,3}\[)?(\d{1,3})\.", r.text.strip())
        if m and expected <= int(m.group(1)) <= expected + 5:
            out.problems.append(f"section numbering stalled after {starts[-1]['num']}: line {r.text[:60]!r} "
                                f"(page {r.page}) looks like section {m.group(1)} but was not recognised")
            break

    def chapter_at(idx: int) -> str | None:
        cur = None
        for ci, label in chapters:
            if ci <= idx:
                cur = label
        return cur

    xs = sorted(round(r.line.x0) for r in refs[first:end_idx])
    body_left = float(xs[len(xs) // 4])
    for k, s in enumerate(starts):
        a = s["idx"]
        b = starts[k + 1]["idx"] if k + 1 < len(starts) else end_idx
        # headings, and centred sub-headings ("Void agreements") just before the next section,
        # belong to neither section
        while b - 1 > a and (b - 1 in excluded or (k + 1 < len(starts) and _is_centered_heading(refs[b - 1], body_left))):
            b -= 1
        lines = [(j, refs[j]) for j in range(a, b) if j not in excluded]
        if not lines:
            out.problems.append(f"section {s['num']} has no text")
            continue
        text = doc.text[lines[0][1].start:lines[-1][1].end]

        # a long title wraps: look for its closing dash in the first three lines
        head_text = " ".join(r.text.strip() for _, r in lines[:3])
        rest = head_text[_IC_SEC_START.match(head_text).end():]
        heading = None
        bm = _IC_BRACKET_TITLE.match(rest)
        if bm:
            heading = bm.group(1).strip()
        else:
            dm = _DASH_SPLIT.search(rest)
            if dm:
                heading = rest[:dm.start()].strip()
        if heading is None:
            out.warnings.append(f"section {s['num']}: no inline heading found before the first dash")

        out.units.extend(_build_units(doc, act_name, chapter_at(a), heading, s["num"], text, lines, max_words))
    for u in out.units:
        if not u.text.strip():
            out.problems.append(f"section {u.section_number} is empty")
    return out


# ====================================================================================== judgments

_SCR_HEADER = re.compile(r"\[(\d{4})\]\s+(\d+)\s+S\.?C\.?R\.?\s+(\d+)\s*:\s*(\d{4}\s+INSC\s+\d+)")
_BODY_MARKER = re.compile(r"^\s*J\s*U\s*D\s*G\s*M\s*E\s*N\s*T\s*$", re.IGNORECASE)
_PARA = re.compile(r"^\s*(\d{1,3})\.\s*" + _LEAD_CH)
_BENCH = re.compile(r"\bJJ?\.\]?\s*$")
_FRONT_HEADINGS = ("issue for consideration", "headnotes", "case law cited", "case law reference",
                   "list of acts", "list of keywords", "case arising from", "appearances for parties",
                   "judgment / order of the supreme court")
_DROP_BLOCKS = ("appearances for parties", "case arising from")
_TAIL = re.compile(r"^\s*(Result of the case|Headnotes prepared by)\b|\(Assisted by\b", re.IGNORECASE)

_ROLE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("arguments", re.compile(
        r"\b(learned|ld\.)\s+(senior\s+|additional\s+|solicitor\s+)*(counsel|advocate|solicitor|attorney)\b"
        r"|\b(appellants?|respondents?|petitioners?|state|accused|complainant)\b[^.]{0,60}\b(submit(ted|s)?|contend(ed|s)?|argu(ed|es)|urge[ds]?)\b"
        r"|^(shri|mr\.|ms\.|smt\.|dr\.)\s[^.]{0,80}\b(submit(ted)?|contend(ed)?|argued|urged)\b", re.I)),
    ("holding", re.compile(
        r"\bwe\s+(hold|declare|direct|set aside|quash)\b"
        r"|\b(appeals?|petitions?|writ petitions?|applications?)\b[^.]{0,50}\b(is|are|stands?)\s+(hereby\s+)?(partly\s+)?(allowed|dismissed|disposed of)\b"
        r"|\bimpugned\s+(order|judgment)[^.]{0,80}\b(is|are|stands?)\s+(hereby\s+)?(set aside|quashed)\b"
        r"|\baccordingly,?\s+(the\s+)?(appeal|petition)", re.I)),
    ("facts", re.compile(
        r"\bbrief facts\b|\bfactual (matrix|background)\b|\baggrieved by\b|\bFIR\b|\bfirst information report\b"
        r"|\bwas\s+(registered|lodged|convicted|acquitted|sentenced|arrested|granted|issued|filed)\b"
        r"|\b(high court|trial court|sessions court|special court|tribunal)\b[^.]{0,100}\b(by|vide)\s+(the\s+)?(impugned\s+)?(order|judgment)\b", re.I)),
    ("reasoning", re.compile(
        r"\bwe\s+are\s+of\s+the\s+(considered\s+|firm\s+)?(view|opinion)\b|\bin\s+our\s+(considered\s+)?(view|opinion)\b"
        r"|\bit\s+is\s+(well|trite|settled)\b|\bthis\s+court\s+(has\s+)?(held|observed|reiterated)\b"
        r"|\bthe\s+question\s+(that\s+)?(arises|for\s+consideration)\b|\bwe\s+(find|are\s+unable\s+to)\b", re.I)),
]


def classify_role(paragraph: str) -> str:
    """Keyword heuristic for what a judgment paragraph is doing. Order matters: what parties
    argued and what the court held are the two distinctions the verifier cares about, so they are
    tested first. Falls back to 'judgment_text' rather than guessing."""
    for label, rx in _ROLE_RULES:
        if rx.search(paragraph):
            return label
    return "judgment_text"


def _mk_chunks(doc: ExtractedDoc, start: int, end: int, max_words: int, **kw) -> list[StructuredChunk]:
    out = []
    for cs, ce in _split_words(doc, start, end, max_words):
        out.append(StructuredChunk(text=doc.text[cs:ce], char_start=cs, char_end=ce,
                                   page_start=doc.page_of(cs), page_end=doc.page_of(max(cs, ce - 1)), **kw))
    return out


def parse_judgment(doc: ExtractedDoc, *, max_words: int) -> ParsedJudgment:
    refs = doc.line_refs()
    res = ParsedJudgment(chunks=[], reporter_citation=None, neutral_citation=None, n_paragraphs=0)
    if not refs:
        res.problems.append("no text extracted")
        return res
    head_text = "\n".join(r.text for r in refs[:14])
    m = _SCR_HEADER.search(head_text)
    if m:
        res.reporter_citation = f"[{m.group(1)}] {m.group(2)} S.C.R. {m.group(3)}"
        res.neutral_citation = re.sub(r"\s+", " ", m.group(4))
    else:
        res.warnings.append("no 'S.C.R. : INSC' citation header found on the first page")

    # -- where the judgment body starts ------------------------------------------------------
    body = next((i for i, r in enumerate(refs) if _BODY_MARKER.match(r.text)), None)
    if body is None:
        jur = next((i for i, r in enumerate(refs) if "JURISDICTION" in r.text.upper() and ":" in r.text), 0)
        body = next((i for i in range(jur, len(refs)) if _PARA.match(refs[i].text) and int(_PARA.match(refs[i].text).group(1)) == 1), None)
        if body is None:
            res.problems.append("cannot locate the start of the judgment (no 'JUDGMENT' marker or paragraph 1)")
            return res
        res.warnings.append("no 'JUDGMENT' marker; body located at paragraph 1")

    # -- editorial front matter (labelled headnote) -----------------------------------------
    bench = next((i for i, r in enumerate(refs[:25]) if _BENCH.search(r.text.strip())), None)
    front_from = (bench + 1) if bench is not None else 0
    block_name, block_first = "headnote", front_from
    blocks: list[tuple[str, int, int]] = []
    for i in range(front_from, body):
        low = refs[i].text.strip().lower()
        if low in _FRONT_HEADINGS:
            if i > block_first:
                blocks.append((block_name, block_first, i))
            block_name, block_first = low, i + 1
    if body > block_first:
        blocks.append((block_name, block_first, body))
    for name, a, b in blocks:
        if name in _DROP_BLOCKS or b <= a:
            continue
        s, e = refs[a].start, refs[b - 1].end
        if name == "list of acts":
            res.list_of_acts = " ".join(doc.text[s:e].split())
        res.chunks += _mk_chunks(doc, s, e, max_words, section_label="headnote",
                                 metadata={"editorial": True, "block": name})

    # -- numbered paragraphs ------------------------------------------------------------------
    tail = next((i for i in range(body, len(refs)) if _TAIL.search(refs[i].text)), len(refs))
    expected, opinion = 1, 1
    starts: list[tuple[int, str, int]] = []  # (line index, number label, opinion)
    for i in range(body, tail):
        m = _PARA.match(refs[i].text)
        if not m:
            continue
        n = int(m.group(1))
        if n == expected:
            starts.append((i, str(n), opinion))
            expected += 1
        elif n == 1 and expected > 3 and any(
                _BENCH.search(refs[j].text.strip()) or _BODY_MARKER.match(refs[j].text)
                for j in range(max(body, i - 4), i)):
            opinion += 1  # a separate opinion (concurring / dissenting) restarting the numbering
            starts.append((i, "1", opinion))
            expected = 2
    if not starts:
        res.problems.append("no numbered paragraphs found in the judgment body")
        return res
    res.n_paragraphs = len(starts)
    lead_first = body + (1 if _BODY_MARKER.match(refs[body].text) else 0)
    if starts[0][0] > lead_first:  # text between the JUDGMENT marker and paragraph 1 (bench, "Leave granted.")
        lead_s, lead_e = refs[lead_first].start, refs[starts[0][0] - 1].end
        if _words(doc.text[lead_s:lead_e]) >= 12:
            res.chunks += _mk_chunks(doc, lead_s, lead_e, max_words, section_label="judgment_text",
                                     metadata={"unnumbered": True, "role_source": "none"})
    for k, (li, label, op) in enumerate(starts):
        nxt = starts[k + 1][0] if k + 1 < len(starts) else tail
        s, e = refs[li].start, refs[nxt - 1].end
        text = doc.text[s:e]
        role = classify_role(text)
        res.chunks += _mk_chunks(
            doc, s, e, max_words, section_label=role, paragraph_number=label,
            metadata={"role_source": "heuristic_v1", "opinion": op})
    if tail == len(refs):
        res.warnings.append("no 'Result of the case' / 'Headnotes prepared by' tail marker: the last paragraph may include editorial matter")
    return res
