"""PDF -> per-page text, keeping page numbers and the layout facts the legal parsers need.

Handles three things a plain `extract_text()` gets wrong on Indian legal PDFs:
  * justified lines whose word gaps are narrower than pdfminer's default tolerance
    ("ofanyproperty,ortoexecute"): `x_tolerance=1.0` recovers them;
  * a right-hand marginal-note column (Gazette pages, where the section title sits) that would
    otherwise be interleaved with the body: split at `margin_x` and returned separately;
  * running headers/footers and Supreme Court Reports margin letters (A, B, C, ...), which are page
    furniture, not law: removed when they repeat, and only then;
  * amendment footnotes under a short rule at the foot of India Code pages ("2. Subs. by Act 3 of
    1951, ..."), which read like section starts: dropped below the rule when `strip_footnotes` is set.

Nothing is guessed: a page with no text layer is flagged `needs_ocr`; it is only OCR'd when
`corpus.ocr_enabled` is set and Tesseract is installed, otherwise the document is not ingested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from legalrag.settings import CorpusConfig

_X_TOL = 1.0
_LINE_TOL = 3.0
_MARGIN_LETTERS = "ABCDEFGH"
_EDGE = 3  # a running header can be three lines: title, page number, rule (Gazette pages)
# India Code footnote separator: a short rule starting at the left text edge, well below the page top.
_FN_RULE_WIDTH = (80.0, 250.0)


class ExtractionError(RuntimeError):
    pass


@dataclass
class Line:
    text: str
    top: float
    x0: float


@dataclass
class LineRef:
    page: int      # 1-based PDF page
    start: int     # character span of the line in ExtractedDoc.text
    end: int
    line: Line

    @property
    def text(self) -> str:
        return self.line.text


@dataclass
class MarginNote:
    text: str
    top: float


@dataclass
class ExtractedPage:
    number: int  # 1-based, position in the PDF file
    lines: list[Line] = field(default_factory=list)
    margin: list[MarginNote] = field(default_factory=list)
    char_count: int = 0
    needs_ocr: bool = False
    ocr_used: bool = False
    removed: list[str] = field(default_factory=list)  # header/footer/margin-letter lines dropped

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


@dataclass
class ExtractedDoc:
    pages: list[ExtractedPage]
    tool: str
    method: str  # 'pdfplumber' | 'pdfplumber+ocr'
    warnings: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def ocr_used(self) -> bool:
        return any(p.ocr_used for p in self.pages)

    @property
    def text(self) -> str:
        """Pages joined by form feed, so a character offset maps back to exactly one page."""
        return "\f".join(p.text for p in self.pages)

    def line_refs(self) -> list[LineRef]:
        """Every body line with its character span in `self.text` and its PDF page."""
        # text = "\f".join("\n".join(page lines)): after the last line of a page comes one "\f",
        # so advancing by len(line) + 1 after *every* line lands on the next page's start; an
        # empty page still contributes its "\f".
        refs, pos = [], 0
        for p in self.pages:
            for ln in p.lines:
                refs.append(LineRef(p.number, pos, pos + len(ln.text), ln))
                pos += len(ln.text) + 1
            if not p.lines:
                pos += 1
        return refs

    def page_starts(self) -> list[int]:
        starts, pos = [], 0
        for p in self.pages:
            starts.append(pos)
            pos += len(p.text) + 1  # + the form feed
        return starts

    def page_of(self, offset: int) -> int:
        """1-based PDF page containing the character at `offset` of `self.text`."""
        starts = self.page_starts()
        page = 1
        for i, s in enumerate(starts, start=1):
            if s <= offset:
                page = i
        return self.pages[page - 1].number


def _group_lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= _LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
    return lines


def _strip_margin_letters(words: list[dict], page_width: float) -> tuple[list[dict], list[str]]:
    """Drop Supreme Court Reports margin letters. A margin letter is a lone A-H at the far right;
    a page's letters run A, B, C... downwards. Both conditions must hold, so an ordinary
    capital 'A' at a line end is never touched."""
    cand = [w for w in words if w["text"] in tuple(_MARGIN_LETTERS) and w["x0"] >= page_width * 0.86]
    cand.sort(key=lambda w: w["top"])
    letters = [w["text"] for w in cand]
    if len(letters) >= 2 and letters == sorted(letters) and len(set(letters)) == len(letters):
        drop = {id(w) for w in cand}
        return [w for w in words if id(w) not in drop], letters
    return words, []


def _footnote_rule_top(pg) -> float | None:
    """Top of the footnote separator rule on this page, or None. The rule is the only short
    horizontal line that starts at the left text edge in the lower two-thirds of the page."""
    words = pg.extract_words(x_tolerance=_X_TOL)
    if not words:
        return None
    left = min(w["x0"] for w in words)
    tops = [e["top"] for e in pg.edges
            if abs(e["top"] - e["bottom"]) < 1 and abs(e["x0"] - left) <= 4
            and _FN_RULE_WIDTH[0] <= e["x1"] - e["x0"] <= _FN_RULE_WIDTH[1] and e["top"] > pg.height / 3]
    return min(tops) if tops else None


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text.lower())).strip()


def _strip_running_furniture(pages: list[ExtractedPage]) -> None:
    """Remove lines that repeat in the first / last three positions of many pages (running
    headers and footers, including page numbers). Only those positions, and only repeats."""
    if len(pages) < 3:
        return
    counts: dict[str, int] = {}
    for p in pages:
        seen = set()
        for ln in p.lines[:_EDGE] + p.lines[-_EDGE:]:
            k = _norm_key(ln.text)
            if k and k not in seen:
                seen.add(k)
                counts[k] = counts.get(k, 0) + 1
    threshold = max(3, int(0.3 * len(pages)))
    repeated = {k for k, n in counts.items() if n >= threshold}
    for p in pages:
        keep, gone = [], []
        edge = set(id(x) for x in p.lines[:_EDGE] + p.lines[-_EDGE:])
        for ln in p.lines:
            if id(ln) in edge and _norm_key(ln.text) in repeated:
                gone.append(ln.text)
            else:
                keep.append(ln)
        p.lines, p.removed = keep, p.removed + gone


def _ocr_page(page, lang: str) -> str:
    try:
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise ExtractionError("corpus.ocr_enabled is set but pytesseract is not installed") from e
    try:  # pragma: no cover - needs the Tesseract binary
        return pytesseract.image_to_string(page.to_image(resolution=200).original, lang=lang)
    except Exception as e:  # noqa: BLE001
        raise ExtractionError(f"OCR failed (is the Tesseract binary installed?): {e}") from e


def extract_pdf(path: Path, cfg: CorpusConfig, *, margin_x: float | None = None,
                margin_left_x: float | None = None, strip_footnotes: bool = False) -> ExtractedDoc:
    """`margin_x`: words starting at or right of it are a right-hand marginal-note column.
    `margin_left_x`: words ending at or left of it are a left-hand one. Gazette pages mirror the
    notes (right on odd pages, left on even), so both are given for those.
    `strip_footnotes`: drop words below a page's footnote rule (India Code consolidations)."""
    warnings: list[str] = []
    pages: list[ExtractedPage] = []
    try:
        pdf = pdfplumber.open(path)
    except Exception as e:  # noqa: BLE001 - pdfminer raises many things for corrupt files
        raise ExtractionError(f"cannot open PDF: {e}") from e
    with pdf:
        if not pdf.pages:
            raise ExtractionError("PDF has no pages")
        for i, pg in enumerate(pdf.pages, start=1):
            words = pg.extract_words(x_tolerance=_X_TOL, keep_blank_chars=False, use_text_flow=False)
            body, margin = words, []
            if margin_x is not None:
                body = [w for w in body if w["x0"] < margin_x]
                margin = [w for w in words if w["x0"] >= margin_x]
            if margin_left_x is not None:
                margin += [w for w in body if w["x1"] <= margin_left_x]
                body = [w for w in body if w["x1"] > margin_left_x]
            body, letters = _strip_margin_letters(body, float(pg.width))
            ep = ExtractedPage(number=i, char_count=sum(len(w["text"]) for w in words))
            if letters:
                ep.removed.append("margin letters " + "".join(letters))
            rule = _footnote_rule_top(pg) if strip_footnotes else None
            if rule is not None:
                notes = [w for w in body if w["top"] >= rule]
                body = [w for w in body if w["top"] < rule]
                ep.removed.append(f"{len(_group_lines(notes))} footnote line(s) below the rule at y={rule:.0f}")
            for ln in _group_lines(body):
                ep.lines.append(Line(" ".join(w["text"] for w in ln), float(ln[0]["top"]), float(ln[0]["x0"])))
            for ln in _group_lines(margin):
                ep.margin.append(MarginNote(" ".join(w["text"] for w in ln), float(ln[0]["top"])))
            if ep.char_count < cfg.min_chars_per_page:
                ep.needs_ocr = True
                if cfg.ocr_enabled:
                    text = _ocr_page(pg, cfg.ocr_language)
                    ep.lines = [Line(t, float(k), 0.0) for k, t in enumerate(text.splitlines()) if t.strip()]
                    ep.ocr_used, ep.needs_ocr = True, False
                    ep.char_count = sum(len(ln.text) for ln in ep.lines)
            pages.append(ep)
    _strip_running_furniture(pages)
    ocr = any(p.ocr_used for p in pages)
    missing = [p.number for p in pages if p.needs_ocr]
    if missing:
        warnings.append(f"no text layer on {len(missing)} page(s) {missing[:10]}; OCR is off or unavailable")
    return ExtractedDoc(pages, tool=f"pdfplumber {pdfplumber.__version__}",
                        method="pdfplumber+ocr" if ocr else "pdfplumber", warnings=warnings)
