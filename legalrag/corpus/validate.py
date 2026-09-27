"""Validation and the four-way trust decision.

Every check returns a `Finding` with a severity:
  error   -> the document is `invalid` and is never indexed (bad file, wrong document, broken numbering);
  review  -> a person has to look (e.g. scanned pages with OCR off): `pending_review`, not indexed;
  warning -> indexed, and the warning is stored with the document's provenance.

`official_verified` needs (a) the manifest to claim an official source, (b) that source's host to be a
government domain, (c) the downloaded bytes to carry the expected identity text, and (d) no error/review
finding. A mirror copy is `supplementary_source`; it becomes `official_verified` only through a
recorded manual promotion. Software never promotes on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from legalrag.corpus.extract import ExtractedDoc
from legalrag.corpus.fetch import FetchError, check_pdf_bytes, sha256_file
from legalrag.corpus.manifest import LockRecord, Manifest, ManifestEntry, VerificationStatus
from legalrag.corpus.structure import ParsedJudgment, ParsedStatute
from legalrag.settings import CorpusConfig

OFFICIAL_HOST_SUFFIXES = (".gov.in", ".nic.in")
_SC_NEUTRAL = re.compile(r"^\d{4} INSC \d+$")
_SCR = re.compile(r"^\[\d{4}\] \d+ S\.C\.R\. \d+$")
_REPORTER_FORMATS = (
    _SCR,
    re.compile(r"^\(\d{4}\) \d+ SCC \d+$"),
    re.compile(r"^AIR \d{4} SC \d+$"),
    re.compile(r"^\d{4} SCC OnLine SC \d+$"),
)
_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_DATE_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b")


@dataclass(frozen=True)
class Finding:
    severity: str   # error | review | warning
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass
class ValidationReport:
    doc_uid: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str) -> None:
        self.findings.append(Finding(severity, code, message))

    def of(self, severity: str) -> list[str]:
        return [str(f) for f in self.findings if f.severity == severity]

    @property
    def errors(self) -> list[str]:
        return self.of("error")

    @property
    def reviews(self) -> list[str]:
        return self.of("review")

    @property
    def warnings(self) -> list[str]:
        return self.of("warning")


def is_official_host(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return host.endswith(OFFICIAL_HOST_SUFFIXES)


def decide_status(entry: ManifestEntry, rep: ValidationReport) -> VerificationStatus:
    if rep.errors:
        return "invalid"
    if rep.reviews:
        return "pending_review"
    if entry.source_authority == "official" or entry.promoted_by:
        return "official_verified"
    return "supplementary_source"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# --------------------------------------------------------------------------- manifest-level checks

def validate_manifest(manifest: Manifest, cfg: CorpusConfig) -> dict[str, list[Finding]]:
    """Checks that need the whole manifest: duplicate citations, conflicting Act versions."""
    out: dict[str, list[Finding]] = {e.doc_uid: [] for e in manifest.entries}
    seen_cit: dict[str, str] = {}
    for e in manifest.entries:
        if e.doc_type == "judgment" and e.neutral_citation:
            other = seen_cit.get(e.neutral_citation)
            if other:
                out[e.doc_uid].append(Finding("error", "DUPLICATE_CITATION",
                                              f"neutral citation {e.neutral_citation!r} is also used by {other}"))
            else:
                seen_cit[e.neutral_citation] = e.doc_uid
    acts = [e for e in manifest.entries if e.doc_type == "statute"]
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            if (a.act_name, a.act_year) != (b.act_name, b.act_year):
                continue
            a0, a1 = a.period.valid_from or date.min, a.period.valid_until or date.max
            b0, b1 = b.period.valid_from or date.min, b.period.valid_until or date.max
            if a0 <= b1 and b0 <= a1:
                msg = f"{a.doc_uid} and {b.doc_uid} are both {a.act_name} ({a.act_year}) with overlapping validity"
                out[a.doc_uid].append(Finding("error", "CONFLICTING_VERSIONS", msg))
                out[b.doc_uid].append(Finding("error", "CONFLICTING_VERSIONS", msg))
    return out


# --------------------------------------------------------------------------- per-document checks

def validate_source(entry: ManifestEntry, lock: LockRecord | None, raw_dir: Path, cfg: CorpusConfig,
                    all_locks: dict[str, LockRecord]) -> ValidationReport:
    """Everything that can be decided before extracting text."""
    rep = ValidationReport(entry.doc_uid)
    url = entry.source.url
    if not url or not url.startswith("https://"):
        rep.add("error", "MISSING_SOURCE_URL", "source.url is missing or not https")
    elif entry.source.kind == "http" and (urlsplit(url).hostname not in set(cfg.source_allowlist)):
        rep.add("error", "HOST_NOT_ALLOWED", f"host {urlsplit(url).hostname!r} is not in source_allowlist")
    if entry.source_authority == "official" and url and not is_official_host(url):
        rep.add("error", "OFFICIAL_CLAIM_UNSUPPORTED",
                f"manifest says 'official' but {urlsplit(url).hostname!r} is not a government domain "
                f"({', '.join(OFFICIAL_HOST_SUFFIXES)}); label it 'supplementary'")
    path = raw_dir / entry.local_file
    if lock is None:
        rep.add("error", "NOT_ACQUIRED", "no lock record: run `corpus fetch` (or `corpus import`) first")
        return rep
    if not path.is_file():
        rep.add("error", "MISSING_FILE", f"{path} does not exist")
        return rep
    try:
        check_pdf_bytes(path.read_bytes())
    except FetchError as e:
        rep.add("error", "CORRUPT_FILE", str(e))
        return rep
    if sha256_file(path) != lock.sha256:
        rep.add("error", "HASH_MISMATCH", "file on disk differs from the hash recorded at download")
    for uid, other in all_locks.items():
        if uid != entry.doc_uid and other.sha256 == lock.sha256 and uid < entry.doc_uid:
            rep.add("error", "DUPLICATE_FILE", f"identical file already registered as {uid}")
    if entry.source_authority == "supplementary" and entry.promoted_by is None:
        rep.add("warning", "SUPPLEMENTARY_COPY", "third-party copy; not byte-verified against the official source")
    # metadata sanity
    if entry.doc_type == "judgment":
        c = entry.court or ""
        if "supreme court" in c.lower() and entry.neutral_citation and not _SC_NEUTRAL.match(entry.neutral_citation):
            rep.add("error", "BAD_CITATION_FORMAT", f"neutral citation {entry.neutral_citation!r} is not 'YYYY INSC N'")
        for rc in entry.reporter_citations:
            if not any(f.match(rc) for f in _REPORTER_FORMATS):
                rep.add("error", "BAD_CITATION_FORMAT",
                        f"reporter citation {rc!r} matches no known format ('[YYYY] V S.C.R. P', '(YYYY) V SCC P', 'AIR YYYY SC P')")
        if entry.decided_on and (entry.decided_on > date.today() or entry.decided_on.year < 1950):
            rep.add("error", "BAD_DATE", f"decided_on {entry.decided_on} is implausible")
        if not entry.neutral_citation:
            rep.add("warning", "NO_NEUTRAL_CITATION", "no neutral citation in the source metadata")
    else:
        if entry.period.valid_from is None:
            rep.add("error", "NO_APPLICABLE_PERIOD", "a statute needs period.valid_from (the temporal check depends on it)")
        if entry.enactment_date and entry.act_year and entry.enactment_date.year != entry.act_year:
            rep.add("error", "BAD_DATE", f"enactment_date {entry.enactment_date} does not match act_year {entry.act_year}")
        if entry.period.basis == "unverified_secondary":
            rep.add("warning", "UNVERIFIED_PERIOD",
                    "valid_from comes from a secondary source; the commencement notification has not been acquired")
        if entry.document_status == "repealed" or (entry.period.valid_until and entry.period.valid_until < date.today()):
            rep.add("warning", "OUTDATED_LEGISLATION", "this Act/version is no longer in force; answers must caveat it")
    return rep


def _find_date(lines: list[str]) -> date | None:
    text = " ".join(lines)
    for m in _DATE_DMY.finditer(text):
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                pass
    for m in _DATE_MDY.finditer(text):
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(2)))
            except ValueError:
                pass
    return None


def validate_extracted(entry: ManifestEntry, doc: ExtractedDoc, parsed: ParsedStatute | ParsedJudgment,
                       rep: ValidationReport) -> ValidationReport:
    """Identity, text quality and structure, after extraction and parsing."""
    total = sum(p.char_count for p in doc.pages)
    if total < 500:
        rep.add("error", "EMPTY_TEXT", f"only {total} characters extracted from {doc.n_pages} page(s)")
        return rep
    no_text = [p.number for p in doc.pages if p.needs_ocr]
    if no_text:
        rep.add("review", "NEEDS_OCR", f"{len(no_text)} page(s) have no text layer {no_text[:8]} and OCR is off/unavailable")
    if doc.ocr_used:
        rep.add("warning", "OCR_USED", "some pages were OCR'd: text may contain recognition errors")
    noise = doc.text.count("�") / max(1, len(doc.text))
    if noise > 0.02:
        rep.add("warning", "ENCODING_NOISE", f"{noise:.1%} of characters are replacement characters")

    first = _norm(" ".join(p.text for p in doc.pages[: entry.identity.pages]))
    for needle in entry.identity.all_of:
        if _norm(needle) not in first:
            rep.add("error", "IDENTITY_MISMATCH",
                    f"expected text {needle!r} not found in the first {entry.identity.pages} page(s): "
                    "this file may not be the document the manifest describes")

    for p in parsed.problems:
        rep.add("error", "STRUCTURE", p)
    for w in parsed.warnings:
        rep.add("warning", "STRUCTURE_WARNING", w)

    if isinstance(parsed, ParsedJudgment):
        if entry.neutral_citation:
            if parsed.neutral_citation is None:
                rep.add("error", "CITATION_NOT_IN_DOCUMENT", "manifest gives a neutral citation but the PDF header has none")
            elif parsed.neutral_citation != entry.neutral_citation:
                rep.add("error", "CITATION_MISMATCH",
                        f"PDF header says {parsed.neutral_citation!r}, manifest says {entry.neutral_citation!r}")
        scr = [c for c in entry.reporter_citations if "S.C.R." in c]
        if scr and parsed.reporter_citation and parsed.reporter_citation not in scr:
            rep.add("error", "CITATION_MISMATCH",
                    f"PDF header says {parsed.reporter_citation!r}, manifest says {scr[0]!r}")
        head = [ln.text for ln in doc.pages[0].lines[:30]]
        found = _find_date(head)
        if found is None:
            rep.add("warning", "DATE_NOT_FOUND", "could not read a decision date from the first page")
        elif entry.decided_on and found != entry.decided_on:
            rep.add("error", "DATE_MISMATCH", f"PDF first page shows {found}, manifest says {entry.decided_on}")
    else:
        if not parsed.units:
            rep.add("error", "STRUCTURE", "no sections parsed")
    return rep
