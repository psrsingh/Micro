"""The corpus manifest: the hand-authored, reviewable record of every document we intend to hold.

Nothing enters the database unless it is listed here, and nothing listed here is trusted just because
it is listed: `validate.py` re-derives identity from the downloaded bytes. The manifest states
*claims* (this file is BNS 2023, from this URL); validation decides whether the file backs them.

`manifest.lock.json` (next to the manifest) is machine-written: retrieval time, hash and size of
each original file. Keeping it apart means a re-fetch never rewrites the reviewed manifest.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

VerificationStatus = Literal["official_verified", "supplementary_source", "pending_review", "invalid"]
SourceAuthority = Literal["official", "supplementary"]

INDEXABLE_STATUSES = ("official_verified", "supplementary_source")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(_Strict):
    name: str                       # human name of the publisher, e.g. "Ministry of Home Affairs"
    url: str                        # where this exact file is fetched from (https)
    landing_url: str | None = None  # page a human would use to find/confirm it
    license: str | None = None      # only what the source actually states; never guessed
    kind: Literal["http", "manual_import"] = "http"


class Identity(_Strict):
    """Text that MUST appear in the extracted document for it to be what the manifest says.
    `all_of` strings are searched (whitespace/case-insensitive) in the first `pages` pages."""

    pages: int = 3
    all_of: list[str] = Field(min_length=1)


class ApplicablePeriod(_Strict):
    valid_from: date | None = None
    valid_until: date | None = None
    # How valid_from was established. 'unverified_secondary' means we have a date from a
    # non-primary source and have NOT seen the commencement notification: it is recorded, and the
    # answer layer can say so, rather than presented as established.
    basis: Literal["commencement_notification", "gazette_text", "unverified_secondary", "not_applicable"] \
        = "not_applicable"
    note: str = ""


class ManifestEntry(_Strict):
    doc_uid: str = Field(pattern=r"^[A-Z0-9][A-Z0-9_.-]{2,60}$")
    doc_type: Literal["statute", "judgment"]
    title: str
    source_authority: SourceAuthority
    source: Source
    local_file: str                  # relative to <corpus_dir>/raw
    identity: Identity
    language: str = "en"
    document_status: Literal["in_force", "repealed", "reported_judgment", "unknown"] = "unknown"

    # statutes
    act_name: str | None = None
    act_year: int | None = None
    act_number: str | None = None    # e.g. "45 of 2023"
    enactment_date: date | None = None
    period: ApplicablePeriod = Field(default_factory=ApplicablePeriod)
    # PDF layout hint: x-coordinate where a right-hand marginal-note column starts (Gazette pages).
    margin_x: float | None = None
    margin_left_x: float | None = None  # ... and the left-hand column on mirrored (even) pages
    # 'section': Act -> Chapter -> Section -> sub-section (parse_statute; Gazette-style PDFs).
    # 'article': Constitution-style PDFs -> Part -> (Chapter) -> Article -> clause, numbered
    # articles with lettered amendment insertions (239AA, 371A), no marginal-note column, and a
    # front-matter table of contents to skip (parse_constitution).
    # 'india_code': India Code consolidations -> inline titles, lettered insertions, repealed-section
    # stubs, an "Arrangement of Sections" to skip (parse_india_code). Use with strip_footnotes.
    numbering: Literal["section", "article", "india_code"] = "section"
    # Drop amendment footnotes below each page's footnote rule (India Code PDFs).
    strip_footnotes: bool = False

    # judgments
    court: str | None = None
    decided_on: date | None = None
    neutral_citation: str | None = None
    reporter_citations: list[str] = Field(default_factory=list)
    case_number: str | None = None

    # Set by a human, never by this software: 'official_verified' for a mirror copy requires a
    # recorded manual check against the official portal.
    promoted_by: str | None = None
    promoted_on: date | None = None
    promotion_note: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _shape(self):
        if self.doc_type == "statute" and not (self.act_name and self.act_year):
            raise ValueError(f"{self.doc_uid}: a statute needs act_name and act_year")
        if self.doc_type == "judgment" and not (self.court and self.decided_on):
            raise ValueError(f"{self.doc_uid}: a judgment needs court and decided_on")
        if self.promoted_by and not (self.promoted_on and self.promotion_note):
            raise ValueError(f"{self.doc_uid}: a manual promotion needs promoted_on and promotion_note")
        return self


class Manifest(_Strict):
    version: int = 1
    description: str = ""
    entries: list[ManifestEntry]

    @model_validator(mode="after")
    def _unique(self):
        uids = [e.doc_uid for e in self.entries]
        dupes = sorted({u for u in uids if uids.count(u) > 1})
        if dupes:
            raise ValueError(f"duplicate doc_uid in manifest: {dupes}")
        files = [e.local_file for e in self.entries]
        dupes = sorted({f for f in files if files.count(f) > 1})
        if dupes:
            raise ValueError(f"two manifest entries share a local_file: {dupes}")
        return self

    def get(self, doc_uid: str) -> ManifestEntry:
        for e in self.entries:
            if e.doc_uid == doc_uid:
                return e
        raise KeyError(doc_uid)


def load_manifest(path: Path) -> Manifest:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Manifest.model_validate(raw)


# -- lock file ---------------------------------------------------------------------------------

class LockRecord(_Strict):
    doc_uid: str
    url: str
    retrieved_at: datetime
    sha256: str
    size_bytes: int
    http_status: int | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    method: Literal["http", "manual_import"] = "http"


def lock_path_for(manifest_path: Path) -> Path:
    return Path(manifest_path).with_name("manifest.lock.json")


def load_lock(manifest_path: Path) -> dict[str, LockRecord]:
    p = lock_path_for(manifest_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {r["doc_uid"]: LockRecord.model_validate(r) for r in data.get("records", [])}


def save_lock(manifest_path: Path, records: dict[str, LockRecord]) -> None:
    p = lock_path_for(manifest_path)
    payload = {"records": [r.model_dump(mode="json") for _, r in sorted(records.items())]}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
