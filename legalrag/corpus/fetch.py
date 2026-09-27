"""Acquire original files, unmodified, from allowlisted hosts; or import ones a human downloaded.

What this deliberately does NOT do: solve or bypass CAPTCHAs, spoof a browser, ignore robots.txt,
or fetch from a host that is not on `corpus.source_allowlist`. Sources that need any of that (the
eCourts portal, eGazette) go through `import_file`, i.e. a person downloads the file in a browser
and registers it against a manifest entry.

Each successful acquisition writes the original bytes to <corpus_dir>/raw/<local_file> exactly as
received, and records retrieval time, SHA-256, size and HTTP validators in the lock file.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from legalrag.corpus.manifest import LockRecord, ManifestEntry, utcnow
from legalrag.settings import CorpusConfig

PDF_MAGIC = b"%PDF-"


class FetchError(RuntimeError):
    """A refused or failed acquisition. The message is what gets recorded as the reason."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pdf_bytes(data: bytes, *, min_bytes: int = 1024) -> None:
    if len(data) < min_bytes:
        raise FetchError(f"file too small to be a real document ({len(data)} bytes)")
    if not data.startswith(PDF_MAGIC):
        raise FetchError("not a PDF (missing %PDF- header); the server may have returned an HTML/error/CAPTCHA page")
    if b"%%EOF" not in data[-2048:]:
        raise FetchError("PDF is truncated or corrupted (no %%EOF marker at the end)")


@dataclass
class Fetcher:
    cfg: CorpusConfig
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | bool] = {}
        self._own_client = self.client is None
        if self.client is None:
            self.client = httpx.Client(follow_redirects=True, timeout=self.cfg.download_timeout_s)
        # Always identify honestly, whichever client was supplied.
        self.client.headers["User-Agent"] = self.cfg.user_agent

    def close(self) -> None:
        if self._own_client and self.client is not None:
            self.client.close()

    # -- policy ----------------------------------------------------------------------------
    def check_allowed(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise FetchError(f"refusing non-https URL: {url}")
        if parts.hostname not in set(self.cfg.source_allowlist):
            raise FetchError(f"host {parts.hostname!r} is not in corpus.source_allowlist")
        if self.cfg.respect_robots_txt and not self._robots_ok(parts.scheme, parts.netloc, url):
            raise FetchError(f"robots.txt for {parts.netloc} disallows {parts.path}")

    def _robots_ok(self, scheme: str, netloc: str, url: str) -> bool:
        """RFC 9309: 4xx on robots.txt = no restrictions; 5xx or unreachable = assume disallowed."""
        if netloc not in self._robots:
            self._throttle(netloc)
            try:
                r = self.client.get(f"{scheme}://{netloc}/robots.txt")  # type: ignore[union-attr]
            except httpx.HTTPError:
                self._robots[netloc] = False
            else:
                if r.status_code == 200:
                    rp = urllib.robotparser.RobotFileParser()
                    rp.parse(r.text.splitlines())
                    self._robots[netloc] = rp
                elif 400 <= r.status_code < 500:
                    self._robots[netloc] = True
                else:
                    self._robots[netloc] = False
        state = self._robots[netloc]
        if isinstance(state, bool):
            return state
        return state.can_fetch(self.cfg.user_agent, url)

    def _throttle(self, host: str) -> None:
        wait = self.cfg.download_min_interval_s - (time.monotonic() - self._last_hit.get(host, -1e9))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    # -- acquisition -----------------------------------------------------------------------
    def fetch(self, entry: ManifestEntry, raw_dir: Path, *, force: bool = False) -> LockRecord:
        if entry.source.kind != "http":
            raise FetchError("source.kind is manual_import: use `corpus import`, not `fetch`")
        dest = raw_dir / entry.local_file
        self.check_allowed(entry.source.url)
        if dest.exists() and not force:
            data = dest.read_bytes()
            check_pdf_bytes(data)
            return LockRecord(doc_uid=entry.doc_uid, url=entry.source.url, retrieved_at=utcnow(),
                              sha256=sha256_bytes(data), size_bytes=len(data), method="http")
        host = urlsplit(entry.source.url).netloc
        self._throttle(host)
        limit = self.cfg.max_download_mb * 1024 * 1024
        try:
            with self.client.stream("GET", entry.source.url) as r:  # type: ignore[union-attr]
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code} from {host}")
                chunks, size = [], 0
                for block in r.iter_bytes():
                    size += len(block)
                    if size > limit:
                        raise FetchError(f"download exceeds max_download_mb={self.cfg.max_download_mb}")
                    chunks.append(block)
                headers = r.headers
                status = r.status_code
        except httpx.HTTPError as e:
            raise FetchError(f"network error: {e}") from e
        data = b"".join(chunks)
        check_pdf_bytes(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return LockRecord(
            doc_uid=entry.doc_uid, url=entry.source.url, retrieved_at=utcnow(), sha256=sha256_bytes(data),
            size_bytes=len(data), http_status=status, content_type=headers.get("content-type"),
            etag=headers.get("etag"), last_modified=headers.get("last-modified"), method="http")


def import_file(entry: ManifestEntry, src: Path, raw_dir: Path) -> LockRecord:
    """Register a file a person downloaded by hand (the documented workflow for CAPTCHA-gated
    portals). The bytes are copied as they are; the lock file records that it was a manual import."""
    src = Path(src)
    if not src.is_file():
        raise FetchError(f"no such file: {src}")
    data = src.read_bytes()
    check_pdf_bytes(data)
    dest = raw_dir / entry.local_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    return LockRecord(doc_uid=entry.doc_uid, url=entry.source.url, retrieved_at=utcnow(),
                      sha256=sha256_bytes(data), size_bytes=len(data), method="manual_import")
