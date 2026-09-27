"""Corpus command line:

    python -m legalrag.corpus fetch    --config real_corpus [--only UID ...] [--force]
    python -m legalrag.corpus import   --config real_corpus UID path/to/file.pdf      # hand-downloaded file
    python -m legalrag.corpus validate --config real_corpus                            # no database needed
    python -m legalrag.corpus build    --config real_corpus [--only UID ...] [--include-pending]
    python -m legalrag.corpus status   --config real_corpus
    python -m legalrag.corpus discover-sc --config real_corpus --years 2024 2025 --terms "Bharatiya Nyaya Sanhita" ...

Nothing here bypasses a CAPTCHA or an access control: see legalrag/corpus/fetch.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

from legalrag.corpus.fetch import FetchError, Fetcher, import_file
from legalrag.corpus.ingest import build, manifest_path, paths_for, process_entry
from legalrag.corpus.manifest import Manifest, load_lock, load_manifest, save_lock
from legalrag.corpus.validate import validate_manifest
from legalrag.settings import Config, load_config

S3_HOST = "indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"
S3_BASE = f"https://{S3_HOST}"


def _select(manifest: Manifest, only: list[str] | None):
    unknown = sorted(set(only or []) - {e.doc_uid for e in manifest.entries})
    if unknown:
        raise SystemExit(f"not in the manifest: {unknown}")
    return [e for e in manifest.entries if not only or e.doc_uid in only]


def cmd_fetch(cfg: Config, args) -> int:
    mpath = manifest_path(cfg)
    manifest = load_manifest(mpath)
    locks = load_lock(mpath)
    raw = paths_for(cfg).raw
    fetcher = Fetcher(cfg.corpus)
    failed = 0
    try:
        for e in _select(manifest, args.only):
            try:
                rec = fetcher.fetch(e, raw, force=args.force)
                locks[e.doc_uid] = rec
                print(f"ok      {e.doc_uid}  {rec.size_bytes:>9,} bytes  sha256={rec.sha256[:16]}...")
            except FetchError as ex:
                failed += 1
                print(f"REFUSED {e.doc_uid}  {ex}")
            save_lock(mpath, locks)
    finally:
        fetcher.close()
    return 1 if failed else 0


def cmd_import(cfg: Config, args) -> int:
    mpath = manifest_path(cfg)
    manifest = load_manifest(mpath)
    entry = manifest.get(args.doc_uid)
    locks = load_lock(mpath)
    try:
        locks[entry.doc_uid] = import_file(entry, Path(args.file), paths_for(cfg).raw)
    except FetchError as ex:
        print(f"REFUSED {entry.doc_uid}  {ex}")
        return 1
    save_lock(mpath, locks)
    print(f"imported {entry.doc_uid} ({locks[entry.doc_uid].size_bytes:,} bytes, sha256={locks[entry.doc_uid].sha256[:16]}...)")
    return 0


def cmd_validate(cfg: Config, args) -> int:
    mpath = manifest_path(cfg)
    manifest = load_manifest(mpath)
    locks = load_lock(mpath)
    cross = validate_manifest(manifest, cfg.corpus)
    rows, bad = [], 0
    for e in _select(manifest, args.only):
        p = process_entry(e, cfg, locks.get(e.doc_uid), locks, cross.get(e.doc_uid))
        n = None
        if p.parsed is not None:
            n = len(getattr(p.parsed, "units", None) or getattr(p.parsed, "chunks", []))
        rows.append({"doc_uid": e.doc_uid, "status": p.status, "units_or_chunks": n, "errors": p.report.errors,
                     "review": p.report.reviews, "warnings": p.report.warnings})
        bad += p.status in ("invalid", "pending_review")
        print(f"{p.status:22} {e.doc_uid}")
        for m in p.report.errors + p.report.reviews:
            print(f"    ! {m}")
        for m in p.report.warnings:
            print(f"    - {m}")
    out = paths_for(cfg).root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {out / 'validation.json'}")
    return 1 if bad else 0


def cmd_build(cfg: Config, args) -> int:
    from legalrag.embed import build_embedder
    from legalrag.store.db import connect, migrate

    applied = migrate(cfg)
    if applied:
        print("migrations applied:", applied)
    results = build(cfg, build_embedder(cfg.embedding), lambda: connect(cfg), only=args.only,
                    include_pending=args.include_pending)
    for r in results:
        print(f"{r.action:9} {r.status:22} {r.doc_uid:28} chunks={r.n_chunks}")
        for m in r.errors + r.reviews:
            print(f"    ! {m}")
        for m in r.warnings:
            print(f"    - {m}")
    return 1 if any(r.action in ("rejected", "failed", "removed") for r in results) else 0


def cmd_report(cfg: Config, args) -> int:
    """Data-quality report (spec Phase 12): counts from the manifest's current validation state
    plus what is actually in the database. Reuses `process_entry`/`validate_manifest` (no new
    validation logic) and existing tables (`documents`, `chunks`, `document_provenance`); nothing
    new is added to the schema."""
    from legalrag.store.db import connect

    mpath = manifest_path(cfg)
    manifest = load_manifest(mpath)
    locks = load_lock(mpath)
    cross = validate_manifest(manifest, cfg.corpus)
    if args.only:
        print(f"NOTE: --only restricts validation to {args.only}; error/URL/duplicate counts below "
              f"are partial (the database counts are not).")
    entries, errors_n, missing_url_n, duplicate_n = [], 0, 0, 0
    for e in _select(manifest, args.only):
        p = process_entry(e, cfg, locks.get(e.doc_uid), locks, cross.get(e.doc_uid))
        codes = {f.code for f in p.report.findings}
        errors_n += bool(p.report.errors)
        missing_url_n += "MISSING_SOURCE_URL" in codes
        duplicate_n += bool({"DUPLICATE_FILE", "DUPLICATE_CITATION"} & codes)
        entries.append({"doc_uid": e.doc_uid, "doc_type": e.doc_type, "status": p.status,
                        "errors": p.report.errors, "reviews": p.report.reviews})
    with connect(cfg) as conn:
        by_type = {r["doc_type"]: r["n"] for r in conn.execute(
            "SELECT doc_type, count(*) n FROM documents GROUP BY doc_type")}
        n_docs = conn.execute("SELECT count(*) n FROM documents").fetchone()["n"]
        n_chunks = conn.execute("SELECT count(*) n FROM chunks").fetchone()["n"]
        n_pages = conn.execute("SELECT coalesce(sum(n_pages), 0) n FROM document_provenance").fetchone()["n"]
    stats = {
        "total_documents": n_docs,
        "total_statutes": by_type.get("statute", 0),
        "total_judgments": by_type.get("judgment", 0),
        "total_pages": n_pages,
        "total_chunks": n_chunks,
        "total_embeddings": n_chunks,   # `chunks.embedding` is NOT NULL: one embedding per chunk
        "manifest_entries": len(manifest.entries),
        "documents_with_errors": errors_n,
        "documents_without_source_urls": missing_url_n,
        "duplicate_documents": duplicate_n,
    }
    print("Corpus report")
    for k, v in stats.items():
        print(f"  {k.replace('_', ' ').capitalize():32} {v}")
    out = paths_for(cfg).root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_report.json").write_text(
        json.dumps({"stats": stats, "entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {out / 'corpus_report.json'}")
    return 0


def cmd_status(cfg: Config, args) -> int:
    from legalrag.store.db import connect

    with connect(cfg) as conn:
        for r in conn.execute("SELECT verification_status s, source_authority a, doc_type t, count(*) n FROM documents "
                              "GROUP BY 1,2,3 ORDER BY 1,2,3"):
            print(f"{r['s']:22} {r['a']:14} {r['t']:9} {r['n']}")
        print("\nrecent ingestion events (rejections/failures first):")
        for r in conn.execute("SELECT manifest_id, stage, status, message FROM ingestion_events "
                              "ORDER BY (status IN ('rejected','failed')) DESC, id DESC LIMIT 20"):
            print(f"  {r['status']:9} {r['manifest_id']:28} {r['stage']:9} {r['message'][:90]}")
    return 0


# ------------------------------------------------------------------------------- discovery

def _title_case_party(s: str) -> str:
    return " ".join(w for w in s.replace("�", "'").split())


def _entry_from_metadata(m: dict) -> dict:
    d, mo, y = m["decision_date"].split("-")
    decided = date(int(y), int(mo), int(d))
    petitioner, respondent = _title_case_party(m["petitioner"]), _title_case_party(m["respondent"])
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", petitioner.lower()).split()][:4]
    uid = "SC-" + (m["case_id"] or f"{y}-SCR-{m['path']}").replace(" ", "-")
    return {
        "doc_uid": uid, "doc_type": "judgment", "title": f"{petitioner} v. {respondent}",
        "source_authority": "supplementary", "language": "en", "document_status": "reported_judgment",
        "source": {"name": "Indian Supreme Court Judgments dataset (Dattam Labs, AWS Open Data; derived from eCourts)",
                   "url": f"{S3_BASE}/data/pdf/year={m['year']}/english/{m['path']}_EN.pdf",
                   "landing_url": "https://registry.opendata.aws/indian-supreme-court-judgments/",
                   "license": "CC-BY-4.0 (as stated on the dataset's registry page)"},
        "local_file": f"sc/{m['year']}/{m['path']}_EN.pdf",
        "identity": {"pages": 2, "all_of": [m["case_id"]] if m["case_id"] else [" ".join(words)]},
        "court": "Supreme Court of India", "decided_on": decided.isoformat(),
        "neutral_citation": m["case_id"] or None, "reporter_citations": [m["citation"]] if m["citation"] else [],
        "notes": f"from dataset metadata; CNR {m['cnr']}",
    }


def cmd_discover(cfg: Config, args) -> int:
    """List reported Supreme Court judgments whose first pages mention any of `--terms`. Downloads
    to <corpus>/discovery (a cache, not the corpus), politely, then writes CANDIDATES for a person to
    approve: nothing is added to the manifest by this command."""
    import pyarrow.parquet as pq
    import pdfplumber

    fetcher = Fetcher(cfg.corpus)
    disc = paths_for(cfg).root / "discovery"
    disc.mkdir(parents=True, exist_ok=True)
    terms = [re.compile(t, re.I) for t in args.terms]
    found: list[dict] = []
    try:
        for year in args.years:
            meta_path = disc / f"metadata_{year}.parquet"
            url = f"{S3_BASE}/metadata/parquet/year={year}/metadata.parquet"
            fetcher.check_allowed(url)
            if not meta_path.exists():
                fetcher._throttle(S3_HOST)
                meta_path.write_bytes(fetcher.client.get(url).content)  # type: ignore[union-attr]
            rows = pq.read_table(meta_path).to_pylist()
            print(f"{year}: {len(rows)} judgments in metadata")
            for i, m in enumerate(rows[: args.limit or None]):
                pdf_url = f"{S3_BASE}/data/pdf/year={m['year']}/english/{m['path']}_EN.pdf"
                cache = disc / f"{m['path']}_EN.pdf"
                if not cache.exists():
                    fetcher.check_allowed(pdf_url)
                    fetcher._throttle(S3_HOST)
                    r = fetcher.client.get(pdf_url)  # type: ignore[union-attr]
                    if r.status_code != 200:
                        continue
                    cache.write_bytes(r.content)
                with pdfplumber.open(cache) as pdf:
                    text = " ".join((pg.extract_text(x_tolerance=1.0) or "") for pg in pdf.pages[:3])
                hits = sorted({t.pattern for t in terms if t.search(text)})
                if hits:
                    e = _entry_from_metadata(m)
                    e["notes"] += f"; matched terms in first 3 pages: {hits}"
                    found.append(e)
                if (i + 1) % 100 == 0:
                    print(f"  {year}: screened {i + 1}, candidates so far {len(found)}")
    finally:
        fetcher.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# CANDIDATES ONLY - generated by `corpus discover-sc` from dataset metadata.\n"
                   "# Review, then move chosen entries into manifest.yaml. Values come from the source's own metadata;\n"
                   "# `corpus validate` re-checks each against the PDF's own header before anything is indexed.\n"
                   + yaml.safe_dump({"version": 1, "entries": found}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{len(found)} candidates -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m legalrag.corpus", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, only=True):
        p.add_argument("--config", required=True)
        if only:
            p.add_argument("--only", nargs="*", help="restrict to these doc_uids")

    p = sub.add_parser("fetch"); common(p); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("import"); common(p, False); p.add_argument("doc_uid"); p.add_argument("file"); p.set_defaults(fn=cmd_import)
    p = sub.add_parser("validate"); common(p); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("build"); common(p); p.add_argument("--include-pending", action="store_true"); p.set_defaults(fn=cmd_build)
    p = sub.add_parser("status"); common(p, False); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("report"); common(p); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("discover-sc"); common(p, False)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--terms", nargs="+", required=True, help="regexes; a judgment matches if any occurs in its first 3 pages")
    p.add_argument("--limit", type=int, default=0, help="screen only the first N judgments per year (0 = all)")
    p.add_argument("--out", default="corpus/candidates_sc.yaml"); p.set_defaults(fn=cmd_discover)

    args = ap.parse_args(argv)
    return args.fn(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
