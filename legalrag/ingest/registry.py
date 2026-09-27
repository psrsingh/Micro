"""Citation registry builder. Runs from bulk name+citation listings without full text.

    python -m legalrag.ingest.registry --config X --listing path/to/listing.csv

CSV columns: case_name, citation, court, decided_on (ISO date, may be blank).
Rows whose citation matches a document we hold are recorded as in_corpus and linked to
it, so the authority check can tell "exists, outside our corpus" from "does not exist".
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import psycopg

from legalrag.settings import load_config
from legalrag.store.db import connect, migrate
from legalrag.store.repos import RegistryRepo


def load_listing(conn: psycopg.Connection, path: Path) -> dict[str, int]:
    repo = RegistryRepo(conn)
    stats = {"rows": 0, "in_corpus": 0, "outside_corpus": 0}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name, cit = row["case_name"].strip(), row["citation"].strip()
            if not name or not cit:
                raise ValueError(f"{path.name}: row with empty case_name/citation: {row}")
            doc = conn.execute(
                "SELECT id FROM documents WHERE neutral_citation = %s", (cit,)
            ).fetchone()
            decided = date.fromisoformat(row["decided_on"]) if row.get("decided_on") else None
            repo.upsert(case_name=name, citation=cit, court=(row.get("court") or None),
                        decided_on=decided, in_corpus=doc is not None,
                        document_id=doc["id"] if doc else None)
            stats["rows"] += 1
            stats["in_corpus" if doc else "outside_corpus"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--listing", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    migrate(cfg)
    with connect(cfg) as conn:
        print(load_listing(conn, args.listing))


if __name__ == "__main__":
    main()
