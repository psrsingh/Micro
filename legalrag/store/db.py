"""Connection handling and migrations. Plain numbered SQL files, applied once each."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from legalrag.settings import Config

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class StoreError(RuntimeError):
    pass


def _connect(dsn: str, *, vector: bool) -> psycopg.Connection:
    conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False, connect_timeout=10)
    if vector:
        register_vector(conn)
    return conn


@contextmanager
def connect(cfg: Config) -> Iterator[psycopg.Connection]:
    """Connection with pgvector types registered. Call `migrate` first on a fresh DB."""
    conn = _connect(cfg.dsn, vector=True)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate(cfg: Config) -> list[str]:
    """Apply pending migrations. Idempotent. Returns the names applied this call."""
    applied: list[str] = []
    # First connection has no vector type yet (the extension may not exist).
    with psycopg.connect(cfg.dsn, row_factory=dict_row, autocommit=True, connect_timeout=10) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8").replace("{{EMBED_DIM}}", str(cfg.embedding.dim))
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
        _check_embedding_meta(conn, cfg)
    return applied


def _check_embedding_meta(conn: psycopg.Connection, cfg: Config) -> None:
    """Pin the embedding model into the DB; refuse to run against a different one."""
    want = {"embedding_model": cfg.embedding.model, "embedding_dim": str(cfg.embedding.dim)}
    have = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM store_meta")}
    for k, v in want.items():
        if k not in have:
            conn.execute("INSERT INTO store_meta (key, value) VALUES (%s, %s)", (k, v))
        elif have[k] != v:
            raise StoreError(
                f"database was built with {k}={have[k]!r} but config {cfg.name!r} has {v!r}; "
                "rebuild the corpus into a fresh database"
            )
