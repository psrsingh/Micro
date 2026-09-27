from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from legalrag.settings import Config, load_config


@pytest.fixture(scope="session")
def base_cfg() -> Config:
    return load_config("phase0")


@pytest.fixture(scope="session")
def test_db_cfg(base_cfg: Config):
    """A throwaway database built from the sample corpus. Skips (loudly, with the reason)
    if Postgres is unreachable; tests marked `db` never pass silently without a database."""
    from urllib.parse import urlsplit, urlunsplit

    from legalrag.ingest.build import build

    admin_dsn = os.environ.get("LEGALRAG_DSN") or base_cfg.db.dsn
    parts = urlsplit(admin_dsn)
    name = f"legalrag_test_{uuid.uuid4().hex[:8]}"
    test_dsn = urlunsplit(parts._replace(path=f"/{name}"))
    try:
        admin = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({e}); start it with `docker compose up -d db`")
    admin.execute(f'CREATE DATABASE "{name}"')
    old = os.environ.get("LEGALRAG_DSN")
    os.environ["LEGALRAG_DSN"] = test_dsn
    try:
        build(base_cfg)
        yield base_cfg
    finally:
        if old is None:
            os.environ.pop("LEGALRAG_DSN", None)
        else:
            os.environ["LEGALRAG_DSN"] = old
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()
