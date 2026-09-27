"""Every run writes results/<timestamp>-<config>.json plus a readable markdown summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legalrag.settings import PROJECT_ROOT, Config

RESULTS_DIR = PROJECT_ROOT / "results"


def write_results(cfg: Config, *, kind: str, corpus_sha: str, model_info: dict[str, Any],
                  records: list[dict[str, Any]], summary: dict[str, Any], notes: list[str],
                  markdown_body: str, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ts}-{cfg.name}"
    payload = {
        "kind": kind, "timestamp": ts, "config_name": cfg.name, "config_hash": cfg.config_hash,
        "config": cfg.reproducible_dump(), "corpus_sha": corpus_sha, "model_info": model_info,
        "notes": notes, "summary": summary, "records": records,
    }
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    header = (
        f"# {kind}: {cfg.name}\n\n"
        f"- timestamp: `{ts}`\n- config hash: `{cfg.config_hash}`\n- corpus sha: `{corpus_sha}`\n"
        f"- models: `{json.dumps(model_info, default=str)}`\n\n"
        + "".join(f"> {n}\n" for n in notes) + "\n"
    )
    md_path.write_text(header + markdown_body, encoding="utf-8")
    return json_path, md_path
