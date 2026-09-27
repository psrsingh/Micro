"""Prompt loader. Prompts are versioned text files: <name>.v<N>.txt, never inline."""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


@cache
def _read(prompt_id: str) -> str:
    name, _, version = prompt_id.partition("@")
    path = PROMPT_DIR / f"{name}.{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {prompt_id} ({path})")
    return path.read_text(encoding="utf-8")


def render(prompt_id: str, **variables: str) -> str:
    """Fill `{{var}}` placeholders. Every placeholder must be supplied and vice versa;
    JSON braces in prompts are left alone."""
    text = _read(prompt_id)
    for k, v in variables.items():
        token = "{{" + k + "}}"
        if token not in text:
            raise KeyError(f"{prompt_id}: no placeholder {token}")
        text = text.replace(token, v)
    if "{{" in text:
        raise KeyError(f"{prompt_id}: unfilled placeholder remains")
    return text


PROMPT_IDS = {
    "query_analysis": "query_analysis@v1",
    "draft": "draft@v1",
    "decompose": "decompose@v1",
    "final": "final@v1",
    "judge": "judge@v1",
    "baseline": "baseline@v1",
    "rag_answer": "rag_answer@v1",
}
