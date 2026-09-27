"""Stage 5: split the draft into atomic claims, each carrying the evidence it cited."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from legalrag.evidence import Claim, Evidence
from legalrag.llm import LLMAdapter, LLMRequest, parse_structured
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.pipeline.schemas import DecomposedClaims

_CITE = re.compile(r"\[(E\d+)\]")


@dataclass
class DecomposeResult:
    claims: list[Claim]
    fallback_reason: str | None


def decompose(llm: LLMAdapter, draft: str, labelled: dict[str, Evidence],
              max_claims: int) -> DecomposeResult:
    pid = PROMPT_IDS["decompose"]
    resp = llm.complete(LLMRequest(task="decompose", prompt=render(pid, draft=draft),
                                   json_output=True, prompt_id=pid))
    parsed, err = parse_structured(resp.text, DecomposedClaims)
    if parsed is not None:
        raw = [(c.text, c.citations) for c in parsed.claims]
        reason = None
    else:
        # Line-based fallback keeps the pipeline honest instead of dropping the draft.
        raw = [(_CITE.sub("", ln).strip(), _CITE.findall(ln)) for ln in draft.splitlines() if ln.strip()]
        reason = f"decomposition parse failed: {err}"

    claims: list[Claim] = []
    for text, cites in raw[:max_claims]:
        resolved, unresolved = [], []
        for label in dict.fromkeys(cites):  # dedupe, keep order
            if label in labelled:
                # Checks are per (claim, evidence) pair: the same chunk cited by two claims
                # must not share one mutable Evidence object.
                resolved.append(copy.deepcopy(labelled[label]))
            else:
                unresolved.append(label)  # a citation to nothing we retrieved: keep the record
        claims.append(Claim(idx=len(claims) + 1, text=text, evidence=resolved,
                            unresolved_citations=unresolved))
    return DecomposeResult(claims, reason)
