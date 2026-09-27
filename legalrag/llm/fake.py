"""Deterministic test double so the plumbing runs with no model at all.

It reads the structured sections of the pipeline prompts (evidence lines, draft text,
claim lines) and produces plausible outputs. It exists to prove stage wiring and
logging; nothing measured with it says anything about model quality.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from legalrag.llm.base import LLMAdapter, LLMRequest, LLMResponse
from legalrag.settings import LLMConfig

_EVIDENCE_LINE = re.compile(r"^\[(E\d+)\][^\n]*? :: (.+)$", re.MULTILINE)
_CITE = re.compile(r"\[(E\d+)\]")
_CLAIM_LINE = re.compile(r"^(C\d+): (.+)$", re.MULTILINE)
_SECTION_REF = re.compile(r"\b[Ss]ection\s+(\d+[A-Z]?)")
_CASE_REF = re.compile(r"\b([A-Z][\w.]+(?: [A-Z][\w.]+)* v\.? [A-Z][\w.]+(?: [A-Z][\w.]+)*)")


_FAKE_STOP = frozenset(
    "a an and are as at be bound by can do does for from has have how in is it may much must of on or "
    "part shall that the to under was were what when which who with within without".split()
)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in _FAKE_STOP and len(w) > 2}


def _after(marker: str, prompt: str) -> str:
    _, _, rest = prompt.partition(marker)
    return rest.strip()


def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.DOTALL)
    return (m.group(1) if m else text).strip()


class FakeLLM(LLMAdapter):
    provider = "fake"

    def __init__(
        self,
        cfg: LLMConfig | None = None,
        responder: Callable[[LLMRequest], str] | None = None,
    ):
        self.cfg = cfg or LLMConfig()
        self._responder = responder
        self.calls: list[LLMRequest] = []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        text = self._responder(req) if self._responder else self._default(req)
        return LLMResponse(
            text=text,
            model_reported=self.cfg.model,
            stop_reason="end_turn",
            input_tokens=len(req.prompt.split()),
            output_tokens=len(text.split()),
            params_applied={"temperature": 0.0},
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": self.cfg.model, "temperature": 0.0, "seed": self.cfg.seed}

    # -- task heuristics -----------------------------------------------------------
    def _default(self, req: LLMRequest) -> str:
        return getattr(self, f"_task_{req.task}", self._task_unknown)(req)

    def _task_unknown(self, req: LLMRequest) -> str:
        return json.dumps({"error": f"fake llm has no handler for task {req.task!r}"})

    def _task_query_analysis(self, req: LLMRequest) -> str:
        q = _after("QUESTION:", req.prompt)
        return json.dumps(
            {
                "issue": q,
                "act_names": [],
                "section_numbers": sorted(set(_SECTION_REF.findall(q))),
                "case_names": _CASE_REF.findall(q),
                "court": None,
                "date_from": None,
                "date_to": None,
                "rewritten_query": q,
                "is_fact_pattern": False,
            }
        )

    def _task_draft(self, req: LLMRequest) -> str:
        # Honour the prompt's contract: passages that do not touch the question produce
        # no statements. "Touch" = shares a content stem (first 5 letters) with it.
        q = _stems(_after("QUESTION:", req.prompt))
        items = [(lbl, t) for lbl, t in _EVIDENCE_LINE.findall(req.prompt) if q & _stems(t)][:3]
        return "\n".join(f"{_first_sentence(text)} [{label}]" for label, text in items)

    def _task_decompose(self, req: LLMRequest) -> str:
        draft = _after("DRAFT:", req.prompt)
        claims = []
        for line in filter(None, (ln.strip() for ln in draft.splitlines())):
            cites = _CITE.findall(line)
            claims.append({"text": _CITE.sub("", line).strip(), "citations": cites})
        return json.dumps({"claims": claims})

    def _task_judge(self, req: LLMRequest) -> str:
        # Word-overlap verdict with a verbatim quote, so the LLM-judge plumbing (quote
        # grounding, offsets) is exercised offline. Says nothing about a real judge.
        claim = _after("CLAIM:", req.prompt).partition("PASSAGE:")[0].strip()
        passage = _after("PASSAGE:", req.prompt)
        c = _stems(claim)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", passage) if x.strip()]
        best = max(sentences, key=lambda x: len(c & _stems(x)), default="")
        overlap = len(c & _stems(best)) / len(c) if c else 0.0
        verdict = "supported" if overlap >= 0.6 else "partial" if overlap >= 0.35 else "unsupported"
        return json.dumps({"verdict": verdict, "quote": best if verdict != "unsupported" else "",
                           "reason": f"stem overlap {overlap:.2f}"})

    def _task_baseline(self, req: LLMRequest) -> str:
        # No knowledge, no documents: an honest fake abstains.
        return json.dumps({"abstain": True, "answer": "", "citations": [], "confidence": 0.0})

    def _task_rag_answer(self, req: LLMRequest) -> str:
        q = _stems(_after("QUESTION:", req.prompt))
        items = [(lbl, t) for lbl, t in _EVIDENCE_LINE.findall(req.prompt) if q & _stems(t)][:3]
        if not items:
            return json.dumps({"abstain": True, "answer": "", "citations": [], "confidence": 0.0})
        return json.dumps({"abstain": False, "answer": " ".join(_first_sentence(t) for _, t in items),
                           "citations": [], "confidence": 0.6})

    def _task_final(self, req: LLMRequest) -> str:
        claims = _CLAIM_LINE.findall(_after("CLAIMS:", req.prompt))
        if not claims:
            return json.dumps({"abstain": True, "statements": []})
        return json.dumps(
            {
                "abstain": False,
                "statements": [
                    {"text": _CITE.sub("", text).strip(), "claim_ids": [int(cid[1:])]}
                    for cid, text in claims
                ],
            }
        )
