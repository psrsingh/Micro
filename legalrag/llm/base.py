"""The one interface every LLM call goes through.

Pipeline and evaluation code depend only on `LLMAdapter`; swapping generators is a
config change. Adapters report what they actually sent (`params_applied`) and what they
had to drop (`params_dropped`) so a result file can never claim determinism that the
backend did not provide.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Transport/backend failure. Stages catch this and record a stage failure."""


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str  # query_analysis | draft | decompose | final | judge | ...
    prompt: str
    system: str | None = None
    max_tokens: int | None = None
    json_output: bool = False
    prompt_id: str | None = None  # e.g. "draft@v1", from the prompt file name


class LLMResponse(BaseModel):
    text: str
    model_reported: str  # what the backend says it served
    stop_reason: str | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    params_applied: dict[str, Any] = Field(default_factory=dict)
    params_dropped: dict[str, str] = Field(default_factory=dict)  # name -> reason

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")


class LLMAdapter(ABC):
    provider: str

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Pinned identity for result files: provider, model, version/digest, params."""


def timed(fn):  # small helper so adapters measure latency the same way
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        out.latency_s = time.perf_counter() - t0
        return out

    return wrapper


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse a JSON value from model output, tolerating code fences and stray prose."""
    s = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in model output")


def parse_structured(text: str, schema: type[T]) -> tuple[T | None, str | None]:
    """Validate model output against a pydantic schema. Returns (value, error)."""
    try:
        return schema.model_validate(extract_json(text)), None
    except (ValueError, ValidationError) as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"
