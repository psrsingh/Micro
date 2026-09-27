"""API-backed generator using the official Anthropic SDK.

Determinism caveat, enforced here rather than hidden: the current Claude models
(Opus 5 / 4.8 / 4.7, Sonnet 5, Fable, Mythos) reject `temperature`, `top_p` and `top_k`
with a 400, and there is no seed parameter. For those models the adapter omits the
sampling params and says so in `params_dropped`, and result files record that the
arm was not temperature-0. Models that still accept sampling (Haiku 4.5, Sonnet 4.6,
Opus 4.6 and earlier) receive temperature as configured.

Model IDs are aliases with no date pin, so the exact served model string returned by
the API is logged on every response.
"""

from __future__ import annotations

import re
import time
from typing import Any

from legalrag.llm.base import LLMAdapter, LLMError, LLMRequest, LLMResponse
from legalrag.settings import LLMConfig

# Models that still accept temperature. Anything unrecognised is treated as NOT
# accepting it, which is the safe direction (a dropped param beats a 400 mid-run).
_SAMPLING_OK = re.compile(
    r"^claude-(haiku-4-5|sonnet-4-6|opus-4-6|sonnet-4-5|opus-4-5|opus-4-1|opus-4-0|sonnet-4-0|3-)"
)
# `output_config.effort` errors on Haiku 4.5 / Sonnet 4.5 and older.
_EFFORT_UNSUPPORTED = re.compile(r"^claude-(haiku-4-5|sonnet-4-5|opus-4-1|opus-4-0|sonnet-4-0|3-)")


def accepts_sampling(model: str) -> bool:
    return bool(_SAMPLING_OK.match(model))


def supports_effort(model: str) -> bool:
    return not _EFFORT_UNSUPPORTED.match(model)


class AnthropicAdapter(LLMAdapter):
    provider = "anthropic"

    def __init__(self, cfg: LLMConfig, client: Any | None = None):
        self.cfg = cfg
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError("anthropic SDK not installed: pip install -e '.[api]'") from e
            # Credentials resolve from ANTHROPIC_API_KEY or an `ant auth login` profile.
            client = anthropic.Anthropic(timeout=cfg.timeout_s)
        self._client = client

    def complete(self, req: LLMRequest) -> LLMResponse:
        import anthropic  # local import keeps the optional dependency optional

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": req.max_tokens or self.cfg.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            kwargs["system"] = req.system

        applied: dict[str, Any] = {}
        dropped: dict[str, str] = {}
        if accepts_sampling(self.cfg.model):
            kwargs["temperature"] = self.cfg.temperature
            applied["temperature"] = self.cfg.temperature
        else:
            dropped["temperature"] = "model rejects sampling params"
        dropped["seed"] = "Anthropic API has no seed parameter"
        if self.cfg.effort and supports_effort(self.cfg.model):
            kwargs["output_config"] = {"effort": self.cfg.effort}
            applied["effort"] = self.cfg.effort
        # No prefill on current models, so JSON is requested through the prompt and
        # validated by the caller against a pydantic schema.

        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            raise LLMError(f"anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic connection error: {e}") from e
        latency = time.perf_counter() - t0

        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(
            text=text,
            model_reported=msg.model,
            stop_reason=msg.stop_reason,
            latency_s=latency,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            request_id=getattr(msg, "_request_id", None),
            params_applied=applied,
            params_dropped=dropped,
        )

    def describe(self) -> dict[str, Any]:
        sampling = accepts_sampling(self.cfg.model)
        return {
            "provider": self.provider,
            "model": self.cfg.model,
            "temperature": self.cfg.temperature if sampling else None,
            "deterministic_sampling": sampling,
            "effort": self.cfg.effort if supports_effort(self.cfg.model) else None,
            "note": "" if sampling else "temperature not sent: model rejects sampling params",
        }
