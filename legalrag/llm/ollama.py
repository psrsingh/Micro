"""Local generator via an Ollama server (localhost only; no internet needed)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from legalrag.llm.base import LLMAdapter, LLMError, LLMRequest, LLMResponse
from legalrag.settings import LLMConfig

DEFAULT_URL = "http://127.0.0.1:11434"  # not "localhost": see DBConfig.dsn on IPv6 stalls


class OllamaAdapter(LLMAdapter):
    provider = "ollama"

    def __init__(self, cfg: LLMConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url or DEFAULT_URL, timeout=cfg.timeout_s, transport=transport
        )
        self._digest: str | None = None

    def _model_digest(self) -> str | None:
        """Pin the exact weights: Ollama tags move, digests do not."""
        if self._digest is None:
            try:
                r = self._client.get("/api/tags")
                r.raise_for_status()
                for m in r.json().get("models", []):
                    if m.get("name") == self.cfg.model or m.get("model") == self.cfg.model:
                        self._digest = m.get("digest")
                        break
            except httpx.HTTPError as e:
                raise LLMError(f"ollama: cannot reach server for model digest: {e}") from e
        return self._digest

    def complete(self, req: LLMRequest) -> LLMResponse:
        options: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "num_predict": req.max_tokens or self.cfg.max_tokens,
        }
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if req.json_output:
            body["format"] = "json"

        t0 = time.perf_counter()
        try:
            r = self._client.post("/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise LLMError(f"ollama request failed: {e}") from e
        latency = time.perf_counter() - t0

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model_reported=data.get("model", self.cfg.model),
            stop_reason=data.get("done_reason"),
            latency_s=latency,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            params_applied={**options, "format": body.get("format")},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.cfg.model,
            "digest": self._model_digest(),
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
        }
