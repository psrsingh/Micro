"""Stage 1: structured query analysis with an explicit raw-query fallback."""

from __future__ import annotations

from dataclasses import dataclass

from legalrag.llm import LLMAdapter, LLMRequest, parse_structured
from legalrag.pipeline.prompt_loader import PROMPT_IDS, render
from legalrag.pipeline.schemas import QueryAnalysis


@dataclass
class QueryAnalysisResult:
    analysis: QueryAnalysis
    fallback_reason: str | None  # None means the model output validated


def analyze_query(llm: LLMAdapter, query: str) -> QueryAnalysisResult:
    pid = PROMPT_IDS["query_analysis"]
    resp = llm.complete(LLMRequest(task="query_analysis", prompt=render(pid, query=query),
                                   json_output=True, prompt_id=pid, max_tokens=512))
    parsed, err = parse_structured(resp.text, QueryAnalysis)
    if parsed is None or not parsed.rewritten_query.strip():
        reason = err or "empty rewritten_query"
        return QueryAnalysisResult(
            QueryAnalysis(issue=query, rewritten_query=query, fallback=True), reason)
    return QueryAnalysisResult(parsed, None)
