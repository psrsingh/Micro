"""Structured-output schemas for LLM stages. Validated with pydantic; stages fall back
explicitly (and log that they did) when validation fails."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issue: str
    act_names: list[str] = Field(default_factory=list)
    section_numbers: list[str] = Field(default_factory=list)
    case_names: list[str] = Field(default_factory=list)
    court: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    rewritten_query: str
    is_fact_pattern: bool = False
    # Set by the stage, never by the model: True when this is the raw-query fallback.
    fallback: bool = False


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class DecomposedClaims(BaseModel):
    claims: list[DraftClaim]


class Statement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    claim_ids: list[int]


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    abstain: bool
    statements: list[Statement] = Field(default_factory=list)
