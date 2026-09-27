"""The one place a `VerifyConfig` becomes a `Verifier`."""

from __future__ import annotations

from legalrag.authority import AuthorityIndex
from legalrag.llm import LLMAdapter
from legalrag.settings import CHECK_NAMES, VerifyConfig
from legalrag.verify.authority import AuthorityCheck
from legalrag.verify.base import Check
from legalrag.verify.citation_metadata import CitationMetadataCheck
from legalrag.verify.claim_support import ClaimSupportCheck, LexicalSupportJudge, LLMSupportJudge, SupportJudge
from legalrag.verify.fake import build_checks as build_fake_checks
from legalrag.verify.fake import phase0_status
from legalrag.verify.rules import make_status_fn
from legalrag.verify.statute import StatuteCheck
from legalrag.verify.temporal import TemporalCheck
from legalrag.verify.verifier import Verifier


def build_support_judge(cfg: VerifyConfig | None, kind: str, llm: LLMAdapter | None) -> SupportJudge:
    if kind == "llm":
        if llm is None:
            raise ValueError("an llm support judge needs an LLM adapter")
        return LLMSupportJudge(llm, cfg.judge_prompt if cfg else "v1")
    pass_at, warn_at = (cfg.lexical_pass, cfg.lexical_warn) if cfg else (0.5, 0.3)
    return LexicalSupportJudge(pass_at, warn_at, cfg.lexical_can_verify if cfg else True)


def build_real_checks(cfg: VerifyConfig, index: AuthorityIndex, llm: LLMAdapter | None) -> list[Check]:
    unknown = set(cfg.enabled_checks) - set(CHECK_NAMES)
    if unknown:
        raise ValueError(f"unknown checks in config: {sorted(unknown)}")
    factories = {
        "authority": lambda: AuthorityCheck(index),
        "citation_metadata": lambda: CitationMetadataCheck(index),
        "statute": lambda: StatuteCheck(index),
        "claim_support": lambda: ClaimSupportCheck(build_support_judge(cfg, cfg.support_judge, llm)),
        "temporal": lambda: TemporalCheck(index, cfg.as_of),
    }
    return [factories[n]() for n in CHECK_NAMES if n in cfg.enabled_checks]


def build_verifier(cfg: VerifyConfig, index: AuthorityIndex | None = None,
                   llm: LLMAdapter | None = None) -> Verifier:
    if cfg.kind == "fake":
        return Verifier(build_fake_checks(cfg), phase0_status, cfg.min_verified_per_claim, cfg.include_flagged)
    if index is None:
        raise ValueError("verify.kind = real needs an AuthorityIndex")
    return Verifier(build_real_checks(cfg, index, llm), make_status_fn(cfg),
                    cfg.min_verified_per_claim, cfg.include_flagged)
