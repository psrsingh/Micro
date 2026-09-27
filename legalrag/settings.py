"""Config system: one pydantic model tree, one YAML file per experiment variant.

A run must be reproducible from the config file alone, so everything that changes
behaviour lives here and `config_hash` covers all of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"
PROJECT_ROOT = PACKAGE_DIR.parent

CHECK_NAMES = ("authority", "citation_metadata", "statute", "claim_support", "temporal")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DBConfig(_Strict):
    # Overridable with LEGALRAG_DSN so the password never has to live in a yaml file.
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 ::1 first, where Docker
    # is not listening, and each connection then hangs ~130 s before falling back.
    dsn: str = "postgresql://legalrag:legalrag@127.0.0.1:5432/legalrag"


class EmbeddingConfig(_Strict):
    kind: Literal["hash", "sentence_transformers"] = "hash"
    model: str = "hash-bow-v1"
    dim: int = 384
    device: str = "cpu"  # sentence_transformers only; the 4 GB GPU is better left to the generator
    batch_size: int = 16


class LLMConfig(_Strict):
    provider: Literal["fake", "ollama", "anthropic"] = "fake"
    model: str = "fake-v1"
    temperature: float = 0.0
    seed: int = 1234
    max_tokens: int = 1024
    base_url: str | None = None  # Ollama server; defaults to http://localhost:11434
    # Anthropic only: current Claude models (Opus 5, Sonnet 5, Fable) reject sampling
    # params, so temperature/seed are dropped for them. `effort` is sent instead.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"
    timeout_s: float = 120.0


class IngestConfig(_Strict):
    corpus_dir: str = "legalrag/tests/fixtures/sample_corpus"
    # Phase 0 counts whitespace words as tokens. Phase 1 switches to the embedding
    # model's own tokenizer.
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 40


class CorpusConfig(_Strict):
    """Real-corpus acquisition and extraction (legalrag.corpus). Environment-level settings: like
    `db` they are left out of `config_hash`; what they produce is identified by `corpus_sha`."""

    # Original files (never modified) and extracted text live here, outside the OneDrive-synced
    # project folder. Override with LEGALRAG_CORPUS_DIR.
    corpus_dir: str = "~/legalrag_corpus"
    manifest_path: str = "corpus/manifest.yaml"
    # Hosts documents may be downloaded from. Anything else is refused, whatever the manifest says.
    source_allowlist: list[str] = Field(default_factory=lambda: [
        "www.mha.gov.in", "indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"])
    download_min_interval_s: float = 1.0   # rate limit between requests to the same host
    download_timeout_s: float = 60.0
    max_download_mb: int = 40
    respect_robots_txt: bool = True
    user_agent: str = "legalrag-research/0.1 (academic prototype; not a crawler)"
    max_documents: int = 200               # processing limit for one run
    min_chars_per_page: int = 40           # below this a page is treated as having no text layer
    ocr_enabled: bool = False              # needs pytesseract + the Tesseract binary; see README
    ocr_language: str = "eng"
    # Statutes: a section longer than this many words is split at its sub-sections.
    statute_max_words: int = 220
    judgment_max_words: int = 220

    @property
    def corpus_path(self) -> Path:
        return Path(os.environ.get("LEGALRAG_CORPUS_DIR", self.corpus_dir)).expanduser()


class RetrievalConfig(_Strict):
    lexical_top_k: int = 100
    dense_top_k: int = 100
    fused_top_k: int = 50
    rrf_k: int = 60
    # None keeps the pre-real-corpus behaviour (no filter). The real-corpus config sets this so
    # `pending_review` / `invalid` documents can never be retrieved.
    allowed_verification_statuses: list[str] | None = None


class RerankConfig(_Strict):
    kind: Literal["lexical_fake", "cross_encoder"] = "lexical_fake"
    model: str = "lexical-overlap-v1"
    device: str = "cpu"  # cross_encoder only
    batch_size: int = 16
    keep: int = 10
    boost_section_match: float = 0.15
    boost_case_name_match: float = 0.15
    boost_holding_reasoning: float = 0.10
    penalty_arguments: float = 0.10


class VerifyConfig(_Strict):
    kind: Literal["fake", "real"] = "fake"
    enabled_checks: list[str] = Field(default_factory=lambda: list(CHECK_NAMES))
    # A claim survives if it has at least this many VERIFIED evidence items.
    min_verified_per_claim: int = 1
    # Synopsis: "generate final answers using only verified or explicitly flagged evidence".
    # When true, a claim with no VERIFIED evidence but at least one FLAGGED item survives and
    # the answer states the caveat. Off by default: strict verified-only.
    include_flagged: bool = False
    # Reference date for "is this provision still in force". None means today, which is not
    # reproducible; every experiment config should pin it.
    as_of: date | None = None
    # How claim_support judges a (claim, passage) pair. `lexical` is an offline word-overlap
    # heuristic; `llm` asks the configured generator and requires a verbatim supporting quote.
    support_judge: Literal["lexical", "llm"] = "lexical"
    lexical_pass: float = 0.5
    lexical_warn: float = 0.3
    # False: word overlap can never reach "supported" (it caps at "partial", i.e. FLAGGED at best). Overlap is
    # similarity, not entailment; real-corpus configs turn this off. Default keeps the old behaviour.
    lexical_can_verify: bool = True
    # judge prompt version for support_judge = llm. v2 also asks whether support is direct or inferred.
    judge_prompt: Literal["v1", "v2"] = "v1"
    # Status rules (kind == "real"). FAIL in one of these checks means REJECTED; a FAIL in any
    # other check, or a WARN anywhere, means FLAGGED. VERIFIED additionally needs a PASS in
    # every `require_pass` check that is enabled.
    reject_on_fail: list[str] = Field(
        default_factory=lambda: ["authority", "citation_metadata", "statute", "claim_support"])
    require_pass: list[str] = Field(default_factory=lambda: ["claim_support"])


class PipelineConfig(_Strict):
    max_claims: int = 12
    # verify.kind = real only. Abstain before drafting when the question names a case or
    # section the corpus cannot ground (see legalrag.verify.query_check).
    abstain_on_unavailable_authority: bool = True


class EvalConfig(_Strict):
    benchmark_path: str = "legalrag/eval/data/benchmark_synthetic.jsonl"
    probes_path: str = "legalrag/eval/data/authority_probes_synthetic.jsonl"
    # Where `eval.compare` validates the benchmark's gold passages: 'files' reads `ingest.corpus_dir`
    # (the synthetic front-matter format, via CorpusView) as it always has; 'db' reads the live
    # database instead, for a benchmark written against a real, DB-ingested corpus. The AVA probes
    # are unaffected by this and always use the file-based synthetic fixtures (they are a fixed
    # regression suite, not a held-out estimate over whatever corpus is configured).
    corpus_source: Literal["files", "db"] = "files"
    arms: list[Literal["baseline", "standard_rag", "verified_rag"]] = Field(
        default_factory=lambda: ["baseline", "standard_rag", "verified_rag"])
    recall_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    # An incorrect answer whose confidence is at least this is a "high-confidence error".
    high_confidence: float = 0.8
    # Judge used to audit claims against evidence when scoring. Kept separate from
    # verify.support_judge so the system is not graded by its own instrument by default.
    audit_judge: Literal["lexical", "llm"] = "lexical"


class Config(_Strict):
    name: str
    seed: int = 1234
    db: DBConfig = Field(default_factory=DBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def reproducible_dump(self) -> dict:
        """Everything that affects results. `db` is environment (host, credentials) and `corpus`
        is acquisition plumbing (paths, rate limits), so both are excluded from the hash and never
        written into result files. Existing runs keep their config_hash: a field added later is
        left out of the hash while it still equals its pre-existing (old-default) meaning."""
        exclude: dict = {"db": True, "corpus": True}
        if self.retrieval.allowed_verification_statuses is None:
            exclude["retrieval"] = {"allowed_verification_statuses"}
        v_default = {k for k, v in (("lexical_can_verify", True), ("judge_prompt", "v1"))
                     if getattr(self.verify, k) == v}
        if v_default:
            exclude["verify"] = v_default
        if self.eval.corpus_source == "files":
            exclude["eval"] = {"corpus_source"}
        return self.model_dump(mode="json", exclude=exclude)

    def canonical_json(self) -> str:
        return json.dumps(self.reproducible_dump(), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    @property
    def dsn(self) -> str:
        return os.environ.get("LEGALRAG_DSN", self.db.dsn)

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> Config:
    """Load a config by file path, or by bare name from legalrag/config/."""
    p = Path(path)
    if not p.exists():
        candidate = CONFIG_DIR / (p.name if p.suffix else f"{p.name}.yaml")
        if not candidate.exists():
            raise FileNotFoundError(f"config not found: {path} (also tried {candidate})")
        p = candidate
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
