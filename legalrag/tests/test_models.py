"""Real-model adapters, exercised with injected stand-in models: no download, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from legalrag.embed import SentenceTransformerEmbedder, build_embedder
from legalrag.evidence import Evidence
from legalrag.pipeline.rerank import CrossEncoderScorer, boosts, rerank
from legalrag.pipeline.schemas import QueryAnalysis
from legalrag.settings import EmbeddingConfig, RerankConfig


class StubEncoder:
    def __init__(self, dim: int):
        self.dim, self.calls = dim, []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.calls.append((len(texts), batch_size, normalize_embeddings))
        return np.tile(np.arange(1, self.dim + 1, dtype=np.float64), (len(texts), 1))  # float64 on purpose


class StubCrossEncoder:
    def __init__(self, logits: list[float]):
        self.logits, self.pairs = logits, None

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs = pairs
        return self.logits[: len(pairs)]


def test_sentence_transformer_embedder_shape_dtype_and_normalisation_request():
    stub = StubEncoder(384)
    emb = SentenceTransformerEmbedder(EmbeddingConfig(kind="sentence_transformers", model="m", dim=384, batch_size=8), model=stub)
    out = emb.embed(["a", "b", "c"])
    assert out.shape == (3, 384) and out.dtype == np.float32
    assert stub.calls == [(3, 8, True)]  # the model is asked to L2-normalise; we do not re-normalise silently
    assert emb.model_id == "m" and emb.dim == 384


def test_embedder_refuses_a_model_whose_width_disagrees_with_the_database_column():
    with pytest.raises(ValueError, match="768-d vectors but config says dim=384"):
        SentenceTransformerEmbedder(EmbeddingConfig(kind="sentence_transformers", dim=384), model=StubEncoder(768))


def test_build_embedder_dispatch_without_the_optional_dependency_is_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_st(name, *a, **kw):
        if name == "sentence_transformers":
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_st)
    with pytest.raises(RuntimeError, match=r"pip install -e"):
        build_embedder(EmbeddingConfig(kind="sentence_transformers"))
    assert build_embedder(EmbeddingConfig()).dim == 384  # the offline embedder needs nothing


def test_cross_encoder_logits_are_squashed_to_the_boost_scale():
    stub = StubCrossEncoder([-10.0, 0.0, 10.0])
    scorer = CrossEncoderScorer(RerankConfig(kind="cross_encoder", model="m"), model=stub)
    scores = scorer.score("q", ["a", "b", "c"])
    assert stub.pairs == [("q", "a"), ("q", "b"), ("q", "c")]
    assert scores[0] < 0.001 and scores[1] == 0.5 and scores[2] > 0.999
    assert scorer.score("q", []) == []


def test_cross_encoder_plugs_into_rerank_with_the_same_boosts():
    stub = StubCrossEncoder([0.0, 0.0])  # equal base scores: only the metadata boosts differ
    scorer = CrossEncoderScorer(RerankConfig(kind="cross_encoder"), model=stub)
    holding = Evidence(1, 1, "x", (0, 1), section_label="holding")
    args = Evidence(2, 2, "y", (0, 1), section_label="arguments")
    qa = QueryAnalysis(issue="i", rewritten_query="q")
    ranked, kept = rerank([args, holding], qa, scorer, RerankConfig(keep=1))
    assert [e.chunk_id for e in ranked] == [1, 2] and [e.chunk_id for e in kept] == [1]
    assert "holding_or_reasoning" in boosts(holding, qa, RerankConfig())


def test_unknown_kinds_are_rejected_by_the_config_schema_before_reaching_a_factory():
    with pytest.raises(ValueError):
        EmbeddingConfig(kind="word2vec")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RerankConfig(kind="bm25")  # type: ignore[arg-type]
