"""Embedding interface: a deterministic hashing embedder for offline plumbing runs and a
sentence-transformers embedder for real ones, both behind `Embedder`."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from legalrag.settings import EmbeddingConfig

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    model_id: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class HashEmbedder:
    """Feature-hashed bag of words. Offline, deterministic, no download. Plumbing only:
    it captures lexical overlap, not meaning."""

    def __init__(self, dim: int = 384, model_id: str = "hash-bow-v1"):
        self.dim = dim
        self.model_id = model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (optional dependency:
    `pip install -e ".[ml]"`). Vectors are L2-normalised so cosine distance in pgvector
    ranks the same as dot product. The configured `dim` must match the model: the DB column
    width is fixed from it, so a mismatch fails here instead of corrupting the index."""

    def __init__(self, cfg: EmbeddingConfig, model=None):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise RuntimeError('sentence-transformers is not installed: pip install -e ".[ml]"') from e
            model = SentenceTransformer(cfg.model, device=cfg.device)
        self._model, self._batch = model, cfg.batch_size
        self.model_id, self.dim = cfg.model, cfg.dim
        actual = model.get_sentence_embedding_dimension()
        if actual != cfg.dim:
            raise ValueError(f"embedding model {cfg.model!r} produces {actual}-d vectors but config says dim={cfg.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        out = self._model.encode(texts, batch_size=self._batch, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32).reshape(len(texts), self.dim)


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.kind == "hash":
        return HashEmbedder(cfg.dim, cfg.model)
    if cfg.kind == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"unknown embedding kind {cfg.kind!r}")
