"""Sentence embeddings for the semantic-dedup filter.

The resampling method only keeps a regenerated sentence if it is *semantically
different* from the original (otherwise "resampling sentence i" is a no-op and its
importance is spuriously zero). That check needs an embedder.

`Embedder` is a one-method protocol so the GPU model (sentence-transformers on
Kaggle) and the deterministic test double share an interface. Vectors are
L2-normalised, so a dot product is cosine similarity.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) array of L2-normalised row vectors."""
        ...


class HashEmbedder:
    """Deterministic bag-of-tokens hashing embedder. No model, no network.

    Good enough for tests and the dummy smoke run: near-identical sentences land
    close, unrelated ones land far, and it is perfectly reproducible.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = [self._encode_one(text) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim))

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[digest[0] % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of `a` and rows of `b`."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    return a @ b.T
