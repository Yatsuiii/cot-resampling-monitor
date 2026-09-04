"""Real sentence-embedding `Embedder` for the Kaggle run.

Kept in its own module so importing `mats` never pulls in sentence-transformers;
the class is only constructed on the GPU runner.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim))
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
