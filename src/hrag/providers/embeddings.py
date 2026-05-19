"""EmbeddingProvider interface and concrete implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

from hrag.config import EmbeddingsConfig


class EmbeddingProvider(ABC):
    name: str = "abstract"

    def __init__(self, config: EmbeddingsConfig):
        self.config = config

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


def get_embedding_provider(config: EmbeddingsConfig) -> EmbeddingProvider:
    name = config.provider.lower().strip()
    if name in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformersProvider(config)
    if name == "openai":
        return OpenAIEmbeddingProvider(config)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required."
            ) from e
        self._model = SentenceTransformer(config.model)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    @property
    def dim(self) -> int:
        return int(self._dim)


def dimension_for_model(model_id: str) -> int | None:
    """Look up the embedding dimension of *model_id* in the curated suggestions.

    Returns the ``dim`` integer when the model appears in
    ``EmbeddingsConfig.suggested_models``, or ``None`` when it is not on the
    list (the provider will still load it; we just cannot cheaply determine
    the dimension without instantiating the model, which is expensive).

    Pure function — no model loading, no network calls.
    """
    from hrag.config import EmbeddingsConfig  # noqa: PLC0415

    defaults = EmbeddingsConfig().suggested_models
    for entry in defaults:
        if entry.get("model") == model_id:
            return entry.get("dim")
    return None


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, config: EmbeddingsConfig):
        super().__init__(config)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("The 'openai' package is required.") from e
        self._client = OpenAI(api_key=api_key)
        # text-embedding-3-small=1536, text-embedding-3-large=3072
        self._dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.config.model, input=list(texts))
        return [d.embedding for d in resp.data]

    @property
    def dim(self) -> int:
        return self._dim_map.get(self.config.model, self.config.dim)
