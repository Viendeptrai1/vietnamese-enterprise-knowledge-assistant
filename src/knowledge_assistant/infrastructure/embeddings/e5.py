from collections.abc import Callable, Sequence
from typing import Protocol

from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider


class SentenceTransformer(Protocol):
    def encode(self, texts: Sequence[str], *, batch_size: int) -> object: ...


class E5EmbeddingProvider(EmbeddingProvider):
    """Lazy adapter for the multilingual E5 sentence-transformer model."""

    model_id = "intfloat/multilingual-e5-small"
    dimension = 384

    def __init__(
        self,
        *,
        model_id: str = model_id,
        batch_size: int = 32,
        model_loader: Callable[[str], SentenceTransformer] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_id = model_id
        self._batch_size = batch_size
        self._model_loader = model_loader or self._load_model
        self._model: SentenceTransformer | None = None

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {chunk}" for chunk in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self._encode([f"query: {query}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._get_model().encode(texts, batch_size=self._batch_size)
        values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [[float(value) for value in vector] for vector in values]

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._model_loader(self.model_id)
        return self._model

    @staticmethod
    def _load_model(model_id: str) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer as SentenceTransformerModel

        return SentenceTransformerModel(model_id)
