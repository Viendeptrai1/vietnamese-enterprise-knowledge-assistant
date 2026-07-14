from abc import ABC, abstractmethod
from typing import Sequence

from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.retrieval import RetrievedChunk


class VectorStore(ABC):
    """Contract for vector database operations."""

    @abstractmethod
    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_id: str,
    ) -> int:
        """Upsert document chunks and their embedding vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        model_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search top_k chunks most similar to query_vector, optionally filtered by model_id."""
