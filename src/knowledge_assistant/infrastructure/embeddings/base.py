from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Contract for document and query embedding implementations."""

    model_id: str
    dimension: int

    @abstractmethod
    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        """Embed document chunks for indexing."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed one user query for retrieval."""
