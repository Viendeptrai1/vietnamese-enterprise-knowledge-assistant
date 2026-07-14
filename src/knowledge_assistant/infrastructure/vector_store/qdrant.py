import uuid
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata
from knowledge_assistant.domain.retrieval import RetrievedChunk
from knowledge_assistant.infrastructure.vector_store.base import VectorStore


class QdrantVectorStore(VectorStore):
    """Local Qdrant vector store implementation using persistent disk path or in-memory storage."""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        path: Path | str = ":memory:",
    ) -> None:
        self.collection_name = collection_name
        self.path = path
        if str(path) == ":memory:" or str(path) == "memory":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=str(path))

    def _ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Generate a deterministic UUID point ID from chunk_id for idempotent upserts."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_id: str,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        if not chunks:
            return 0

        dimension = len(vectors[0])
        self._ensure_collection(dimension)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(mode="json"),
                "ordinal": chunk.ordinal,
                "model_id": model_id,
            }
            points.append(
                PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=list(vector),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(chunks)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        model_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self.client.collection_exists(self.collection_name):
            return []

        query_filter = None
        if model_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="model_id",
                        match=MatchValue(value=model_id),
                    )
                ]
            )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
        )

        results: list[RetrievedChunk] = []
        for hit in hits:
            if hit.payload is None:
                continue
            if model_id is not None and hit.payload.get("model_id") != model_id:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=str(hit.payload["chunk_id"]),
                    document_id=str(hit.payload["document_id"]),
                    content=str(hit.payload["content"]),
                    metadata=DocumentMetadata.model_validate(hit.payload["metadata"]),
                    ordinal=int(hit.payload["ordinal"]),
                    score=float(hit.score),
                )
            )

        return results
