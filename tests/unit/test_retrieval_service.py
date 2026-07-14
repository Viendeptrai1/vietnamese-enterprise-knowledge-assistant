from datetime import datetime, timezone
from typing import Sequence

import pytest

from knowledge_assistant.application.ingestion.chunking import Chunker
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata, NormalizedDocument
from knowledge_assistant.domain.retrieval import IndexSummary, RetrievedChunk
from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider
from knowledge_assistant.infrastructure.vector_store.base import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_id: str = "fake-e5", dimension: int = 4) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.embed_documents_call_count = 0
        self.embed_query_call_count = 0

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        self.embed_documents_call_count += 1
        return [[0.1] * self.dimension for _ in chunks]

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_call_count += 1
        return [0.1] * self.dimension


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.storage: dict[str, tuple[DocumentChunk, list[float], str]] = {}

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_id: str,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.storage[chunk.chunk_id] = (chunk, list(vector), model_id)
        return len(chunks)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        model_id: str | None = None,
    ) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for idx, (chunk, _, stored_model_id) in enumerate(self.storage.values()):
            if model_id is not None and stored_model_id != model_id:
                continue
            # Generate descending scores for testing
            score = 0.9 - (idx * 0.1)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    ordinal=chunk.ordinal,
                    score=score,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def sample_document(doc_id: str = "doc-1", content: str = "Đoạn 1. Đoạn 2.") -> NormalizedDocument:
    meta = DocumentMetadata(
        document_id=doc_id,
        source_path=f"docs/{doc_id}.md",
        source_type="markdown",
        title=f"Title {doc_id}",
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:abc",
    )
    return NormalizedDocument(metadata=meta, content=content)


def test_retrieval_service_index_calls_embedding_provider_once_per_batch() -> None:
    chunker = Chunker(max_characters=1000, overlap_characters=100)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    service = RetrievalService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    doc1 = sample_document("doc-1", "Đoạn 1 văn bản.\n\nĐoạn 2 văn bản.")
    doc2 = sample_document("doc-2", "Đoạn 3 văn bản.\n\nĐoạn 4 văn bản.")

    summary: IndexSummary = service.index([doc1, doc2])

    assert summary.total_documents == 2
    assert summary.total_chunks > 0
    assert summary.upserted_chunks == summary.total_chunks
    assert embedding_provider.embed_documents_call_count == 1


def test_retrieval_service_search_returns_descending_scores() -> None:
    chunker = Chunker(max_characters=1000, overlap_characters=100)
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    service = RetrievalService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    doc = sample_document("doc-1", "Đoạn A.\n\nĐoạn B.\n\nĐoạn C.")
    service.index([doc])

    results: list[RetrievedChunk] = service.search("câu hỏi test", top_k=2)

    assert len(results) <= 2
    assert embedding_provider.embed_query_call_count == 1
    if len(results) > 1:
        assert results[0].score >= results[1].score


def test_retrieval_service_idempotency_and_model_filtering() -> None:
    chunker = Chunker(max_characters=1000, overlap_characters=100)
    provider_v1 = FakeEmbeddingProvider(model_id="model-v1")
    vector_store = FakeVectorStore()
    service_v1 = RetrievalService(
        chunker=chunker,
        embedding_provider=provider_v1,
        vector_store=vector_store,
    )

    doc = sample_document("doc-1", "Nội dung bất biến.")
    summary1 = service_v1.index([doc])
    assert len(vector_store.storage) == summary1.total_chunks

    # Re-indexing the exact same chunks with the exact same model should replace identical point IDs in storage
    summary2 = service_v1.index([doc])
    assert len(vector_store.storage) == summary1.total_chunks
    assert summary2.upserted_chunks == summary1.total_chunks

    # Now search with a different active provider (model-v2)
    provider_v2 = FakeEmbeddingProvider(model_id="model-v2")
    service_v2 = RetrievalService(
        chunker=chunker,
        embedding_provider=provider_v2,
        vector_store=vector_store,
    )
    results = service_v2.search("test query", top_k=5)
    # Since all stored chunks have model-v1 and active provider is model-v2, results must be empty
    assert results == []
