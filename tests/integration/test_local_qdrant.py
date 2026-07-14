from datetime import datetime, timezone
from pathlib import Path

from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata
from knowledge_assistant.infrastructure.vector_store.qdrant import QdrantVectorStore


def sample_chunk(chunk_id: str, content: str = "Nội dung kiểm thử", ordinal: int = 1) -> DocumentChunk:
    meta = DocumentMetadata(
        document_id="doc-1",
        source_path="docs/guide.md",
        source_type="markdown",
        title="Guide",
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:123",
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=meta.document_id,
        content=content,
        metadata=meta,
        ordinal=ordinal,
    )


def test_qdrant_vector_store_in_memory_upsert_and_search() -> None:
    store = QdrantVectorStore(collection_name="test_mem", path=":memory:")
    chunk1 = sample_chunk("doc-1:c1", "Qdrant vector store local testing", 1)
    chunk2 = sample_chunk("doc-1:c2", "RAG system using E5 embeddings", 2)

    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    upserted = store.upsert([chunk1, chunk2], vectors, model_id="e5-small")
    assert upserted == 2

    # Search with exact vector of chunk1
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2, model_id="e5-small")
    assert len(results) == 2
    assert results[0].chunk_id == chunk1.chunk_id
    assert results[0].score >= results[1].score


def test_qdrant_vector_store_idempotency_and_model_filtering(tmp_path: Path) -> None:
    store = QdrantVectorStore(collection_name="test_disk", path=tmp_path / "qdrant_data")
    chunk = sample_chunk("doc-1:c1", "Nội dung bất biến", 1)
    vector = [[0.5, 0.5]]

    store.upsert([chunk], vector, model_id="model-v1")
    results_v1 = store.search([0.5, 0.5], top_k=5, model_id="model-v1")
    assert len(results_v1) == 1

    # Re-upsert identical chunk with same chunk_id replaces point (count stays 1)
    store.upsert([chunk], vector, model_id="model-v1")
    results_v1_again = store.search([0.5, 0.5], top_k=5, model_id="model-v1")
    assert len(results_v1_again) == 1

    # Search with a different model_id must return empty list due to model filter
    results_v2 = store.search([0.5, 0.5], top_k=5, model_id="model-v2")
    assert results_v2 == []
