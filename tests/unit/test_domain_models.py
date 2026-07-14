from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from knowledge_assistant.domain.answers import AnswerResponse
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata, NormalizedDocument
from knowledge_assistant.domain.retrieval import Citation, IndexSummary, RetrievedChunk


def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-123",
        source_path="docs/guide.pdf",
        source_type="pdf",
        title="Guide",
        page_number=4,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:abc123",
    )


def test_domain_models_round_trip_valid_values() -> None:
    document_metadata = metadata()
    document = NormalizedDocument(metadata=document_metadata, content="Nội dung chuẩn hóa")
    chunk = DocumentChunk(
        chunk_id="doc-123:chunk-0001",
        document_id=document_metadata.document_id,
        content="Nội dung một đoạn",
        metadata=document_metadata,
        ordinal=1,
    )
    citation = Citation(
        chunk_id=chunk.chunk_id,
        source_path=document_metadata.source_path,
        page_number=document_metadata.page_number,
        excerpt=chunk.content,
    )
    answer = AnswerResponse(
        answer="Câu trả lời",
        citations=[citation],
        retrieved_chunks=[chunk],
        latency_ms=12.5,
    )

    assert NormalizedDocument.model_validate(document.model_dump()) == document
    assert AnswerResponse.model_validate(answer.model_dump()) == answer
    assert answer.citations[0].chunk_id == chunk.chunk_id


def test_document_metadata_rejects_unsupported_source_type() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata(
            **{
                **metadata().model_dump(),
                "source_type": "html",
            }
        )


def test_document_chunk_requires_stable_chunk_id() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            document_id="doc-123",
            content="Nội dung một đoạn",
            metadata=metadata(),
            ordinal=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", ""),
        ("ordinal", -1),
    ],
)
def test_document_chunk_rejects_invalid_identity_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            **{
                "chunk_id": "doc-123:chunk-0001",
                "document_id": "doc-123",
                "content": "Nội dung một đoạn",
                "metadata": metadata(),
                "ordinal": 1,
                field: value,
            }
        )


def test_document_chunk_rejects_metadata_document_id_mismatch() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="doc-456:chunk-0001",
            document_id="doc-456",
            content="Nội dung một đoạn",
            metadata=metadata(),
            ordinal=1,
        )


def test_domain_models_and_nested_models_are_immutable() -> None:
    document_metadata = metadata()
    document = NormalizedDocument(metadata=document_metadata, content="Nội dung chuẩn hóa")
    citation = Citation(
        chunk_id="doc-123:chunk-0001",
        source_path=document_metadata.source_path,
        page_number=document_metadata.page_number,
        excerpt="Nội dung một đoạn",
    )
    chunk = DocumentChunk(
        chunk_id="doc-123:chunk-0001",
        document_id=document_metadata.document_id,
        content="Nội dung một đoạn",
        metadata=document_metadata,
        ordinal=1,
    )
    answer = AnswerResponse(
        answer="Câu trả lời",
        citations=[citation],
        retrieved_chunks=[chunk],
        latency_ms=12.5,
    )

    for model, field, value in [
        (document_metadata, "title", "Changed"),
        (document, "content", "Changed"),
        (citation, "excerpt", "Changed"),
        (chunk, "content", "Changed"),
        (answer, "answer", "Changed"),
    ]:
        with pytest.raises(ValidationError):
            setattr(model, field, value)

    with pytest.raises(ValidationError):
        document.metadata.title = "Changed"
    with pytest.raises(ValidationError):
        chunk.metadata.title = "Changed"
    with pytest.raises(ValidationError):
        answer.retrieved_chunks[0].content = "Changed"


def test_answer_response_collections_are_immutable() -> None:
    document_metadata = metadata()
    chunk = DocumentChunk(
        chunk_id="doc-123:chunk-0001",
        document_id=document_metadata.document_id,
        content="Nội dung một đoạn",
        metadata=document_metadata,
        ordinal=1,
    )
    citation = Citation(
        chunk_id=chunk.chunk_id,
        source_path=document_metadata.source_path,
        page_number=document_metadata.page_number,
        excerpt=chunk.content,
    )
    answer = AnswerResponse(
        answer="Câu trả lời",
        citations=[citation],
        retrieved_chunks=[chunk],
        latency_ms=12.5,
    )

    assert isinstance(answer.citations, tuple)
    assert isinstance(answer.retrieved_chunks, tuple)
    with pytest.raises(TypeError):
        answer.citations[0] = citation
    with pytest.raises((AttributeError, TypeError)):
        answer.retrieved_chunks.append(chunk)


def test_retrieval_domain_models() -> None:
    document_metadata = metadata()
    retrieved = RetrievedChunk(
        chunk_id="doc-123:chunk-0001",
        document_id=document_metadata.document_id,
        content="Nội dung một đoạn",
        metadata=document_metadata,
        ordinal=1,
        score=0.88,
    )
    summary = IndexSummary(total_documents=2, total_chunks=10, upserted_chunks=5)

    assert isinstance(retrieved, DocumentChunk)
    assert retrieved.score == 0.88
    assert summary.upserted_chunks == 5
    with pytest.raises(ValidationError):
        IndexSummary(total_documents=-1, total_chunks=10, upserted_chunks=5)
