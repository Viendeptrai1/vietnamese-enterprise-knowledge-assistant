from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from knowledge_assistant.domain.answers import AnswerResponse
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata, NormalizedDocument
from knowledge_assistant.domain.retrieval import Citation


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


def test_domain_models_are_immutable() -> None:
    document_metadata = metadata()

    with pytest.raises(ValidationError):
        document_metadata.title = "Changed"
