from datetime import datetime, timezone

from knowledge_assistant.domain.documents import DocumentMetadata, NormalizedDocument


def document(content: str) -> NormalizedDocument:
    metadata = DocumentMetadata(
        document_id="doc-123",
        source_path="handbook/policy.md",
        source_type="markdown",
        title="Company policy",
        page_number=7,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:content",
    )
    return NormalizedDocument(metadata=metadata, content=content)


def test_chunker_preserves_identity_and_source_metadata() -> None:
    from knowledge_assistant.application.ingestion.chunking import Chunker

    chunks = Chunker(max_characters=40, overlap_characters=8).chunk(
        document("First paragraph contains policy.\n\nSecond paragraph contains details.")
    )

    assert [chunk.document_id for chunk in chunks] == ["doc-123", "doc-123"]
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert all(chunk.metadata.source_path == "handbook/policy.md" for chunk in chunks)
    assert all(chunk.metadata.page_number == 7 for chunk in chunks)


def test_chunker_is_deterministic_and_respects_maximum_characters() -> None:
    from knowledge_assistant.application.ingestion.chunking import Chunker

    source = document(
        "This is a deliberately oversized sentence that must be split safely. "
        "Another sentence follows to make a repeatable sequence."
    )
    chunker = Chunker(max_characters=35, overlap_characters=5)

    first_run = chunker.chunk(source)
    second_run = chunker.chunk(source)

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert [chunk.content for chunk in first_run] == [chunk.content for chunk in second_run]
    assert all(len(chunk.content) <= 35 for chunk in first_run)
