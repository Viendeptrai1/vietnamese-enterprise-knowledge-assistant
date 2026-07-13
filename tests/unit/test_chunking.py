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
    assert all(chunk.metadata.content_hash == "sha256:content" for chunk in chunks)


def test_chunker_packs_paragraphs_until_the_configured_limit() -> None:
    from knowledge_assistant.application.ingestion.chunking import Chunker

    chunks = Chunker(max_characters=30, overlap_characters=0).chunk(document("one two\n\nthree four\n\nfive six"))

    assert [chunk.content for chunk in chunks] == ["one two\n\nthree four\n\nfive six"]


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


def test_chunker_overlapping_character_fallback_preserves_suffix_and_prefix() -> None:
    from knowledge_assistant.application.ingestion.chunking import Chunker

    chunks = Chunker(max_characters=10, overlap_characters=3).chunk(document("abcdefghijklmnopqrst"))

    assert [chunk.content for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrst"]
    assert chunks[0].content[-3:] == chunks[1].content[:3]
    assert chunks[1].content[-3:] == chunks[2].content[:3]
    assert all(len(chunk.content) <= 10 for chunk in chunks)
