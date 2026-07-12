from hashlib import sha256
from pathlib import Path

import pytest
from knowledge_assistant.infrastructure.parsers.dispatcher import (
    ParserDispatcher,
    UnsupportedDocumentTypeError,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def test_markdown_parser_produces_one_normalized_document() -> None:
    document = ParserDispatcher().for_path(FIXTURES / "guide.md").parse(FIXTURES / "guide.md")[0]

    assert document.metadata.source_type == "markdown"
    assert document.metadata.title == "Hướng dẫn nội bộ"
    assert document.metadata.page_number is None
    assert (
        document.content
        == "# Hướng dẫn nội bộ\n\nĐây là tài liệu Markdown dùng cho kiểm thử.\n\nNội dung được chuẩn hóa cục bộ."
    )
    assert document.metadata.content_hash == f"sha256:{sha256(document.content.encode()).hexdigest()}"
    assert document.metadata.modified_at.tzinfo is not None


def test_pdf_parser_produces_one_normalized_document_per_page() -> None:
    documents = ParserDispatcher().for_path(FIXTURES / "guide.pdf").parse(FIXTURES / "guide.pdf")

    assert [document.metadata.page_number for document in documents] == [1, 2]
    assert [document.content for document in documents] == [
        "Trang mot cua huong dan PDF.",
        "Trang hai cua huong dan PDF.",
    ]
    assert all(document.metadata.source_type == "pdf" for document in documents)


def test_docx_parser_extracts_nonempty_paragraphs_and_heading() -> None:
    document = ParserDispatcher().for_path(FIXTURES / "guide.docx").parse(FIXTURES / "guide.docx")[0]

    assert document.metadata.source_type == "docx"
    assert document.metadata.title == "Huong dan DOCX"
    assert document.content == "Huong dan DOCX\n\nDoan van DOCX dau tien.\n\nDoan van DOCX thu hai."


def test_dispatcher_rejects_unsupported_extensions() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document type"):
        ParserDispatcher().for_path(FIXTURES / "guide.txt")
