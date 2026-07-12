from pathlib import Path

from knowledge_assistant.application.ingestion.service import IngestionService

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def test_ingestion_sorts_documents_and_reports_malformed_files(tmp_path: Path) -> None:
    (tmp_path / "z-malformed.pdf").write_bytes(b"not a PDF")
    (tmp_path / "y-malformed.docx").write_bytes(b"not a DOCX")
    (tmp_path / "a-guide.md").write_text((FIXTURES / "guide.md").read_text())
    (tmp_path / "b-guide.docx").write_bytes((FIXTURES / "guide.docx").read_bytes())

    service = IngestionService()

    documents = service.ingest(tmp_path)

    assert [document.metadata.source_path for document in documents] == [
        "a-guide.md",
        "b-guide.docx",
    ]
    assert [error.path.name for error in service.errors] == ["y-malformed.docx", "z-malformed.pdf"]
    assert all(error.path.name in error.message for error in service.errors)


def test_ingestion_does_not_traverse_outside_the_document_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "guide.md").write_text("# Nested\n\nDocument")

    documents = IngestionService().ingest(tmp_path)

    assert [document.metadata.source_path for document in documents] == ["nested/guide.md"]
