from dataclasses import dataclass
from pathlib import Path

from knowledge_assistant.domain.documents import NormalizedDocument
from knowledge_assistant.infrastructure.parsers.base import DocumentParseError
from knowledge_assistant.infrastructure.parsers.dispatcher import ParserDispatcher


@dataclass(frozen=True)
class IngestionError:
    path: Path
    message: str


class IngestionService:
    def __init__(self, dispatcher: ParserDispatcher | None = None) -> None:
        self._dispatcher = dispatcher or ParserDispatcher()
        self.errors: list[IngestionError] = []

    def ingest(self, root: Path) -> list[NormalizedDocument]:
        document_root = root.resolve()
        if not document_root.is_dir():
            raise ValueError(f"Document root does not exist or is not a directory: {root}")

        self.errors = []
        documents: list[NormalizedDocument] = []
        for path in sorted(document_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                resolved_path = path.resolve()
                relative_path = resolved_path.relative_to(document_root)
                parser = self._dispatcher.for_path(resolved_path)
                documents.extend(
                    self._with_relative_source_path(document, relative_path) for document in parser.parse(resolved_path)
                )
            except (DocumentParseError, ValueError) as error:
                self.errors.append(IngestionError(path=path, message=str(error)))
        return documents

    @staticmethod
    def _with_relative_source_path(document: NormalizedDocument, relative_path: Path) -> NormalizedDocument:
        metadata = document.metadata.model_copy(update={"source_path": relative_path.as_posix()})
        return document.model_copy(update={"metadata": metadata})
