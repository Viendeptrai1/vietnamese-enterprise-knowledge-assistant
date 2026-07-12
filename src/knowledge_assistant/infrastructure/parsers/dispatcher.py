from pathlib import Path

from knowledge_assistant.infrastructure.parsers.base import (
    DocumentParser,
    UnsupportedDocumentTypeError,
)
from knowledge_assistant.infrastructure.parsers.docx import DocxParser
from knowledge_assistant.infrastructure.parsers.markdown import MarkdownParser
from knowledge_assistant.infrastructure.parsers.pdf import PdfParser


class ParserDispatcher:
    def __init__(self, parsers: tuple[DocumentParser, ...] | None = None) -> None:
        self._parsers = parsers or (MarkdownParser(), PdfParser(), DocxParser())

    def for_path(self, path: Path) -> DocumentParser:
        suffix = path.suffix.lower()
        for parser in self._parsers:
            if suffix in parser.supported_suffixes:
                return parser
        raise UnsupportedDocumentTypeError(path, f"Unsupported document type: {suffix or '<none>'}")
