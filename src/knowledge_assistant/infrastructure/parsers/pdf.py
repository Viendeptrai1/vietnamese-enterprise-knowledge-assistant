from pathlib import Path

import fitz

from knowledge_assistant.domain.documents import NormalizedDocument
from knowledge_assistant.infrastructure.parsers.base import (
    DocumentParseError,
    DocumentParser,
    normalized_document,
)


class PdfParser(DocumentParser):
    supported_suffixes = (".pdf",)

    def parse(self, path: Path) -> list[NormalizedDocument]:
        try:
            with fitz.open(path) as pdf:
                title = str(pdf.metadata.get("title") or path.stem)
                return [
                    normalized_document(
                        path=path,
                        source_type="pdf",
                        content=page.get_text(),
                        title=title,
                        page_number=page_number,
                    )
                    for page_number, page in enumerate(pdf, start=1)
                ]
        except (fitz.FileDataError, OSError, RuntimeError, ValueError) as error:
            raise DocumentParseError(path, "could not parse PDF file") from error
