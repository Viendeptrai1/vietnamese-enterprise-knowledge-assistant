from pathlib import Path

from knowledge_assistant.domain.documents import NormalizedDocument
from knowledge_assistant.infrastructure.parsers.base import (
    DocumentParseError,
    DocumentParser,
    normalized_document,
)


class MarkdownParser(DocumentParser):
    supported_suffixes = (".md", ".markdown")

    def parse(self, path: Path) -> list[NormalizedDocument]:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise DocumentParseError(path, "could not read Markdown file") from error

        title = next(
            (line.removeprefix("#").strip() for line in content.splitlines() if line.startswith("#")),
            path.stem,
        )
        return [
            normalized_document(
                path=path,
                source_type="markdown",
                content=content,
                title=title,
            )
        ]
