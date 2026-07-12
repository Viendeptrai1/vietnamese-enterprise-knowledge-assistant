import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from knowledge_assistant.domain.documents import (
    DocumentMetadata,
    NormalizedDocument,
    SourceType,
)


class DocumentParseError(Exception):
    """Raised when a supported local document cannot be parsed."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


class UnsupportedDocumentTypeError(DocumentParseError):
    """Raised when no local parser supports a file suffix."""


class DocumentParser(ABC):
    supported_suffixes: tuple[str, ...]

    @abstractmethod
    def parse(self, path: Path) -> list[NormalizedDocument]:
        """Parse one supported local file into normalized documents."""


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    normalized: list[str] = []
    previous_was_blank = False
    for line in lines:
        if not line:
            if normalized and not previous_was_blank:
                normalized.append("")
            previous_was_blank = True
            continue
        normalized.append(line)
        previous_was_blank = False
    return "\n".join(normalized).strip()


def normalized_document(
    *,
    path: Path,
    source_type: SourceType,
    content: str,
    title: str,
    page_number: int | None = None,
) -> NormalizedDocument:
    normalized_content = normalize_whitespace(content)
    source_path = path.as_posix()
    document_identity = f"{source_path}:{page_number or 0}"
    return NormalizedDocument(
        metadata=DocumentMetadata(
            document_id=f"sha256:{sha256(document_identity.encode()).hexdigest()}",
            source_path=source_path,
            source_type=source_type,
            title=title,
            page_number=page_number,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            content_hash=f"sha256:{sha256(normalized_content.encode()).hexdigest()}",
        ),
        content=normalized_content,
    )
