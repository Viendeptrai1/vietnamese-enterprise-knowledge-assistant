from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceType = Literal["pdf", "markdown", "docx"]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    source_path: str
    source_type: SourceType
    title: str
    page_number: int | None = None
    modified_at: datetime
    content_hash: str


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    content: str
