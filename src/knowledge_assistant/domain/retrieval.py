from pydantic import BaseModel, ConfigDict, Field

from knowledge_assistant.domain.chunks import DocumentChunk


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_path: str
    page_number: int | None = None
    excerpt: str


class RetrievedChunk(DocumentChunk):
    score: float


class IndexSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    upserted_chunks: int = Field(ge=0)
