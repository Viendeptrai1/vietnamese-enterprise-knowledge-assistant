from pydantic import BaseModel, ConfigDict

from knowledge_assistant.domain.documents import DocumentMetadata


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    metadata: DocumentMetadata
    ordinal: int
