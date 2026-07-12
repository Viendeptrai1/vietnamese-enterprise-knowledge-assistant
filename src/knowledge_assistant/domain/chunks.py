from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowledge_assistant.domain.documents import DocumentMetadata


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str
    content: str
    metadata: DocumentMetadata
    ordinal: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_document_id(self) -> "DocumentChunk":
        if self.document_id != self.metadata.document_id:
            raise ValueError("document_id must match metadata.document_id")
        return self
