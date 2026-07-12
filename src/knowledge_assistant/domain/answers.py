from pydantic import BaseModel, ConfigDict

from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.retrieval import Citation


class AnswerResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[DocumentChunk, ...]
    latency_ms: float
