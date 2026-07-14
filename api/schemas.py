from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_ready: bool


class IngestionRequest(BaseModel):
    path: str = Field(..., description="Path relative to document root.")

    @field_validator("path")
    @classmethod
    def validate_path_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ingestion path must not be empty.")
        return value.strip()


class IngestionFailureSchema(BaseModel):
    source_path: str
    error: str


class IngestionSummaryResponse(BaseModel):
    root_path: str
    total_documents: int
    successful: int
    failed: int
    failures: list[IngestionFailureSchema]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer from knowledge base.")
    top_k: int = Field(5, ge=1, le=100, description="Number of top chunks to retrieve.")

    @field_validator("question")
    @classmethod
    def validate_question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty.")
        return value.strip()
