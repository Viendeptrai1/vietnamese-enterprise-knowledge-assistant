from pydantic import BaseModel, ConfigDict


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_path: str
    page_number: int | None = None
    excerpt: str
