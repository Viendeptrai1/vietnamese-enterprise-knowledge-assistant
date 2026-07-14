from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
from fastapi.testclient import TestClient
from knowledge_assistant.application.commands import ApplicationContainer
from knowledge_assistant.application.generation.service import GenerationService
from knowledge_assistant.application.ingestion.chunking import Chunker
from knowledge_assistant.application.ingestion.service import IngestionService
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata
from knowledge_assistant.domain.retrieval import RetrievedChunk
from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider
from knowledge_assistant.infrastructure.inference.base import TextGenerator
from knowledge_assistant.infrastructure.parsers.dispatcher import ParserDispatcher
from knowledge_assistant.infrastructure.vector_store.base import VectorStore

from api.main import create_app


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model_id = "fake-e5"
        self.dimension = 4

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in chunks]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * self.dimension


class FakeVectorStore(VectorStore):
    def __init__(self, chunks: Sequence[RetrievedChunk] = ()) -> None:
        self.chunks = list(chunks)

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        model_id: str,
    ) -> int:
        return len(chunks)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        model_id: str | None = None,
    ) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeTextGenerator(TextGenerator):
    def __init__(self, output: str = "Câu trả lời từ fake API model.") -> None:
        self.model_id = "fake-qwen"
        self.output = output

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return self.output


class FakeApplicationContainer(ApplicationContainer):
    def __init__(self, retrieved_chunks: Sequence[RetrievedChunk] = ()) -> None:
        super().__init__()
        self._retrieved_chunks = list(retrieved_chunks)

    def ingestion_service(self) -> IngestionService:
        return IngestionService(dispatcher=ParserDispatcher())

    def retrieval_service(self) -> RetrievalService:
        return RetrievalService(
            chunker=Chunker(),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(chunks=self._retrieved_chunks),
        )

    def generation_service(self) -> GenerationService:
        return GenerationService(
            retrieval_service=self.retrieval_service(),
            generator=FakeTextGenerator(),
        )


@pytest.fixture
def fake_document_root(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    sample = docs_dir / "sample.md"
    sample.write_text("# Tài liệu API\nNội dung kiểm thử API ingestion.", encoding="utf-8")
    return docs_dir


@pytest.fixture
def client(fake_document_root: Path) -> TestClient:
    meta = DocumentMetadata(
        document_id="doc-api",
        source_path="sample.md",
        source_type="markdown",
        title="Sample API",
        page_number=1,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:api",
    )
    chunk = RetrievedChunk(
        chunk_id="doc-api:c1",
        document_id="doc-api",
        content="Nội dung kiểm thử API ingestion.",
        metadata=meta,
        ordinal=1,
        score=0.95,
    )
    container = FakeApplicationContainer(retrieved_chunks=[chunk])
    app = create_app(container=container, document_root=fake_document_root)
    return TestClient(app)


def test_api_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True


def test_api_query_validation_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "   ", "top_k": 5})
    assert response.status_code in (400, 422)


def test_api_query_validation_invalid_top_k(client: TestClient) -> None:
    response = client.post("/query", json={"question": "Xin chào?", "top_k": -1})
    assert response.status_code in (400, 422)


def test_api_query_successful_with_citations(client: TestClient) -> None:
    response = client.post("/query", json={"question": "Hỏi thông tin gì?", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Câu trả lời từ fake API model."
    assert "citations" in data
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source_path"] == "sample.md"
    assert "retrieved_chunks" in data


def test_api_ingest_path_traversal_rejected(client: TestClient) -> None:
    response = client.post("/documents/ingest", json={"path": "../../etc/passwd"})
    assert response.status_code in (400, 403)


def test_api_ingest_absolute_path_rejected(client: TestClient) -> None:
    response = client.post("/documents/ingest", json={"path": "/etc/passwd"})
    assert response.status_code in (400, 403)


def test_api_ingest_successful_relative_path(client: TestClient) -> None:
    response = client.post("/documents/ingest", json={"path": "sample.md"})
    assert response.status_code == 200
    data = response.json()
    assert "successful" in data
    assert data["successful"] == 1
    assert data["failed"] == 0
