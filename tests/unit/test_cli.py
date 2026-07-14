from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
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
from typer.testing import CliRunner

from cli.main import app, set_container


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
    def __init__(self, output: str = "Câu trả lời từ fake CLI model.") -> None:
        self.model_id = "fake-qwen"
        self.output = output

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return self.output


class FakeApplicationContainer(ApplicationContainer):
    def __init__(self, root_path: Path) -> None:
        super().__init__()
        self.root_path = root_path
        self._retrieved_chunks: list[RetrievedChunk] = []

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
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_container(tmp_path: Path) -> FakeApplicationContainer:
    doc_file = tmp_path / "sample.md"
    doc_file.write_text("# Tiêu đề\nNội dung tài liệu kiểm thử CLI.", encoding="utf-8")
    container = FakeApplicationContainer(root_path=tmp_path)
    set_container(container)
    yield container
    set_container(None)


def test_cli_help_does_not_load_models_and_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "index" in result.output
    assert "query" in result.output
    assert "evaluate" in result.output


def test_cli_ingest_invalid_path_returns_nonzero_exit_code(runner: CliRunner) -> None:
    result = runner.invoke(app, ["ingest", "/path/does/not/exist/cli/test"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_cli_ingest_valid_path_prints_deterministic_summary(
    runner: CliRunner, fake_container: FakeApplicationContainer
) -> None:
    result = runner.invoke(app, ["ingest", str(fake_container.root_path)])
    assert result.exit_code == 0
    assert "Ingestion Summary" in result.output
    assert "Total documents processed: 1" in result.output
    assert "Successful: 1 | Failed: 0" in result.output


def test_cli_query_prints_answer_and_citations(runner: CliRunner, fake_container: FakeApplicationContainer) -> None:
    meta = DocumentMetadata(
        document_id="doc-1",
        source_path="sample.md",
        source_type="markdown",
        title="Sample",
        page_number=1,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:123",
    )
    chunk = RetrievedChunk(
        chunk_id="doc-1:c1",
        document_id="doc-1",
        content="Nội dung tài liệu kiểm thử CLI.",
        metadata=meta,
        ordinal=1,
        score=0.9,
    )
    fake_container._retrieved_chunks.append(chunk)

    result = runner.invoke(app, ["query", "Hỏi nội dung gì?"])
    assert result.exit_code == 0
    assert "Câu trả lời từ fake CLI model." in result.output
    assert "Citations:" in result.output
    assert "sample.md (Page 1)" in result.output


def test_cli_index_valid_path_and_json_option(runner: CliRunner, fake_container: FakeApplicationContainer) -> None:
    result = runner.invoke(app, ["index", str(fake_container.root_path), "--json"])
    assert result.exit_code == 0
    assert '"root_path":' in result.output
    assert '"total_chunks":' in result.output


def test_cli_evaluate_valid_path(runner: CliRunner, fake_container: FakeApplicationContainer) -> None:
    result = runner.invoke(app, ["evaluate", str(fake_container.root_path)])
    assert result.exit_code == 0
    assert "Evaluation Status" in result.output
