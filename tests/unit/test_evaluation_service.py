import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
from knowledge_assistant.application.evaluation.service import (
    EvaluationRecord,
    EvaluationService,
    EvaluationSummary,
)
from knowledge_assistant.application.generation.service import GenerationService
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata
from knowledge_assistant.domain.retrieval import RetrievedChunk
from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider
from knowledge_assistant.infrastructure.inference.base import TextGenerator
from knowledge_assistant.infrastructure.vector_store.base import VectorStore


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
    def __init__(self, output: str = "Câu trả lời đánh giá.") -> None:
        self.model_id = "fake-qwen"
        self.output = output

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        return self.output


@pytest.fixture
def fake_generation_service() -> GenerationService:
    meta = DocumentMetadata(
        document_id="doc-eval",
        source_path="guide.md",
        source_type="markdown",
        title="Guide",
        page_number=1,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:eval",
    )
    chunk = RetrievedChunk(
        chunk_id="doc-eval:c1",
        document_id="doc-eval",
        content="Nội dung tài liệu hướng dẫn đánh giá.",
        metadata=meta,
        ordinal=1,
        score=0.92,
    )
    vector_store = FakeVectorStore(chunks=[chunk])
    retrieval = RetrievalService(
        chunker=None,  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )
    return GenerationService(retrieval_service=retrieval, generator=FakeTextGenerator())


def test_evaluation_service_empty_dataset(tmp_path: Path, fake_generation_service: GenerationService) -> None:
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    service = EvaluationService(generation_service=fake_generation_service)
    summary = service.run(dataset_path)
    assert isinstance(summary, EvaluationSummary)
    assert summary.total_questions == 0
    assert summary.hit_at_k_rate == 0.0
    assert summary.average_latency_ms == 0.0


def test_evaluation_service_hit_calculation_and_normalization(
    tmp_path: Path, fake_generation_service: GenerationService
) -> None:
    dataset_path = tmp_path / "eval.jsonl"
    records = [
        {"question": "Câu hỏi 1", "expected_source": "guide.md"},
        {"question": "Câu hỏi 2", "expected_source": "missing.pdf"},
    ]
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    dataset_path.write_text("\n".join(lines), encoding="utf-8")

    service = EvaluationService(generation_service=fake_generation_service)
    summary = service.run(dataset_path, top_k=3)

    assert summary.total_questions == 2
    assert summary.hit_at_k_rate == 0.5
    assert len(summary.records) == 2

    assert isinstance(summary.records[0], EvaluationRecord)
    assert summary.records[0].hit_at_k is True
    assert summary.records[0].retrieved_sources == ["guide.md"]
    assert summary.records[1].hit_at_k is False
