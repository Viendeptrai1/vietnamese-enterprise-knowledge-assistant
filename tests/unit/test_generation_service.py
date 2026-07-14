from datetime import datetime, timezone
from typing import Sequence

import pytest
from knowledge_assistant.application.generation.prompts import UNKNOWN_ANSWER, build_bounded_prompt
from knowledge_assistant.application.generation.service import GenerationService
from knowledge_assistant.application.ingestion.chunking import Chunker
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import DocumentMetadata
from knowledge_assistant.domain.retrieval import RetrievedChunk
from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider
from knowledge_assistant.infrastructure.inference.base import TextGenerator
from knowledge_assistant.infrastructure.inference.mlx_qwen import MlxQwenGenerator
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
    def __init__(self, output: str = "Câu trả lời grounded.") -> None:
        self.model_id = "fake-qwen"
        self.output = output
        self.calls: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        self.calls.append(prompt)
        return self.output


def sample_retrieved_chunk(chunk_id: str, content: str, page_num: int | None = 1) -> RetrievedChunk:
    meta = DocumentMetadata(
        document_id="doc-1",
        source_path="docs/guide.pdf",
        source_type="pdf",
        title="Guide",
        page_number=page_num,
        modified_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        content_hash="sha256:abc",
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=meta.document_id,
        content=content,
        metadata=meta,
        ordinal=1,
        score=0.85,
    )


def test_build_bounded_prompt_limits_context_and_includes_source_labels() -> None:
    chunk1 = sample_retrieved_chunk("doc-1:c1", "A" * 150, page_num=2)
    chunk2 = sample_retrieved_chunk("doc-1:c2", "B" * 150, page_num=3)
    chunk3 = sample_retrieved_chunk("doc-1:c3", "C" * 150, page_num=4)

    prompt, selected = build_bounded_prompt("Hỏi về ABC?", [chunk1, chunk2, chunk3], max_context_chars=350)

    # Chunks 1 and 2 take ~300+ chars with labels, chunk 3 should be omitted
    assert len(selected) < 3
    assert chunk1 in selected
    assert "docs/guide.pdf, trang 2" in prompt
    assert "--- CÂU HỎI ---\nHỏi về ABC?" in prompt


def test_generation_service_returns_unknown_answer_when_no_context_retrieved() -> None:
    retrieval_service = RetrievalService(
        chunker=Chunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(chunks=[]),
    )
    generator = FakeTextGenerator()
    service = GenerationService(retrieval_service=retrieval_service, generator=generator)

    resp = service.answer("câu hỏi không có context", top_k=5)

    assert resp.answer == UNKNOWN_ANSWER
    assert resp.citations == ()
    assert resp.retrieved_chunks == ()
    assert len(generator.calls) == 0  # Model not invoked when context is empty


def test_generation_service_retains_citations_and_invokes_generator() -> None:
    chunk1 = sample_retrieved_chunk("doc-1:c1", "Thông tin 1.")
    chunk2 = sample_retrieved_chunk("doc-1:c2", "Thông tin 2.")
    retrieval_service = RetrievalService(
        chunker=Chunker(),
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(chunks=[chunk1, chunk2]),
    )
    generator = FakeTextGenerator(output="Kết quả tổng hợp từ thông tin 1 và 2.")
    service = GenerationService(retrieval_service=retrieval_service, generator=generator)

    resp = service.answer("Hỏi gì đó?", top_k=2)

    assert resp.answer == "Kết quả tổng hợp từ thông tin 1 và 2."
    assert len(resp.citations) == 2
    assert resp.citations[0].chunk_id == chunk1.chunk_id
    assert resp.citations[0].source_path == chunk1.metadata.source_path
    assert resp.citations[0].page_number == chunk1.metadata.page_number
    assert len(generator.calls) == 1
    assert "Thông tin 1." in generator.calls[0]
    assert "Thông tin 2." in generator.calls[0]


def test_mlx_qwen_generator_lazy_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = MlxQwenGenerator(model_id="test-qwen")
    assert generator._model is None
    assert generator._tokenizer is None

    # Simulate mlx_lm missing to verify our boundary exception handling without triggering network download
    import importlib

    orig_import = importlib.import_module

    def mock_import(name: str, package: str | None = None):
        if name == "mlx_lm":
            raise ImportError("No module named 'mlx_lm'")
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", mock_import)

    with pytest.raises(RuntimeError, match="mlx_lm is not installed"):
        generator.generate("prompt test")
