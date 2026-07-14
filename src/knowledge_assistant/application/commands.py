import os
from pathlib import Path
from typing import Any

from knowledge_assistant.application.evaluation.service import EvaluationService
from knowledge_assistant.application.generation.service import GenerationService
from knowledge_assistant.application.ingestion.chunking import Chunker
from knowledge_assistant.application.ingestion.service import IngestionService
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.infrastructure.embeddings.e5 import E5EmbeddingProvider
from knowledge_assistant.infrastructure.inference.mlx_qwen import MlxQwenGenerator
from knowledge_assistant.infrastructure.parsers.dispatcher import ParserDispatcher
from knowledge_assistant.infrastructure.vector_store.qdrant import QdrantVectorStore


class ApplicationContainer:
    """Factory creating and wiring application use case services from configuration settings."""

    def __init__(
        self,
        embedding_model_id: str | None = None,
        vector_store_path: Path | str | None = None,
        qwen_model_id: str | None = None,
        mlx_cache_path: str | None = None,
    ) -> None:
        self.embedding_model_id = embedding_model_id or os.environ.get(
            "EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-small"
        )
        self.vector_store_path = vector_store_path or os.environ.get("VECTOR_STORE_PATH", "data/indexes")
        self.qwen_model_id = qwen_model_id or os.environ.get(
            "QWEN_MODEL_ID", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )

        self.mlx_cache_path = mlx_cache_path or os.environ.get("MLX_MODEL_CACHE_PATH", "data/models")

    def ingestion_service(self) -> IngestionService:
        return IngestionService(dispatcher=ParserDispatcher())

    def retrieval_service(self) -> RetrievalService:
        chunker = Chunker()
        embedding_provider = E5EmbeddingProvider(model_id=self.embedding_model_id)
        vector_store = QdrantVectorStore(collection_name="knowledge_base", path=self.vector_store_path)
        return RetrievalService(
            chunker=chunker,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

    def generation_service(self) -> GenerationService:
        retrieval = self.retrieval_service()
        generator = MlxQwenGenerator(
            model_id=self.qwen_model_id,
            model_cache_path=self.mlx_cache_path,
        )
        return GenerationService(retrieval_service=retrieval, generator=generator)

    def evaluation_service(self) -> EvaluationService:
        return EvaluationService(generation_service=self.generation_service())


def run_ingest(path: Path, container: ApplicationContainer | None = None) -> dict[str, Any]:
    """Ingest documents from path into normalized models."""
    if not path.exists():
        raise FileNotFoundError(f"Document root path does not exist: {path}")
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"Document root path is not valid: {path}")
    if container is None:
        container = ApplicationContainer()
    service = container.ingestion_service()
    documents = service.ingest(path)
    return {
        "root_path": str(path),
        "total_documents": len(documents) + len(service.errors),
        "successful": len(documents),
        "failed": len(service.errors),
        "failures": [{"source_path": str(f.path), "error": str(f.message)} for f in service.errors],
    }


def run_index(path: Path, container: ApplicationContainer | None = None) -> dict[str, Any]:
    """Ingest documents from path and index chunks into vector store."""
    if not path.exists():
        raise FileNotFoundError(f"Document root path does not exist: {path}")
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"Document root path is not valid: {path}")
    if container is None:
        container = ApplicationContainer()
    ingest_service = container.ingestion_service()
    documents = ingest_service.ingest(path)
    retrieval_service = container.retrieval_service()
    index_summary = retrieval_service.index(documents)
    return {
        "root_path": str(path),
        "ingested_documents": len(documents),
        "total_chunks": index_summary.total_chunks,
        "upserted_chunks": index_summary.upserted_chunks,
    }


def run_query(question: str, top_k: int = 5, container: ApplicationContainer | None = None) -> dict[str, Any]:
    """Retrieve grounded answer and citations for question."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if container is None:
        container = ApplicationContainer()
    generation_service = container.generation_service()
    response = generation_service.answer(question, top_k=top_k)
    return {
        "question": question,
        "answer": response.answer,
        "latency_ms": response.latency_ms,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "page_number": c.page_number,
                "excerpt": c.excerpt,
            }
            for c in response.citations
        ],
    }


def run_evaluate(path: Path, top_k: int = 5, container: ApplicationContainer | None = None) -> dict[str, Any]:
    """Run evaluation harness against dataset path."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset path does not exist: {path}")
    if container is None:
        container = ApplicationContainer()
    service = container.evaluation_service()
    summary = service.run(path, top_k=top_k)
    return {
        "dataset_path": str(path),
        "status": "completed",
        "total_questions": summary.total_questions,
        "hit_at_k_rate": summary.hit_at_k_rate,
        "average_latency_ms": summary.average_latency_ms,
        "records_processed": summary.total_questions,
    }
