import time

from knowledge_assistant.application.generation.prompts import UNKNOWN_ANSWER, build_bounded_prompt
from knowledge_assistant.application.retrieval.service import RetrievalService
from knowledge_assistant.domain.answers import AnswerResponse
from knowledge_assistant.domain.retrieval import Citation, RetrievedChunk
from knowledge_assistant.infrastructure.inference.base import TextGenerator


class GenerationService:
    """Application service orchestrating grounded answer generation from retrieved context."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generator: TextGenerator,
        max_context_chars: int = 3000,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generator = generator
        self.max_context_chars = max_context_chars

    def answer(self, question: str, top_k: int = 5) -> AnswerResponse:
        """Retrieve relevant context and generate a grounded answer with citations."""
        start_time = time.perf_counter()
        retrieved: list[RetrievedChunk] = self.retrieval_service.search(question, top_k=top_k)
        if not retrieved:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return AnswerResponse(
                answer=UNKNOWN_ANSWER,
                citations=(),
                retrieved_chunks=(),
                latency_ms=round(latency_ms, 2),
            )

        prompt, selected_chunks = build_bounded_prompt(
            question=question,
            chunks=retrieved,
            max_context_chars=self.max_context_chars,
        )
        if not selected_chunks:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return AnswerResponse(
                answer=UNKNOWN_ANSWER,
                citations=(),
                retrieved_chunks=(),
                latency_ms=round(latency_ms, 2),
            )

        generated_text = self.generator.generate(prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        citations = tuple(
            Citation(
                chunk_id=chunk.chunk_id,
                source_path=chunk.metadata.source_path,
                page_number=chunk.metadata.page_number,
                excerpt=chunk.content,
            )
            for chunk in selected_chunks
        )

        return AnswerResponse(
            answer=generated_text,
            citations=citations,
            retrieved_chunks=tuple(selected_chunks),
            latency_ms=round(latency_ms, 2),
        )
