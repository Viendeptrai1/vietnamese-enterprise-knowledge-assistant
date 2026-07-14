from typing import Sequence

from knowledge_assistant.application.ingestion.chunking import Chunker
from knowledge_assistant.domain.documents import NormalizedDocument
from knowledge_assistant.domain.retrieval import IndexSummary, RetrievedChunk
from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider
from knowledge_assistant.infrastructure.vector_store.base import VectorStore


class RetrievalService:
    """Application service for indexing normalized documents and searching knowledge base."""

    def __init__(
        self,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index(self, documents: Sequence[NormalizedDocument]) -> IndexSummary:
        """Chunk documents, embed chunks in batches, and upsert them to vector store."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunker.chunk(doc))

        upserted_count = 0
        for i in range(0, len(all_chunks), self.batch_size):
            batch_chunks = all_chunks[i : i + self.batch_size]
            texts = [chunk.content for chunk in batch_chunks]
            vectors = self.embedding_provider.embed_documents(texts)
            upserted_count += self.vector_store.upsert(
                chunks=batch_chunks,
                vectors=vectors,
                model_id=self.embedding_provider.model_id,
            )

        return IndexSummary(
            total_documents=len(documents),
            total_chunks=len(all_chunks),
            upserted_chunks=upserted_count,
        )

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Embed question and retrieve top_k matching chunks from the active model id."""
        if not question.strip():
            return []
        query_vector = self.embedding_provider.embed_query(question)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            model_id=self.embedding_provider.model_id,
        )
