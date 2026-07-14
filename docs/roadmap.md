# Project Roadmap & Phase Tracking: Vietnamese Enterprise Knowledge Assistant

## Current Phase: Phase 10 — Production MVP (Complete)

We have successfully designed, implemented, tested, and verified the complete vertical slice of the local-first Vietnamese Enterprise Knowledge Assistant RAG system.

### Completed Phases

- [x] **Phase 1: Domain Modeling (`src/knowledge_assistant/domain/`)**
  - Implemented immutable Pydantic entities with NFC Unicode normalization and deterministic hashing.
- [x] **Phase 2: Document Ingestion & Parsers (`src/knowledge_assistant/infrastructure/parsers/`)**
  - Implemented `PyMuPDFParser` (.pdf), `MarkdownParser` (.md), `DocxParser` (.docx), and `ParserDispatcher`.
  - Implemented `IngestionService` with graceful error capturing (`IngestionError`).
- [x] **Phase 3: Chunking Engine (`src/knowledge_assistant/application/ingestion/chunking.py`)**
  - Implemented `Chunker` using recursive character splitting with overlap and metadata inheritance.
- [x] **Phase 4: Embedding Provider (`src/knowledge_assistant/infrastructure/embeddings/e5.py`)**
  - Implemented `E5EmbeddingProvider` (`intfloat/multilingual-e5-small`) with asymmetric prefixes (`query: `/`passage: `).
- [x] **Phase 5: Vector Store (`src/knowledge_assistant/infrastructure/vector_store/qdrant.py`)**
  - Implemented `QdrantVectorStore` supporting `:memory:` and persistent disk storage (`qdrant-client`).
- [x] **Phase 6: Retrieval Service (`src/knowledge_assistant/application/retrieval/service.py`)**
  - Implemented document chunking, batch vector upserting, and top-k vector searching.
- [x] **Phase 7: Generation & Citation Engine (`src/knowledge_assistant/infrastructure/inference/mlx_qwen.py`)**
  - Implemented `MlxQwenGenerator` using Apple Silicon `mlx-lm` with lazy weight loading.
  - Implemented `GenerationService` (`src/knowledge_assistant/application/generation/service.py`) with Vietnamese system prompt formatting and precise citation linking (`Citation.excerpt`).
- [x] **Phase 8: CLI Vertical Slice (`cli/main.py` & `commands.py`)**
  - Built `Typer` CLI with `ingest`, `index`, `query`, and `evaluate` commands supporting `--json` output.
- [x] **Phase 9: FastAPI Service (`api/main.py` & `api/schemas.py`)**
  - Built REST API (`GET /health`, `POST /documents/ingest`, `POST /query`) with strict relative path verification and path traversal rejection.
- [x] **Phase 10: Evaluation Harness & Documentation (`src/knowledge_assistant/application/evaluation/service.py`)**
  - Implemented `EvaluationService` (`Hit@K` and average latency calculation), evaluation test fixtures, and comprehensive architecture/context documentation.

## Future Phases (Post-MVP Extensions)

- [ ] **Phase 11: Hybrid Search & Reranking**
  - Combine dense embeddings (`multilingual-e5-small`) with sparse BM25 / SPLADE scoring and cross-encoder reranking (`BAAI/bge-reranker-large`).
- [ ] **Phase 12: Advanced Chunking & Table/Image Extraction**
  - Support semantic markdown/headers chunking and multi-modal table structure preservation inside PDF/DOCX documents.
- [ ] **Phase 13: Conversation Memory & Multi-turn RAG**
  - Add session memory management and query condensation to support multi-turn conversational follow-ups.
- [ ] **Phase 14: Continuous Evaluation Dashboard**
  - Build a web UI for running dataset evaluations and visualizing hit rates and latency distributions over time.
