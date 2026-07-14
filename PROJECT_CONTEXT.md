# Project Context: Vietnamese Enterprise Knowledge Assistant

## Quick Summary for AI Assistants & Developers

This repository implements a **local-first, Apple Silicon (Mac M2 16GB) optimized Vietnamese Enterprise Knowledge Assistant RAG system** cleanly structured with Domain-Driven Design (DDD).

- **Current Status:** Phase 10 (Production MVP Complete).
- **Primary Tech Stack:** Python 3.11, Poetry (`poetry == 1.8.4`), Pydantic v2, Typer, FastAPI, Qdrant (`qdrant-client`), Sentence Transformers (`intfloat/multilingual-e5-small`), Apple Silicon MLX (`mlx-lm` with `mlx-community/Qwen2.5-1.5B-Instruct-4bit`).

## Core Commands

Always execute commands inside Poetry:

```bash
# Run tests (Unit & Integration test suite with fakes, zero network/model loading)
poetry run pytest -q

# Run static checks and formatting
poetry run ruff check src/knowledge_assistant cli api tests/unit tests/integration
poetry run ruff format --check src/knowledge_assistant cli api tests/unit tests/integration

# Run CLI commands
poetry run knowledge-assistant --help
poetry run knowledge-assistant ingest path/to/documents
poetry run knowledge-assistant index path/to/documents
poetry run knowledge-assistant query "Quy trình làm việc và chính sách công ty là gì?" --top-k 5
poetry run knowledge-assistant evaluate tests/fixtures/evaluation/questions.jsonl

# Run FastAPI Server
poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## Architecture & Layer Boundaries

The codebase strictly separates dependencies:
1. `src/knowledge_assistant/domain/` — Pure Pydantic entities (`NormalizedDocument`, `DocumentChunk`, `RetrievedChunk`, `Citation`, `AnswerResponse`). ZERO imports from outer layers.
2. `src/knowledge_assistant/infrastructure/` — Adapters for parsing (`pymupdf`, `python-docx`), embeddings (`E5EmbeddingProvider`), vector database (`QdrantVectorStore`), and local inference (`MlxQwenGenerator`).
3. `src/knowledge_assistant/application/` — Orchestration services (`IngestionService`, `RetrievalService`, `GenerationService`, `EvaluationService`) and dependency injection container (`ApplicationContainer`).
4. `cli/` & `api/` — Vertical presentation slices delegating all business logic to application services.

## Known Limitations (Local MVP)

1. **Hardware & OS:** Optimized specifically for macOS with Apple Silicon (`mlx-lm`). Execution on Linux/Windows or CUDA requires substituting `MlxQwenGenerator` with a compatible HuggingFace/vLLM generator.
2. **Local Storage:** Qdrant runs locally (`:memory:` or local directory persistence). No remote cluster clustering is enabled by default.
3. **Document Formats:** Supports `.pdf`, `.md`, and `.docx`. Complex OCR or multi-column table layout extraction is out of scope for MVP.

## Mandatory Testing & Modification Rules

- **Rule of Verification:** Every new feature, bug fix, or behavioral change MUST be verified by a focused automated unit or integration test before committing.
- **Test Speed & Isolation:** Tests must run in seconds and NEVER load MLX models (`mlx-lm`) or download HuggingFace weights. Always inject fakes (`FakeEmbeddingProvider`, `FakeVectorStore`, `FakeTextGenerator`, `FakeApplicationContainer`).
- **No Hallucinations:** Check existing domain models and infrastructure adapters before creating new classes or introducing dependencies.
