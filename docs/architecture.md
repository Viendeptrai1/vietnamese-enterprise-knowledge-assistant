# Architecture Documentation: Vietnamese Enterprise Knowledge Assistant

## Overview

The Vietnamese Enterprise Knowledge Assistant is a local-first, Apple Silicon (Mac M2 16GB) optimized RAG MVP designed using Domain-Driven Design (DDD) principles. It enables offline, zero-leakage enterprise document processing and question answering without relying on external cloud endpoints or proprietary APIs.

## Architectural Layers & Boundaries

The codebase strictly enforces clean dependency boundaries across four distinct layers:

```
[ CLI Vertical Slice (cli/) / FastAPI Service (api/) ]
                        |
                        v
[ Application Orchestration & Factories (src/knowledge_assistant/application/) ]
                        |
                        v
[ Domain Entities & Interfaces (src/knowledge_assistant/domain/) ]
                        ^
                        | (Implements interfaces / adapters)
[ Infrastructure Adapters (src/knowledge_assistant/infrastructure/) ]
```

### 1. Domain Layer (`src/knowledge_assistant/domain/`)
- **Entities & Value Objects:** `DocumentMetadata`, `NormalizedDocument`, `DocumentChunk`, `RetrievedChunk`, `Citation`, `AnswerResponse`.
- **Pure Domain Rules:**
  - `DocumentMetadata.page_number` is 1-indexed for visual documents (PDF) and `None` for continuous text streams (Markdown, DOCX).
  - All textual domain models enforce NFC Unicode normalization (`unicodedata.normalize("NFC", text)`).
  - Document IDs (`document_id`) and chunk IDs (`chunk_id`) are deterministically generated based on content hashing (`sha256:source_path:hash` and `document_id:ordinal`).
- **Dependencies:** Pure Python and Pydantic only. Zero imports from application or infrastructure layers.

### 2. Infrastructure Layer (`src/knowledge_assistant/infrastructure/`)
- **Document Parsers (`parsers/`):** Adapters implementing `DocumentParser` (`PyMuPDFParser` for `.pdf`, `MarkdownParser` for `.md`, `DocxParser` for `.docx`, and `ParserDispatcher` for format-routing).
- **Embedding Adapters (`embeddings/`):** `E5EmbeddingProvider` wraps `intfloat/multilingual-e5-small` (`sentence-transformers`) and applies `query: ` and `passage: ` asymmetric prefixes as required by E5 models.
- **Vector Store Adapters (`vector_store/`):** `QdrantVectorStore` wraps local Qdrant persistence (`qdrant-client` using `:memory:` or local folder storage).
- **Inference Adapters (`inference/`):** `MlxQwenGenerator` wraps Apple Silicon MLX (`mlx-lm` with model `mlx-community/Qwen2.5-1.5B-Instruct-4bit`). Implements lazy loading (`_ensure_loaded()`) so that CLI `--help`, static checks, or quick non-generation tasks execute in milliseconds without loading heavy model weights into RAM.

### 3. Application Layer (`src/knowledge_assistant/application/`)
- **Use Case Orchestration (`ingestion/`, `retrieval/`, `generation/`, `evaluation/`):**
  - `IngestionService` coordinates document parsing and error capturing (`IngestionError`).
  - `Chunker` implements recursive character splitting with configurable overlaps.
  - `RetrievalService` coordinates chunking, embedding generation, and Qdrant indexing/searching.
  - `GenerationService` builds grounded Vietnamese system prompts (`system_prompt`), formats context with explicit `[Source i]` headers, and returns `AnswerResponse` with deterministic citations (`citation.excerpt` derived exactly from `RetrievedChunk`).
  - `EvaluationService` runs RAG accuracy evaluation (`Hit@K`) against validation JSONL datasets.
- **Dependency Container (`commands.py`):** `ApplicationContainer` isolates all infrastructure instantiations and provides pure factory use cases (`run_ingest`, `run_index`, `run_query`, `run_evaluate`).

### 4. Presentation & Vertical Slices (`cli/` & `api/`)
- **CLI (`cli/main.py`):** Built with `Typer` and `Rich`. Exposes `ingest`, `index`, `query`, and `evaluate` commands with `--json` flags for automation.
- **FastAPI Service (`api/main.py`):** Exposes `GET /health`, `POST /documents/ingest`, and `POST /query`. Enforces strict relative path resolution against `DOCUMENT_ROOT` and blocks path traversal attempts (`..`).

## Configuration & Environment Variables

- `QDRANT_PATH` — Local directory for Qdrant vector database storage (defaults to `:memory:` or `./data/qdrant_db`).
- `DOCUMENT_ROOT` — Base root directory for document ingestion via REST API (defaults to `./data/documents`).
- `MLX_MODEL_ID` — HuggingFace model ID for local MLX inference (defaults to `mlx-community/Qwen2.5-1.5B-Instruct-4bit`).

## Testing Rules & Quality Standards

- **Isolated Testing:** Unit and vertical slice tests use fake providers (`FakeEmbeddingProvider`, `FakeVectorStore`, `FakeTextGenerator`, `FakeApplicationContainer`) to guarantee fast, reliable execution without network access or model downloads.
- **Rule of Testing:** Every bug fix, architectural modification, or behavior change must be accompanied by a focused, automated unit or integration test before commit.
