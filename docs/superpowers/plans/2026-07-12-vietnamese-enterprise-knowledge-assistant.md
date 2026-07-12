# Vietnamese Enterprise Knowledge Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Vietnamese enterprise document assistant that ingests PDF/Markdown/DOCX files, retrieves grounded context, and answers through Qwen3.5-2B via MLX with CLI and FastAPI interfaces.

**Architecture:** Create a clean `src/knowledge_assistant` package with domain, application, and infrastructure boundaries. Keep the original LLM Twin implementation under `legacy/`; the new application uses local files, a multilingual embedding provider, a local Qdrant index, and an MLX generation adapter. CLI and FastAPI are thin adapters over the same application services.

**Tech Stack:** Python 3.11, Poetry, PyMuPDF, python-docx, Pydantic, Qdrant Client local mode, `intfloat/multilingual-e5-small`, MLX/MLX-LM, Qwen3.5-2B, Typer, FastAPI, pytest.

## Global Constraints

- Input formats are local PDF, Markdown, and DOCX files only.
- The original LLM Twin code and data remain available under `legacy/`.
- The generation model is `Qwen/Qwen3.5-2B` and runs through MLX/MLX-LM on Apple Silicon.
- The core local flow must not require AWS, SageMaker, Comet ML, or OpenAI.
- MVP does not include web UI, web crawling, fine-tuning, multi-tenancy, or enterprise authentication.
- The API must not read files outside the configured document root.
- Ingestion must be idempotent for unchanged files.
- Every task ends with a focused test/check and a separate Git commit.

---

## File Map

### New files

- `src/knowledge_assistant/domain/documents.py`: normalized document and source metadata models.
- `src/knowledge_assistant/domain/chunks.py`: chunk models and stable chunk identity.
- `src/knowledge_assistant/domain/retrieval.py`: retrieval result and citation models.
- `src/knowledge_assistant/domain/answers.py`: answer response model.
- `src/knowledge_assistant/application/ingestion/service.py`: ingest use case orchestration.
- `src/knowledge_assistant/application/retrieval/service.py`: index and search use cases.
- `src/knowledge_assistant/application/generation/service.py`: grounded answer orchestration.
- `src/knowledge_assistant/application/evaluation/service.py`: evaluation use case.
- `src/knowledge_assistant/infrastructure/parsers/*.py`: PDF, Markdown, and DOCX adapters.
- `src/knowledge_assistant/infrastructure/embeddings/e5.py`: multilingual E5 embedding adapter.
- `src/knowledge_assistant/infrastructure/vector_store/qdrant.py`: local Qdrant adapter.
- `src/knowledge_assistant/infrastructure/inference/mlx_qwen.py`: MLX Qwen adapter.
- `src/knowledge_assistant/config.py`: typed environment/config settings.
- `cli/main.py`: Typer commands.
- `api/main.py`: FastAPI application and request/response schemas.
- `tests/fixtures/documents/*`: small deterministic fixture files.
- `tests/unit/*` and `tests/integration/*`: focused behavior tests.
- `docs/architecture.md`: implementation architecture and data flow.
- `docs/roadmap.md`: post-MVP milestones.
- `PROJECT_CONTEXT.md`: persistent agent/developer context.

### Existing files to modify

- `pyproject.toml`: new package layout, dependencies, CLI entry point, and task commands.
- `.gitignore`: document/index directories and MLX model cache paths.
- `README.md`: new project purpose, setup, commands, architecture, and limits.

### Existing files to move

- `llm_engineering/`, `pipelines/`, `steps/`, `tools/`, `configs/`, `code_snippets/`, old tests, Docker/AWS configs, and old model documentation move under `legacy/llm-engineers-handbook/`.
- The handbook PDF and parsed Markdown remain at root or under `docs/reference/` because they are learning references, not runtime source.

---

### Task 1: Isolate the legacy project and create the new package skeleton

**Files:**
- Create: `legacy/llm-engineers-handbook/` by moving the old implementation directories/files listed in the File Map.
- Create: `src/knowledge_assistant/__init__.py`
- Create: `src/knowledge_assistant/domain/__init__.py`
- Create: `src/knowledge_assistant/application/__init__.py`
- Create: `src/knowledge_assistant/infrastructure/__init__.py`
- Create: `cli/__init__.py`
- Create: `api/__init__.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Test: `tests/unit/test_project_layout.py`

**Interfaces:**
- Produces an importable package named `knowledge_assistant`.
- The old implementation remains readable under `legacy/llm-engineers-handbook/`.

- [ ] **Step 1: Write the layout test**

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_new_package_and_legacy_tree_exist():
    assert (ROOT / "src/knowledge_assistant/__init__.py").exists()
    assert (ROOT / "legacy/llm-engineers-handbook").is_dir()
    assert not (ROOT / "llm_engineering").exists()
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `poetry run pytest tests/unit/test_project_layout.py -q`

Expected: FAIL because the new package and legacy directory do not exist yet.

- [ ] **Step 3: Move legacy files and add package markers**

Use `git mv` for the existing LLM Twin directories and files. Do not move `.git`, the handbook references, `README.md`, `pyproject.toml`, or the root `.gitignore`. Add empty package markers under `src/knowledge_assistant`, `cli`, and `api`.

- [ ] **Step 4: Update Poetry package discovery**

Configure `pyproject.toml` to package `src/knowledge_assistant`, expose the command `knowledge-assistant = "cli.main:app"`, and retain the existing lock file until dependency changes are made.

- [ ] **Step 5: Run the test and verify it passes**

Run: `poetry run pytest tests/unit/test_project_layout.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add legacy src cli api tests/unit/test_project_layout.py pyproject.toml .gitignore
git commit -m "refactor: isolate legacy project and create app skeleton"
```

---

### Task 2: Define document and chunk domain models

**Files:**
- Create: `src/knowledge_assistant/domain/documents.py`
- Create: `src/knowledge_assistant/domain/chunks.py`
- Create: `src/knowledge_assistant/domain/retrieval.py`
- Create: `src/knowledge_assistant/domain/answers.py`
- Create: `tests/unit/test_domain_models.py`

**Interfaces:**
- `SourceType = Literal["pdf", "markdown", "docx"]`.
- `DocumentMetadata(document_id, source_path, source_type, title, page_number, modified_at, content_hash)`.
- `NormalizedDocument(metadata, content)`.
- `DocumentChunk(chunk_id, document_id, content, metadata, ordinal)`.
- `Citation(chunk_id, source_path, page_number, excerpt)`.
- `AnswerResponse(answer, citations, retrieved_chunks, latency_ms)`.

- [ ] **Step 1: Write failing model tests**

Test that valid models round-trip, unsupported source types fail, and `DocumentChunk` requires a stable `chunk_id`.

- [ ] **Step 2: Run the focused tests**

Run: `poetry run pytest tests/unit/test_domain_models.py -q`

Expected: FAIL because the models do not exist.

- [ ] **Step 3: Implement immutable Pydantic/domain dataclasses**

Use explicit types, no framework imports outside the domain model dependency, and preserve page number as optional metadata.

- [ ] **Step 4: Run the focused tests**

Run: `poetry run pytest tests/unit/test_domain_models.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_assistant/domain tests/unit/test_domain_models.py
git commit -m "feat: add knowledge assistant domain models"
```

---

### Task 3: Implement local PDF, Markdown, and DOCX parsing

**Files:**
- Create: `src/knowledge_assistant/infrastructure/parsers/base.py`
- Create: `src/knowledge_assistant/infrastructure/parsers/markdown.py`
- Create: `src/knowledge_assistant/infrastructure/parsers/pdf.py`
- Create: `src/knowledge_assistant/infrastructure/parsers/docx.py`
- Create: `src/knowledge_assistant/infrastructure/parsers/dispatcher.py`
- Create: `src/knowledge_assistant/application/ingestion/service.py`
- Create: `tests/fixtures/documents/guide.md`
- Create: `tests/fixtures/documents/guide.pdf`
- Create: `tests/fixtures/documents/guide.docx`
- Create: `tests/unit/test_parsers.py`
- Create: `tests/unit/test_ingestion_service.py`
- Modify: `pyproject.toml`

**Interfaces:**
- `DocumentParser.supported_suffixes: tuple[str, ...]`.
- `DocumentParser.parse(path: Path) -> list[NormalizedDocument]`.
- `ParserDispatcher.for_path(path: Path) -> DocumentParser`.
- `IngestionService.ingest(root: Path) -> list[NormalizedDocument]`.

- [ ] **Step 1: Add parser dependencies**

Add `pymupdf`, `python-docx`, and `pydantic` to the main dependency group. Keep parsing independent from Qdrant and MLX.

- [ ] **Step 2: Write parser tests**

Test Markdown produces one document, PDF produces one document per page with `page_number`, DOCX extracts paragraphs, unsupported extensions raise a typed error, and malformed files are reported without aborting other files.

- [ ] **Step 3: Run parser tests**

Run: `poetry run pytest tests/unit/test_parsers.py -q`

Expected: FAIL until the parser adapters are implemented.

- [ ] **Step 4: Implement the parser adapters**

Normalize whitespace, preserve headings where available, compute SHA-256 content hashes, and use file modification time in metadata. Never silently parse an unsupported suffix.

- [ ] **Step 5: Implement ingestion orchestration**

Walk only the configured document root, sort paths for deterministic output, dispatch by suffix, collect per-file errors, and return normalized documents without indexing them.

- [ ] **Step 6: Run parser and ingestion tests**

Run: `poetry run pytest tests/unit/test_parsers.py tests/unit/test_ingestion_service.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_assistant/infrastructure/parsers src/knowledge_assistant/application/ingestion tests/fixtures tests/unit pyproject.toml
git commit -m "feat: ingest local pdf markdown and docx documents"
```

---

### Task 4: Add deterministic chunking and multilingual embeddings

**Files:**
- Create: `src/knowledge_assistant/application/ingestion/chunking.py`
- Create: `src/knowledge_assistant/infrastructure/embeddings/base.py`
- Create: `src/knowledge_assistant/infrastructure/embeddings/e5.py`
- Create: `tests/unit/test_chunking.py`
- Create: `tests/unit/test_embeddings.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`

**Interfaces:**
- `Chunker.chunk(document: NormalizedDocument) -> list[DocumentChunk]`.
- `EmbeddingProvider.embed_documents(chunks: list[str]) -> list[list[float]]`.
- `EmbeddingProvider.embed_query(query: str) -> list[float]`.
- Default embedding model: `intfloat/multilingual-e5-small`.

- [ ] **Step 1: Write chunking tests**

Test that chunks preserve document identity, have deterministic IDs, respect configured maximum characters, and retain page/source metadata.

- [ ] **Step 2: Run chunking tests**

Run: `poetry run pytest tests/unit/test_chunking.py -q`

Expected: FAIL before the chunker exists.

- [ ] **Step 3: Implement deterministic chunking**

Split on paragraph boundaries first, use a fixed overlap, and fall back to sentence/character boundaries for oversized paragraphs. Generate `chunk_id` from `document_id` and ordinal/content hash.

- [ ] **Step 4: Write embedding contract tests**

Use a fake provider in unit tests to verify query/document prefixes and vector shape without downloading a model. Do not make network/model downloads part of unit tests.

- [ ] **Step 5: Implement the E5 provider**

Prefix documents with `passage:` and queries with `query:`, expose model ID and dimension, and batch embedding calls. Load the model lazily so `--help` and tests do not initialize ML models.

- [ ] **Step 6: Run tests**

Run: `poetry run pytest tests/unit/test_chunking.py tests/unit/test_embeddings.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_assistant/application/ingestion/chunking.py src/knowledge_assistant/infrastructure/embeddings tests/unit pyproject.toml .env.example
git commit -m "feat: add deterministic chunking and multilingual embeddings"
```

---

### Task 5: Implement local Qdrant indexing and retrieval

**Files:**
- Create: `src/knowledge_assistant/infrastructure/vector_store/base.py`
- Create: `src/knowledge_assistant/infrastructure/vector_store/qdrant.py`
- Create: `src/knowledge_assistant/application/retrieval/service.py`
- Create: `tests/unit/test_retrieval_service.py`
- Create: `tests/integration/test_local_qdrant.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`

**Interfaces:**
- `VectorStore.upsert(chunks: list[DocumentChunk], vectors: list[list[float]]) -> int`.
- `VectorStore.search(query_vector: list[float], top_k: int) -> list[RetrievedChunk]`.
- `RetrievalService.index(documents: list[NormalizedDocument]) -> IndexSummary`.
- `RetrievalService.search(question: str, top_k: int = 5) -> list[RetrievedChunk]`.

- [ ] **Step 1: Write retrieval service tests with a fake vector store**

Verify that indexing chunks calls the embedding provider once per batch, search returns descending scores, and unchanged content does not create duplicate points.

- [ ] **Step 2: Run unit tests**

Run: `poetry run pytest tests/unit/test_retrieval_service.py -q`

Expected: FAIL before the service exists.

- [ ] **Step 3: Implement Qdrant local adapter**

Use `QdrantClient(path=...)` for local persistence. Create one collection with the embedding dimension from the provider, cosine distance, and payload fields for all citation metadata.

- [ ] **Step 4: Implement idempotent upsert**

Use deterministic point IDs derived from `chunk_id`; upserting the same chunk replaces the existing point. Store the embedding model ID in payload metadata.

- [ ] **Step 5: Implement search service**

Embed the query once, retrieve top-k points, map payloads back to `RetrievedChunk`, and reject results whose stored embedding model ID differs from the active provider.

- [ ] **Step 6: Run integration tests**

Run: `poetry run pytest tests/unit/test_retrieval_service.py tests/integration/test_local_qdrant.py -q`

Expected: PASS without Docker or a network Qdrant server.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_assistant/infrastructure/vector_store src/knowledge_assistant/application/retrieval tests pyproject.toml .env.example
git commit -m "feat: add local qdrant indexing and retrieval"
```

---

### Task 6: Add MLX Qwen generation and grounded answer service

**Files:**
- Create: `src/knowledge_assistant/infrastructure/inference/base.py`
- Create: `src/knowledge_assistant/infrastructure/inference/mlx_qwen.py`
- Create: `src/knowledge_assistant/application/generation/prompts.py`
- Create: `src/knowledge_assistant/application/generation/service.py`
- Create: `tests/unit/test_generation_service.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`

**Interfaces:**
- `TextGenerator.generate(prompt: str, max_tokens: int, temperature: float) -> str`.
- `GenerationService.answer(question: str, top_k: int = 5) -> AnswerResponse`.
- The prompt must instruct the model to answer only from context and return an explicit unknown response when context is insufficient.

- [ ] **Step 1: Write generation tests with fake retrieval and fake generator**

Verify context is bounded, citations are retained in the response, the no-context path returns the configured unknown answer, and prompt construction never includes a document outside retrieved chunks.

- [ ] **Step 2: Run generation tests**

Run: `poetry run pytest tests/unit/test_generation_service.py -q`

Expected: FAIL before the service exists.

- [ ] **Step 3: Implement the prompt builder**

Use a stable Vietnamese system instruction, numbered context blocks, source labels, and a fixed answer schema. Limit context by characters/tokens before generation.

- [ ] **Step 4: Implement the MLX adapter lazily**

Load `mlx_lm.load` only on the first generation call. Keep model ID, max tokens, temperature, and model cache path in typed settings. Do not import MLX during domain/application unit tests.

- [ ] **Step 5: Implement grounded answer orchestration**

Call retrieval, build the bounded prompt, generate the answer, attach citations from the selected chunks, and return latency metadata. If retrieval returns no usable context, do not invoke the model.

- [ ] **Step 6: Run tests**

Run: `poetry run pytest tests/unit/test_generation_service.py -q`

Expected: PASS without loading Qwen weights.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_assistant/application/generation src/knowledge_assistant/infrastructure/inference tests/unit pyproject.toml .env.example
git commit -m "feat: add grounded qwen generation through mlx"
```

---

### Task 7: Build the CLI vertical slice

**Files:**
- Create: `src/knowledge_assistant/application/commands.py`
- Create: `cli/main.py`
- Create: `tests/unit/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- `knowledge-assistant ingest PATH`.
- `knowledge-assistant index PATH`.
- `knowledge-assistant query QUESTION --top-k 5`.
- `knowledge-assistant evaluate PATH`.

- [ ] **Step 1: Write CLI tests with injected fake services**

Verify `--help` does not load MLX, invalid document roots produce non-zero exit status, `ingest` prints a deterministic summary, and `query` prints answer plus citations.

- [ ] **Step 2: Run CLI tests**

Run: `poetry run pytest tests/unit/test_cli.py -q`

Expected: FAIL before the Typer app exists.

- [ ] **Step 3: Implement dependency construction**

Create one application factory that reads settings and wires parser, chunker, embedding, vector store, retrieval, and generation providers. Keep CLI command functions thin.

- [ ] **Step 4: Implement commands**

Use exit codes for failures, human-readable output by default, and `--json` for machine-readable summaries. Never print secrets or full document contents.

- [ ] **Step 5: Run tests and help check**

Run: `poetry run pytest tests/unit/test_cli.py -q` and `poetry run knowledge-assistant --help`.

Expected: tests PASS and help lists the four commands without model initialization.

- [ ] **Step 6: Commit**

```bash
git add cli src/knowledge_assistant/application/commands.py tests/unit/test_cli.py pyproject.toml README.md
git commit -m "feat: add knowledge assistant cli"
```

---

### Task 8: Expose the FastAPI service

**Files:**
- Create: `api/schemas.py`
- Create: `api/main.py`
- Create: `tests/integration/test_api.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- `GET /health` returns `{ "status": "ok", "model_ready": bool }`.
- `POST /documents/ingest` accepts `{ "path": "relative/path" }` and returns an ingestion summary.
- `POST /query` accepts `{ "question": str, "top_k": int = 5 }` and returns `AnswerResponse`.

- [ ] **Step 1: Write API tests with an injected fake application service**

Test health response, query validation for empty questions and invalid `top_k`, successful query response with citations, and path traversal rejection for ingestion.

- [ ] **Step 2: Run API tests**

Run: `poetry run pytest tests/integration/test_api.py -q`

Expected: FAIL before the FastAPI app exists.

- [ ] **Step 3: Implement schemas and app factory**

Use Pydantic request/response schemas and an `create_app(services=None)` factory so tests do not load MLX or real Qdrant.

- [ ] **Step 4: Implement endpoints**

Resolve incoming paths against the configured document root, reject paths that escape it, delegate all work to application services, and map domain errors to stable HTTP status codes.

- [ ] **Step 5: Run API tests**

Run: `poetry run pytest tests/integration/test_api.py -q`.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api tests/integration README.md pyproject.toml
git commit -m "feat: expose knowledge assistant fastapi api"
```

---

### Task 9: Add evaluation fixtures and developer documentation

**Files:**
- Create: `src/knowledge_assistant/application/evaluation/service.py`
- Create: `tests/fixtures/evaluation/questions.jsonl`
- Create: `tests/unit/test_evaluation_service.py`
- Create: `docs/architecture.md`
- Create: `docs/roadmap.md`
- Create: `PROJECT_CONTEXT.md`
- Modify: `README.md`

**Interfaces:**
- Evaluation record fields: `question`, `expected_source`, `retrieved_sources`, `hit_at_k`, `answer`, `latency_ms`.
- `EvaluationService.run(dataset_path: Path, top_k: int = 5) -> EvaluationSummary`.

- [ ] **Step 1: Write evaluation tests**

Test hit@k calculation, source normalization, empty retrieval behavior, and aggregate summary calculation using fake retrieval/generation services.

- [ ] **Step 2: Run evaluation tests**

Run: `poetry run pytest tests/unit/test_evaluation_service.py -q`

Expected: FAIL before the evaluation service exists.

- [ ] **Step 3: Implement evaluation**

Read JSONL records, run retrieval/answer services, compare normalized source paths, calculate hit@k and average latency, and write JSON output without logging answer contents by default.

- [ ] **Step 4: Write architecture and context docs**

Document commands, boundaries, configuration, current phase, known limitations, testing rules, and the rule that every behavior change needs a focused test. `PROJECT_CONTEXT.md` must be short enough for an agent to read at task start.

- [ ] **Step 5: Update README**

Replace the old LLM Twin quickstart with local setup, MLX model setup, document directory layout, CLI/API examples, evaluation command, and explicit MVP limitations. Link the legacy project separately.

- [ ] **Step 6: Run the full local test suite**

Run: `poetry run pytest tests/unit tests/integration -q`.

Expected: PASS; tests must not require AWS, OpenAI, Docker, or downloaded model weights.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_assistant/application/evaluation tests/fixtures docs PROJECT_CONTEXT.md README.md
git commit -m "docs: add evaluation and developer context"
```

---

### Task 10: Final verification and handoff

**Files:**
- Modify: `README.md` only if verification finds stale commands or paths.
- Modify: `docs/architecture.md` only if implementation differs from the design.

- [ ] **Step 1: Run static checks**

Run: `poetry run ruff check src cli api tests` and `poetry run ruff format --check src cli api tests`.

Expected: both commands PASS.

- [ ] **Step 2: Run the full test suite**

Run: `poetry run pytest tests/unit tests/integration -q`.

Expected: all tests PASS without external services.

- [ ] **Step 3: Verify command surface**

Run: `poetry run knowledge-assistant --help` and `poetry run uvicorn api.main:app --help`.

Expected: both commands exit successfully without loading model weights.

- [ ] **Step 4: Review the final diff**

Run: `git diff HEAD~1 --stat`, `git status --short`, and `git log --oneline -10`.

Expected: no accidental secrets, generated model files, or untracked runtime indexes are committed; working tree is clean after any final documentation commit.

- [ ] **Step 5: Commit any verification-only documentation correction**

```bash
git add README.md docs/architecture.md
git commit -m "docs: finalize local assistant setup"
```

