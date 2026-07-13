# Task 4 Report: Deterministic Chunking and Multilingual Embeddings

## Delivered

- Added `Chunker.chunk(document)` with deterministic SHA-256 chunk identifiers derived from the document ID, ordinal, and chunk content.
- Chunks preserve the normalized document ID and immutable document metadata, including source path and page number.
- Chunking packs paragraph boundaries first, falls back to sentence boundaries, then character boundaries for oversized text. Every emitted chunk obeys the configured maximum-character limit; overlap is bounded to preserve that limit.
- Added the abstract `EmbeddingProvider` contract and the `E5EmbeddingProvider` adapter for `intfloat/multilingual-e5-small` (384 dimensions).
- E5 prefixes document chunks with `passage:` and queries with `query:`. Model construction is lazy, and callers can inject a loader for isolated tests.
- Added `EMBEDDING_MODEL_ID=intfloat/multilingual-e5-small` to `.env.example`. `sentence-transformers` was already declared in `pyproject.toml`, so no dependency or lockfile change was required.

## Test-Driven Evidence

1. Added the new focused tests before implementation and ran them in the red state. Collection failed because `knowledge_assistant.infrastructure.embeddings` did not exist.
2. The initial green run revealed the zero-overlap-budget edge case (`text[-0:]` returns the complete string in Python), allowing an oversize chunk. The existing maximum-character test reproduced it; the implementation now explicitly produces no overlap when that budget is zero.
3. Focused verification: `poetry run pytest tests/unit/test_chunking.py tests/unit/test_embeddings.py -q` — 4 passed.
4. Regression verification: `poetry run pytest -q` — 21 passed (5 pre-existing third-party deprecation warnings from SWIG bindings).
5. Static verification: Ruff check and format check passed for the Task 4 source and tests; `git diff --check` passed.

## Self-Review

- Confirmed all required interfaces and default E5 model ID are present.
- Confirmed the E5 module imports `sentence_transformers` only inside the lazy loader, so construction and unit tests do not load a model or use the network.
- Confirmed vector prefixing, batch-size forwarding, model metadata, deterministic IDs, configured size bound, source/page metadata, and fake-provider test isolation are covered.
- No unrelated working-tree changes were reverted or included.

## Commit

- `7e0ab9b feat: add deterministic chunking and multilingual embeddings`

## Review Fixes

- Replaced non-overlapping oversized character slices with fixed-stride windows using `max_characters - overlap_characters`. Fallback chunks now prove suffix-to-prefix overlap and remain within the configured maximum, including the final short window.
- Added regression coverage for paragraph packing, oversized fallback overlap and bounds, deterministic output, document identity, source/page metadata, and content-hash preservation.
- `E5EmbeddingProvider` now resolves its default model from `EMBEDDING_MODEL_ID` at construction time, falling back to `intfloat/multilingual-e5-small`; an explicit `model_id` still takes precedence. Model loading remains lazy.
- Added vector-width validation against the declared 384-dimensional E5 contract and an isolated fake-model test for the failure case. Tests use injected loaders and never download a model.
- Focused verification: `poetry run pytest tests/unit/test_chunking.py tests/unit/test_embeddings.py -q` — 8 passed.
- Full unit verification: `poetry run pytest -q` — 25 passed with 5 pre-existing SWIG deprecation warnings.
- Static verification: focused Ruff check, Ruff format check, and `git diff --check` passed.
