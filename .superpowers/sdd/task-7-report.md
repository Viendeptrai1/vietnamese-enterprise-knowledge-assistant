# Task 7 Report: Build the CLI Vertical Slice

## Status

Complete. Implemented the Typer CLI vertical slice (`cli/main.py`) with `ingest`, `index`, `query`, and `evaluate` commands. Implemented the application use case factory and container (`src/knowledge_assistant/application/commands.py:ApplicationContainer`) that cleanly decouples model initialization so `--help` and tests never load MLX or E5 models unless required.

## Commit Hashes

- `d011999` — `feat: add knowledge assistant cli`

## Files Changed

- `cli/main.py`
- `src/knowledge_assistant/application/commands.py`
- `tests/unit/test_cli.py`
- `pyproject.toml`
- `poetry.lock`
- `README.md`

## Verification

### Unit test suite for CLI vertical slice
Command:
```console
poetry run pytest tests/unit/test_cli.py -q
```
Output:
```text
......                                                                   [100%]
6 passed, 5 warnings in 0.93s
```

### CLI Help Check
Command:
```console
poetry run knowledge-assistant --help
```
Output:
```text
 Usage: knowledge-assistant [OPTIONS] COMMAND [ARGS]...                         
                                                                                
 Vietnamese Enterprise Knowledge Assistant CLI based on Qwen3.5-2B and local    
 Qdrant.                                                                        
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ ingest     Ingest local documents (PDF, Markdown, DOCX) into normalized      │
│            domain models.                                                    │
│ index      Chunk and index ingested documents into local Qdrant vector       │
│            database.                                                         │
│ query      Retrieve grounded answer and citations from local knowledge base. │
│ evaluate   Run evaluation harness against dataset path.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Full Test Suite & Linter Check
Command:
```console
poetry run pytest -q && poetry run ruff check src/knowledge_assistant cli tests/unit tests/integration
```
Output:
```text
41 passed, 5 warnings in 1.04s
All checks passed!
```

## Self-Review
- Verified `knowledge-assistant --help` executes in milliseconds without initializing MLX (`mlx-lm`) or downloading Sentence Transformers models.
- Verified that `--json` output flag outputs machine-readable summaries across `ingest`, `index`, `query`, and `evaluate`.
- Verified that `ApplicationContainer` dependency injection allows unit tests to inject fakes for zero-network execution.
