import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from knowledge_assistant.application.commands import ApplicationContainer
from knowledge_assistant.domain.answers import AnswerResponse

from api.schemas import (
    HealthResponse,
    IngestionFailureSchema,
    IngestionRequest,
    IngestionSummaryResponse,
    QueryRequest,
)


def create_app(
    container: ApplicationContainer | None = None,
    document_root: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Vietnamese Enterprise Knowledge Assistant API",
        description="Local RAG API powered by Apple Silicon MLX and Qdrant.",
        version="0.1.0",
    )
    app.state.container = container or ApplicationContainer()
    root_str = document_root or os.environ.get("DOCUMENT_ROOT", "data/documents")
    app.state.document_root = Path(root_str).resolve()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        is_ready = request.app.state.container is not None
        return HealthResponse(status="ok", model_ready=is_ready)

    @app.post("/documents/ingest", response_model=IngestionSummaryResponse)
    def ingest_documents(req: IngestionRequest, request: Request) -> IngestionSummaryResponse:
        base_root: Path = request.app.state.document_root
        if Path(req.path).is_absolute():
            raise HTTPException(status_code=400, detail="Path must be relative to document root.")

        target_path = (base_root / req.path).resolve()
        try:
            target_path.relative_to(base_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Path traversal outside document root rejected.") from None

        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Target path does not exist: {req.path}")

        container: ApplicationContainer = request.app.state.container
        service = container.ingestion_service()
        try:
            documents = service.ingest(target_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        failures = [IngestionFailureSchema(source_path=str(f.path), error=str(f.message)) for f in service.errors]
        return IngestionSummaryResponse(
            root_path=str(target_path),
            total_documents=len(documents) + len(service.errors),
            successful=len(documents),
            failed=len(service.errors),
            failures=failures,
        )

    @app.post("/query", response_model=AnswerResponse)
    def query_endpoint(req: QueryRequest, request: Request) -> AnswerResponse:
        container: ApplicationContainer = request.app.state.container
        service = container.generation_service()
        try:
            response = service.answer(req.question, top_k=req.top_k)
            return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
