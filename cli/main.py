import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from knowledge_assistant.application.commands import (
    ApplicationContainer,
    run_evaluate,
    run_index,
    run_ingest,
    run_query,
)
from rich.console import Console

app = typer.Typer(
    name="knowledge-assistant",
    help="Vietnamese Enterprise Knowledge Assistant CLI based on Qwen2.5-1.5B-Instruct-4bit and local Qdrant.",
    no_args_is_help=True,
)
console = Console()

_container: Optional[ApplicationContainer] = None


def get_container() -> ApplicationContainer:
    global _container
    if _container is None:
        _container = ApplicationContainer()
    return _container


def set_container(container: Optional[ApplicationContainer]) -> None:
    global _container
    _container = container


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Path to document directory or file to ingest.")],
    json_output: Annotated[bool, typer.Option("--json", help="Output summary in JSON format.")] = False,
) -> None:
    """Ingest local documents (PDF, Markdown, DOCX) into normalized domain models."""
    try:
        result = run_ingest(path, container=get_container())
        if json_output:
            console.print(json.dumps(result, ensure_ascii=False))
        else:
            console.print(f"[bold green]Ingestion Summary:[/bold green] {result['root_path']}")
            console.print(f"Total documents processed: {result['total_documents']}")
            console.print(f"Successful: {result['successful']} | Failed: {result['failed']}")
            if result["failures"]:
                console.print("[red]Failures:[/red]")
                for f in result["failures"]:
                    console.print(f"  - {f['source_path']}: {f['error']}")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Path to document directory or file to index.")],
    json_output: Annotated[bool, typer.Option("--json", help="Output summary in JSON format.")] = False,
) -> None:
    """Chunk and index ingested documents into local Qdrant vector database."""
    try:
        result = run_index(path, container=get_container())
        if json_output:
            console.print(json.dumps(result, ensure_ascii=False))
        else:
            console.print(f"[bold green]Indexing Summary:[/bold green] {result['root_path']}")
            console.print(f"Ingested documents: {result['ingested_documents']}")
            console.print(f"Total chunks generated: {result['total_chunks']}")
            console.print(f"Chunks upserted: {result['upserted_chunks']}")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask the knowledge base.")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of top chunks to retrieve.")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output summary in JSON format.")] = False,
) -> None:
    """Retrieve grounded answer and citations from local knowledge base."""
    try:
        result = run_query(question, top_k=top_k, container=get_container())
        if json_output:
            console.print(json.dumps(result, ensure_ascii=False))
        else:
            console.print(f"[bold cyan]Question:[/bold cyan] {result['question']}")
            console.print(f"[bold green]Answer:[/bold green] {result['answer']}")
            console.print(f"[dim]Latency: {result['latency_ms']} ms[/dim]")
            if result["citations"]:
                console.print("[bold yellow]Citations:[/bold yellow]")
                for idx, c in enumerate(result["citations"], start=1):
                    page = f" (Page {c['page_number']})" if c["page_number"] else ""
                    console.print(f"  [{idx}] {c['source_path']}{page}")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def evaluate(
    path: Annotated[Path, typer.Argument(help="Path to evaluation dataset JSONL file.")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of top chunks to retrieve for hit@k.")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output summary in JSON format.")] = False,
) -> None:
    """Run evaluation harness against dataset path."""
    try:
        result = run_evaluate(path, top_k=top_k, container=get_container())
        if json_output:
            console.print(json.dumps(result, ensure_ascii=False))
        else:
            console.print(f"[bold green]Evaluation Status:[/bold green] {result['status']}")
            console.print(f"Dataset path: {result['dataset_path']}")
            console.print(f"Total questions evaluated: {result.get('total_questions', 0)}")
            console.print(f"Hit@{top_k} rate: [bold cyan]{result.get('hit_at_k_rate', 0.0):.2%}[/bold cyan]")
            console.print(f"Average latency: [dim]{result.get('average_latency_ms', 0.0):.2f} ms[/dim]")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
