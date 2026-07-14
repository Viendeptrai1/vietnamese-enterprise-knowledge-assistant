import json
from dataclasses import dataclass
from pathlib import Path

from knowledge_assistant.application.generation.service import GenerationService


@dataclass(frozen=True)
class EvaluationRecord:
    question: str
    expected_source: str
    retrieved_sources: list[str]
    hit_at_k: bool
    answer: str
    latency_ms: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_questions: int
    hit_at_k_rate: float
    average_latency_ms: float
    records: list[EvaluationRecord]


class EvaluationService:
    def __init__(self, generation_service: GenerationService) -> None:
        self._generation_service = generation_service

    def run(self, dataset_path: Path, top_k: int = 5) -> EvaluationSummary:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")
        if not dataset_path.is_file():
            raise ValueError(f"Evaluation dataset path must be a file: {dataset_path}")

        records: list[EvaluationRecord] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = str(data.get("question", "")).strip()
                expected_source = str(data.get("expected_source") or data.get("source") or "").strip()
                if not question:
                    continue

                response = self._generation_service.answer(question, top_k=top_k)
                retrieved_sources = [chunk.metadata.source_path for chunk in response.retrieved_chunks]
                hit = self._is_hit(expected_source, retrieved_sources)

                record = EvaluationRecord(
                    question=question,
                    expected_source=expected_source,
                    retrieved_sources=retrieved_sources,
                    hit_at_k=hit,
                    answer=response.answer,
                    latency_ms=response.latency_ms,
                )
                records.append(record)

        total = len(records)
        if total == 0:
            return EvaluationSummary(
                total_questions=0,
                hit_at_k_rate=0.0,
                average_latency_ms=0.0,
                records=[],
            )

        hit_rate = sum(1 for r in records if r.hit_at_k) / total
        avg_latency = sum(r.latency_ms for r in records) / total
        return EvaluationSummary(
            total_questions=total,
            hit_at_k_rate=hit_rate,
            average_latency_ms=avg_latency,
            records=records,
        )

    @staticmethod
    def _is_hit(expected: str, retrieved_sources: list[str]) -> bool:
        if not expected:
            return False
        expected_norm = expected.replace("\\", "/").lower()
        expected_name = Path(expected_norm).name
        for src in retrieved_sources:
            src_norm = str(src).replace("\\", "/").lower()
            if expected_norm in src_norm or src_norm in expected_norm or Path(src_norm).name == expected_name:
                return True
        return False
