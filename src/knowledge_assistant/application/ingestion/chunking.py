import re
from hashlib import sha256

from knowledge_assistant.domain.chunks import DocumentChunk
from knowledge_assistant.domain.documents import NormalizedDocument


class Chunker:
    """Split normalized documents into deterministic, metadata-preserving chunks."""

    def __init__(self, *, max_characters: int = 1_000, overlap_characters: int = 100) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")
        if not 0 <= overlap_characters < max_characters:
            raise ValueError("overlap_characters must be between 0 and max_characters - 1")
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters

    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        segments = self._segments(document.content)
        contents = self._pack(segments)
        return [
            DocumentChunk(
                chunk_id=self._chunk_id(document.metadata.document_id, ordinal, content),
                document_id=document.metadata.document_id,
                content=content,
                metadata=document.metadata,
                ordinal=ordinal,
            )
            for ordinal, content in enumerate(contents)
        ]

    def _segments(self, content: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content) if paragraph.strip()]
        return [piece for paragraph in paragraphs for piece in self._split_oversized(paragraph)]

    def _split_oversized(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self._max_characters:
            return [paragraph]

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self._max_characters:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(self._character_pieces(sentence))
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self._max_characters:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _character_pieces(self, text: str) -> list[str]:
        step = self._max_characters - self._overlap_characters
        return [text[start : start + self._max_characters] for start in range(0, len(text), step)]

    def _pack(self, segments: list[str]) -> list[str]:
        contents: list[str] = []
        current = ""
        for segment in segments:
            if current and self._overlap_characters and segment.startswith(current[-self._overlap_characters :]):
                contents.append(current)
                current = segment
                continue
            candidate = f"{current}\n\n{segment}".strip() if current else segment
            if len(candidate) <= self._max_characters:
                current = candidate
                continue
            if current:
                contents.append(current)
            overlap = current[-self._overlap_characters :].strip() if self._overlap_characters else ""
            available_overlap = self._max_characters - len(segment) - (2 if overlap else 0)
            overlap = overlap[-available_overlap:].strip() if available_overlap > 0 else ""
            current = f"{overlap}\n\n{segment}".strip() if overlap else segment
        if current:
            contents.append(current)
        return contents

    @staticmethod
    def _chunk_id(document_id: str, ordinal: int, content: str) -> str:
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        return f"{document_id}:chunk:{ordinal}:{content_hash}"
