from typing import Sequence

from knowledge_assistant.domain.retrieval import RetrievedChunk

SYSTEM_INSTRUCTION = """Bạn là trợ lý tri thức doanh nghiệp Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào các đoạn thông tin (context) được cung cấp bên dưới.
Quy tắc trả lời:
1. Chỉ sử dụng thông tin có trong phần 'NGỮ CẢNH ĐƯỢC CUNG CẤP'. Không tự bịa đặt hay suy diễn thông tin ngoài ngữ cảnh.
2. Nếu ngữ cảnh không có thông tin hoặc không đủ thông tin để trả lời câu hỏi, hãy trả lời chính xác câu sau: 'Tôi không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi này.'
3. Trả lời rõ ràng, chính xác bằng tiếng Việt."""

UNKNOWN_ANSWER = "Tôi không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi này."


def build_bounded_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    max_context_chars: int = 3000,
) -> tuple[str, list[RetrievedChunk]]:
    """Build a Vietnamese RAG prompt bounded by character length and return selected chunks."""
    selected_chunks: list[RetrievedChunk] = []
    context_blocks: list[str] = []
    current_chars = 0

    for idx, chunk in enumerate(chunks, start=1):
        page_info = f", trang {chunk.metadata.page_number}" if chunk.metadata.page_number is not None else ""
        block = f"[Đoạn {idx}] Nguồn: {chunk.metadata.source_path}{page_info}\nNội dung: {chunk.content.strip()}"
        block_len = len(block) + 2  # + newlines
        if current_chars + block_len > max_context_chars and selected_chunks:
            break
        context_blocks.append(block)
        selected_chunks.append(chunk)
        current_chars += block_len

    context_text = "\n\n".join(context_blocks)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- NGỮ CẢNH ĐƯỢC CUNG CẤP ---\n"
        f"{context_text}\n\n"
        f"--- CÂU HỎI ---\n"
        f"{question.strip()}\n\n"
        f"--- CÂU TRẢ LỜI ---\n"
    )
    return prompt, selected_chunks
