# Vietnamese Enterprise Knowledge Assistant — Design Spec

## 1. Mục tiêu

Chuyển repository hiện tại từ project LLM Twin của tác giả thành một project mới có tên **Vietnamese Enterprise Knowledge Assistant**. Project mới là một hệ thống RAG chạy local, tập trung vào tài liệu doanh nghiệp tiếng Việt và phù hợp với Mac M2 có 16 GB RAM.

MVP phải hỗ trợ:

- Nhập tài liệu local ở dạng PDF, Markdown và DOCX.
- Chuẩn hóa nội dung và metadata tài liệu.
- Chunking, embedding, indexing và retrieval.
- Sinh câu trả lời tiếng Việt bằng Qwen3.5-2B chạy qua MLX.
- Trả về citation/source của các đoạn context được sử dụng.
- Chạy được qua CLI và expose HTTP API bằng FastAPI.
- Có tests và evaluation cơ bản cho ingestion, retrieval và answer grounding.

## 2. Phạm vi không làm trong MVP

- Không crawl LinkedIn, Medium, GitHub hoặc web URL.
- Không dùng dữ liệu author/profile của LLM Twin làm domain chính.
- Không fine-tune Qwen3.5-2B trong giai đoạn đầu.
- Không yêu cầu AWS, SageMaker, Comet ML hoặc OpenAI để chạy core flow local.
- Không xây web UI trong MVP.
- Không triển khai multi-tenant hoặc enterprise authentication.

## 3. Chiến lược migration

Code và tài liệu của project cũ được giữ trong thư mục `legacy/` để tham khảo. Root repository sẽ chỉ chứa project mới và các tài nguyên dùng chung rõ ràng.

Migration không xóa dữ liệu cũ. Mỗi bước phải tạo một commit riêng, để có thể kiểm tra hoặc quay lại an toàn.

Các phần có thể tái sử dụng sẽ được đưa qua adapter hoặc copy có chọn lọc, không để domain mới phụ thuộc trực tiếp vào các tên gọi như `UserDocument`, `AuthorDocument`, `LLM Twin` hoặc crawler mạng xã hội.

## 4. Cấu trúc repository mục tiêu

```text
.
├── src/
│   └── knowledge_assistant/
│       ├── domain/
│       │   ├── documents.py
│       │   ├── chunks.py
│       │   ├── retrieval.py
│       │   └── answers.py
│       ├── application/
│       │   ├── ingestion/
│       │   ├── retrieval/
│       │   ├── generation/
│       │   └── evaluation/
│       └── infrastructure/
│           ├── parsers/
│           ├── embeddings/
│           ├── vector_store/
│           └── inference/
├── api/
│   └── main.py
├── cli/
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── documents/
│   └── indexes/
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   └── superpowers/specs/
├── legacy/
├── configs/
├── pyproject.toml
└── README.md
```

Các module mới phải có trách nhiệm rõ ràng:

- `domain`: entity, value object và interface nghiệp vụ; không phụ thuộc framework.
- `application`: use case như ingest, retrieve, answer và evaluate.
- `infrastructure`: implementation cụ thể của parser, embedding model, vector store và MLX.
- `cli`: giao diện dòng lệnh, chỉ gọi application use cases.
- `api`: HTTP adapter, không chứa business logic.

## 5. Data flow

```text
PDF / Markdown / DOCX
        ↓
Document parser
        ↓
Normalized document + metadata
        ↓
Chunker
        ↓
Embedding model
        ↓
Vector store
        ↓
User query
        ↓
Query embedding + retrieval + optional reranking
        ↓
Prompt with bounded context
        ↓
Qwen3.5-2B via MLX
        ↓
Vietnamese answer + citations + metadata
```

Mỗi chunk phải giữ tối thiểu:

- `document_id`
- `chunk_id`
- `source_path`
- `source_type`
- `page_number` nếu có
- `title` nếu có
- `content`
- `content_hash`
- `created_at` hoặc file modification time

Ingestion phải idempotent: cùng một file chưa thay đổi không tạo thêm bản index trùng lặp.

## 6. Model và runtime

- Generation model: `Qwen/Qwen3.5-2B`.
- Runtime: MLX/MLX-LM trên Apple Silicon.
- Embedding và reranking giữ dạng provider interface để có thể thay model mà không đổi application layer.
- Model phải được cấu hình qua environment/config, không hard-code trong use case.
- Core local flow không phụ thuộc OpenAI API.

MVP ưu tiên inference ổn định, context nhỏ và output có cấu trúc hơn là tối đa hóa context window. Fine-tuning chỉ được xem xét sau khi có evaluation baseline.

## 7. CLI và API contract

CLI dự kiến:

```text
knowledge-assistant ingest <path>
knowledge-assistant index
knowledge-assistant query "<question>"
knowledge-assistant evaluate
```

FastAPI dự kiến:

- `GET /health`: kiểm tra service và model readiness.
- `POST /documents/ingest`: nhận path local trong môi trường server và chạy ingestion.
- `POST /query`: nhận `{ "question": "...", "top_k": 5 }`.
- Response query gồm `answer`, `citations`, `retrieved_chunks` và metadata latency.

API không được tự ý đọc file ngoài configured document root.

## 8. Error handling và observability

- Parser lỗi phải báo rõ file nào lỗi và lý do, không làm mất toàn bộ batch.
- Query không có context đủ tin cậy phải trả lời rõ rằng không tìm thấy thông tin phù hợp.
- Không đưa toàn bộ document vào prompt; context phải bị giới hạn theo token/chunk budget.
- Log các bước ingestion, số lượng document/chunk, retrieval latency và generation latency.
- Không log nội dung nhạy cảm mặc định.

## 9. Testing và evaluation

Unit tests:

- Parser PDF/Markdown/DOCX.
- Metadata normalization.
- Chunking và content hash.
- Retrieval result ordering.
- Citation formatting.
- Prompt construction.

Integration tests:

- Ingest fixture documents vào vector store local.
- Query fixture knowledge base.
- FastAPI health/query endpoints.

Evaluation dataset ban đầu là một tập nhỏ câu hỏi tiếng Việt có ground-truth source. Metrics tối thiểu:

- Retrieval hit@k.
- Citation/source correctness.
- Answer groundedness bằng rule/checker thủ công ở giai đoạn đầu.
- Latency và token/output size.

## 10. Definition of Done cho MVP

- Người dùng copy PDF/Markdown/DOCX vào document directory.
- Một CLI command ingest và index được các file đó.
- Một CLI command trả lời câu hỏi tiếng Việt bằng Qwen3.5-2B local.
- Câu trả lời có citation đến file và page/chunk khi khả dụng.
- FastAPI expose được health check và query endpoint.
- Có fixture, unit tests và integration test cơ bản.
- README mô tả setup local, commands, architecture và limitations.
- Không cần AWS hoặc API key trả phí để chạy core demo.

## 11. Thứ tự triển khai

1. Tạo `legacy/` và di chuyển code/data cũ vào đó.
2. Tạo skeleton package mới và cập nhật project metadata.
3. Implement domain entities và parser interfaces.
4. Implement PDF/Markdown/DOCX ingestion.
5. Implement chunking, embeddings và vector store adapter.
6. Implement retrieval và citation model.
7. Tích hợp Qwen3.5-2B qua MLX.
8. Thêm CLI.
9. Thêm FastAPI.
10. Thêm evaluation, tests, README và developer context/memory.

