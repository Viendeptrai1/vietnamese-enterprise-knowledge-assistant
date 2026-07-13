from collections.abc import Sequence

from knowledge_assistant.infrastructure.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    model_id = "fake-multilingual-model"
    dimension = 3

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(chunks)]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query)), 0.0, 1.0]


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        values = list(texts)
        self.calls.append((values, batch_size))
        return [[float(index), 0.5, 1.0] for index, _ in enumerate(values)]


def test_embedding_provider_contract_returns_vectors_with_declared_dimension() -> None:
    provider = FakeEmbeddingProvider()

    document_vectors = provider.embed_documents(["first", "second"])
    query_vector = provider.embed_query("where is the policy?")

    assert provider.model_id == "fake-multilingual-model"
    assert len(document_vectors) == 2
    assert all(len(vector) == provider.dimension for vector in document_vectors)
    assert len(query_vector) == provider.dimension


def test_e5_provider_prefixes_inputs_batches_calls_and_loads_lazily() -> None:
    from knowledge_assistant.infrastructure.embeddings.e5 import E5EmbeddingProvider

    model = FakeSentenceTransformer()
    load_calls = 0

    def load_model(_: str) -> FakeSentenceTransformer:
        nonlocal load_calls
        load_calls += 1
        return model

    provider = E5EmbeddingProvider(batch_size=2, model_loader=load_model)

    assert provider.model_id == "intfloat/multilingual-e5-small"
    assert provider.dimension == 384
    assert load_calls == 0

    document_vectors = provider.embed_documents(["một", "hai", "ba"])
    query_vector = provider.embed_query("tìm chính sách")

    assert document_vectors == [[0.0, 0.5, 1.0], [1.0, 0.5, 1.0], [2.0, 0.5, 1.0]]
    assert query_vector == [0.0, 0.5, 1.0]
    assert load_calls == 1
    assert model.calls == [
        (["passage: một", "passage: hai", "passage: ba"], 2),
        (["query: tìm chính sách"], 2),
    ]
