from abc import ABC, abstractmethod


class TextGenerator(ABC):
    """Contract for text generation implementations."""

    model_id: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate text from prompt up to max_tokens."""
