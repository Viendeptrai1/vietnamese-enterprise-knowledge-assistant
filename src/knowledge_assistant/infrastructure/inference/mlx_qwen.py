import importlib
from typing import Any

from knowledge_assistant.infrastructure.inference.base import TextGenerator


class MlxQwenGenerator(TextGenerator):
    """Lazy-loading MLX text generator adapter for Qwen models on Apple Silicon."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        model_cache_path: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.model_cache_path = model_cache_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                mlx_lm = importlib.import_module("mlx_lm")
            except ImportError as exc:
                raise RuntimeError(
                    "mlx_lm is not installed. Please install mlx-lm to run local MLX generation."
                ) from exc

            # Load model lazily on first generation call
            if self.model_cache_path:
                self._model, self._tokenizer = mlx_lm.load(
                    self.model_id, tokenizer_config={"cache_dir": self.model_cache_path}
                )
            else:
                self._model, self._tokenizer = mlx_lm.load(self.model_id)
        return self._model, self._tokenizer

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        model, tokenizer = self._load()
        try:
            mlx_lm = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise RuntimeError("mlx_lm is not installed.") from exc

        response = mlx_lm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return str(response).strip()
