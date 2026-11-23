
"""Generate responses using a locally hosted Ollama language model."""

from dataclasses import dataclass

import ollama

from src.rag.prompt_builder import PromptPackage


@dataclass(frozen=True)
class GenerationConfig:
    """Store generation parameters for the language model."""

    temperature: float = 0.0
    num_predict: int = 1200


class LLMGenerator:
    """Generate text responses using the configured Ollama model."""

    def __init__(
        self,
        model_name: str = "qwen3:4b",
        client: ollama.Client | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        """Initialize the generator with an Ollama client and model."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name.strip()
        self.client = client if client is not None else ollama.Client()
        self.config = config or GenerationConfig()

        if not isinstance(self.config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig instance.")

        if not 0.0 <= self.config.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if (
            isinstance(self.config.num_predict, bool)
            or not isinstance(self.config.num_predict, int)
            or self.config.num_predict <= 0
        ):
            raise ValueError("num_predict must be a positive integer.")

    def generate(self, prompt_package: PromptPackage) -> str:
        """Generate a response from a validated prompt package."""

        if not isinstance(prompt_package, PromptPackage):
            raise TypeError(
                "prompt_package must be a PromptPackage instance."
            )

        if not prompt_package.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty.")

        if not prompt_package.user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=prompt_package.messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.num_predict,
                },
                think=False,
            )

        except Exception as exc:
            raise RuntimeError(
                f"LLM generation failed using model "
                f"'{self.model_name}': {exc}"
            ) from exc

        try:
            generated_text = response.message.content
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Ollama returned an invalid response structure."
            ) from exc

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError("The LLM returned an empty response.")

        return generated_text.strip()
