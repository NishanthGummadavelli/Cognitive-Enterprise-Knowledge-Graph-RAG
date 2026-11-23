
"""Tests for the Ollama-based LLM generator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.rag.llm_generator import GenerationConfig, LLMGenerator
from src.rag.prompt_builder import PromptPackage


@pytest.fixture
def mock_client() -> MagicMock:
    """Provide a mocked Ollama client."""

    client = MagicMock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content="Employees receive 20 days of annual leave. [S1]"
        )
    )
    return client


@pytest.fixture
def prompt_package() -> PromptPackage:
    """Provide a representative prompt package."""

    return PromptPackage(
        system_prompt="Answer using only the supplied documents.",
        user_prompt="How many days of annual leave are provided?",
    )


def test_generator_initialization(mock_client: MagicMock) -> None:
    """Initialize the generator with the configured model."""

    generator = LLMGenerator(
        model_name="qwen3:4b",
        client=mock_client,
    )

    assert generator.model_name == "qwen3:4b"
    assert generator.client is mock_client


def test_default_generation_config() -> None:
    """Verify the default generation parameters."""

    config = GenerationConfig()

    assert config.temperature == 0.0
    assert config.num_predict == 1200


def test_generate_success(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Generate and return a valid LLM response."""

    generator = LLMGenerator(client=mock_client)

    result = generator.generate(prompt_package)

    assert result == "Employees receive 20 days of annual leave. [S1]"
    mock_client.chat.assert_called_once()


def test_generate_passes_correct_model_and_messages(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Pass the configured model and prompt messages to Ollama."""

    generator = LLMGenerator(
        model_name="qwen3:4b",
        client=mock_client,
    )

    generator.generate(prompt_package)

    call_kwargs = mock_client.chat.call_args.kwargs

    assert call_kwargs["model"] == "qwen3:4b"
    assert call_kwargs["messages"] == prompt_package.messages
    assert call_kwargs["think"] is False


def test_generate_passes_generation_options(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Pass the configured generation parameters to Ollama."""

    config = GenerationConfig(
        temperature=0.2,
        num_predict=800,
    )

    generator = LLMGenerator(
        client=mock_client,
        config=config,
    )

    generator.generate(prompt_package)

    options = mock_client.chat.call_args.kwargs["options"]

    assert options["temperature"] == 0.2
    assert options["num_predict"] == 800


def test_generate_strips_response_whitespace(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Remove unnecessary whitespace from generated responses."""

    mock_client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="  Valid answer.  ")
    )

    generator = LLMGenerator(client=mock_client)

    result = generator.generate(prompt_package)

    assert result == "Valid answer."


def test_rejects_invalid_prompt_package(
    mock_client: MagicMock,
) -> None:
    """Reject objects that are not PromptPackage instances."""

    generator = LLMGenerator(client=mock_client)

    with pytest.raises(TypeError):
        generator.generate("invalid")


def test_rejects_empty_system_prompt(
    mock_client: MagicMock,
) -> None:
    """Reject prompts without system instructions."""

    generator = LLMGenerator(client=mock_client)

    package = PromptPackage(
        system_prompt=" ",
        user_prompt="Valid question",
    )

    with pytest.raises(ValueError):
        generator.generate(package)


def test_rejects_empty_user_prompt(
    mock_client: MagicMock,
) -> None:
    """Reject prompts without a user message."""

    generator = LLMGenerator(client=mock_client)

    package = PromptPackage(
        system_prompt="Valid system instructions",
        user_prompt=" ",
    )

    with pytest.raises(ValueError):
        generator.generate(package)


def test_handles_ollama_connection_error(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Convert Ollama connection failures into clear runtime errors."""

    mock_client.chat.side_effect = ConnectionError(
        "Ollama server is unavailable."
    )

    generator = LLMGenerator(client=mock_client)

    with pytest.raises(RuntimeError, match="LLM generation failed"):
        generator.generate(prompt_package)


def test_handles_invalid_response_structure(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Reject an invalid response returned by Ollama."""

    mock_client.chat.return_value = object()

    generator = LLMGenerator(client=mock_client)

    with pytest.raises(
        RuntimeError,
        match="invalid response structure",
    ):
        generator.generate(prompt_package)


def test_rejects_empty_llm_response(
    mock_client: MagicMock,
    prompt_package: PromptPackage,
) -> None:
    """Reject empty model output."""

    mock_client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="   ")
    )

    generator = LLMGenerator(client=mock_client)

    with pytest.raises(RuntimeError, match="empty response"):
        generator.generate(prompt_package)


def test_invalid_model_name_is_rejected(
    mock_client: MagicMock,
) -> None:
    """Reject an empty model name."""

    with pytest.raises(ValueError):
        LLMGenerator(
            model_name=" ",
            client=mock_client,
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_invalid_temperature_is_rejected(
    mock_client: MagicMock,
    temperature: float,
) -> None:
    """Reject temperatures outside the supported range."""

    with pytest.raises(ValueError):
        LLMGenerator(
            client=mock_client,
            config=GenerationConfig(temperature=temperature),
        )


@pytest.mark.parametrize(
    "num_predict",
    [0, -1, True],
)
def test_invalid_num_predict_is_rejected(
    mock_client: MagicMock,
    num_predict: int,
) -> None:
    """Reject invalid output-token limits."""

    with pytest.raises(ValueError):
        LLMGenerator(
            client=mock_client,
            config=GenerationConfig(num_predict=num_predict),
        )


def test_invalid_config_type_is_rejected(
    mock_client: MagicMock,
) -> None:
    """Reject configuration objects of the wrong type."""

    with pytest.raises(TypeError):
        LLMGenerator(
            client=mock_client,
            config="invalid",
        )
