from urllib.error import URLError
from unittest.mock import patch

import pytest

from clipai.knowledge.provider import OllamaProvider


def test_ollama_provider_reports_connection_failure() -> None:
    with (
        patch("clipai.knowledge.provider.urlopen", side_effect=URLError("offline")),
        pytest.raises(RuntimeError, match="Ollama request failed"),
    ):
        OllamaProvider("http://ollama:11434").generate("prompt", model="model")
