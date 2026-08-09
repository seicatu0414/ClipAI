import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LlmProvider(Protocol):
    name: str

    def generate(self, prompt: str, *, model: str) -> str: ...


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, timeout_seconds: float = 300) -> None:
        self._url = f"{base_url.rstrip('/')}/api/generate"
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, model: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            }
        ).encode()
        request = Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error
        generated = result.get("response")
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError("Ollama returned no generated response")
        return generated
