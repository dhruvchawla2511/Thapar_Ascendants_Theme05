"""Minimal Ollama HTTP client (standard library only, no extra dependencies).

Configuration (environment variables, all optional):
    OLLAMA_HOST     default http://127.0.0.1:11434
    OLLAMA_MODEL    default qwen2.5:14b
    OLLAMA_TIMEOUT  seconds, default 120 (the first call loads the model)
    OLLAMA_NUM_CTX  context window tokens, default 8192

Every failure is converted into a subclass of LLMError so callers can handle
"Ollama is down / slow / missing the model / sent garbage" without crashing.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Protocol

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_TIMEOUT = 120.0
DEFAULT_NUM_CTX = 8192


class LLMError(Exception):
    """Base class for every LLM-related failure."""


class LLMUnavailableError(LLMError):
    """Ollama is not reachable (not running, wrong host, connection refused)."""


class LLMTimeoutError(LLMError):
    """Ollama did not answer within the timeout."""


class LLMModelNotFoundError(LLMError):
    """The requested model is not installed in Ollama."""


class LLMResponseError(LLMError):
    """Ollama answered, but the answer was malformed or an HTTP error."""


class LLMClient(Protocol):
    """Anything with this method can act as the LLM (real client or a test fake)."""

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str: ...


def normalize_host(host: str) -> str:
    """'127.0.0.1:11434' -> 'http://127.0.0.1:11434' (no trailing slash)."""
    host = (host or "").strip() or DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
        return value if value > 0 else default
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
        return value if value > 0 else default
    except ValueError:
        return default


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.2,
    ) -> None:
        self.host = normalize_host(host)
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> "OllamaClient":
        return cls(
            host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
            model=os.environ.get("OLLAMA_MODEL", "").strip() or DEFAULT_MODEL,
            timeout=_float_env("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT),
            num_ctx=_int_env("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX),
        )

    # ---- public API ----------------------------------------------------

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"

        body = self._request("POST", "/api/chat", payload, self.timeout)
        try:
            data = json.loads(body)
        except (ValueError, TypeError) as exc:
            raise LLMResponseError(f"Ollama returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMResponseError("Ollama returned an unexpected response shape.")
        if data.get("error"):
            raise self._error_from_message(str(data["error"]))
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama response is missing message.content.")
        return content

    def is_available(self) -> bool:
        """True if the Ollama server answers. Never raises."""
        try:
            self._request("GET", "/api/tags", None, 3.0)
            return True
        except LLMError:
            return False

    def model_installed(self) -> bool:
        """True if the configured model is installed. Never raises."""
        try:
            data = json.loads(self._request("GET", "/api/tags", None, 3.0))
            names = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
        except (LLMError, ValueError, AttributeError, TypeError):
            return False
        wanted = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
        return bool(names & wanted)

    # ---- internals -----------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None, timeout: float) -> str:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.host + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - body is best-effort only
                text = ""
            raise self._error_from_http(exc.code, text) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {timeout:.0f}s."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(
                    f"Ollama did not respond within {timeout:.0f}s."
                ) from exc
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.host} ({exc.reason}). "
                "Start it with: ollama serve"
            ) from exc
        except OSError as exc:  # connection reset etc.
            raise LLMUnavailableError(f"Connection to Ollama failed: {exc}") from exc

    def _error_from_http(self, code: int, text: str) -> LLMError:
        message = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("error"):
                message = str(parsed["error"])
        except ValueError:
            pass
        if code == 404 or "not found" in message.lower():
            return self._model_missing()
        return LLMResponseError(f"Ollama returned HTTP {code}: {message or 'no details'}")

    def _error_from_message(self, message: str) -> LLMError:
        if "not found" in message.lower():
            return self._model_missing()
        return LLMResponseError(f"Ollama error: {message}")

    def _model_missing(self) -> LLMModelNotFoundError:
        return LLMModelNotFoundError(
            f"Model '{self.model}' is not available. Install it with: ollama pull {self.model}"
        )
