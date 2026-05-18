from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0


class OllamaClient:
    """
    Async client for the Ollama /api/chat endpoint.

    Usage::

        client = OllamaClient("http://192.168.50.100:11434", model="llama3.2:3b")
        msg = await client.chat(messages=[{"role": "user", "content": "Hello"}])
        print(msg["content"])
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        POST /api/chat and return the assistant message dict.

        The returned dict has at least a "role" key and one or both of:
          "content"    — the spoken reply
          "tool_calls" — list of {function: {name, arguments}} dicts

        Raises httpx.HTTPError on network/API failure.
        """
        full_messages = []
        if self._system_prompt:
            full_messages.append({"role": "system", "content": self._system_prompt})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        logger.debug("llm: POST /api/chat model=%s messages=%d", self._model, len(full_messages))

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        msg = data.get("message", {})
        logger.debug(
            "llm: response content=%r tool_calls=%d",
            (msg.get("content") or "")[:80],
            len(msg.get("tool_calls") or []),
        )
        return msg
