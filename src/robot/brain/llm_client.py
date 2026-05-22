from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Short connect timeout detects a dead server fast; long read timeout lets inference finish.
_PRIMARY_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
_FALLBACK_TIMEOUT_S = 120.0  # CPU inference is slower
# How long to wait before re-probing the primary server after a failure.
_REPROBLE_INTERVAL_S = 60.0


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
        timeout: httpx.Timeout | float = _FALLBACK_TIMEOUT_S,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
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


class FallbackOllamaClient:
    """
    Tries the primary (LAN) Ollama server on every request.
    Falls back to a local server when the primary is unreachable,
    and re-probes the primary every 60 seconds to switch back
    automatically when the LAN server comes back online.
    """

    def __init__(
        self,
        primary_url: str,
        primary_model: str,
        fallback_url: str,
        fallback_model: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._primary = OllamaClient(primary_url, primary_model, _PRIMARY_TIMEOUT, system_prompt)
        self._fallback = OllamaClient(fallback_url, fallback_model, _FALLBACK_TIMEOUT_S, system_prompt)
        self._primary_url = primary_url
        self._using_fallback = False
        self._last_failed_at: float = 0.0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        # Skip the primary while it's known-down, unless the reproble window has passed.
        primary_worth_trying = (
            not self._using_fallback
            or (time.monotonic() - self._last_failed_at >= _REPROBLE_INTERVAL_S)
        )

        if primary_worth_trying:
            try:
                result = await self._primary.chat(messages, tools)
                if self._using_fallback:
                    logger.info("llm: primary server back online — switching back to %s", self._primary_url)
                    self._using_fallback = False
                return result
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
                if not self._using_fallback:
                    logger.warning(
                        "llm: primary server unreachable — switching to local model"
                    )
                self._using_fallback = True
                self._last_failed_at = time.monotonic()

        logger.debug("llm: using local fallback model")
        try:
            return await self._fallback.chat(messages, tools)
        except httpx.HTTPStatusError as exc:
            logger.error("llm: fallback HTTP error %s — %s", exc.response.status_code, exc.response.text[:200])
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.error("llm: fallback server also unreachable — %s", exc)
            raise
