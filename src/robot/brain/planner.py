from __future__ import annotations

"""
Reusable goal-directed planning loop.

The planner is the core AI reasoning engine — it knows nothing about navigation
specifically.  Any caller can hand it:
  - a goal string
  - a scene snapshot
  - a set of tool schemas
  - a dict of tool executors (name → async callable)

Phase 2 voice tasks ("find a red ball", "what do you see") will reuse this
same planner with different goals and tool sets.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from ..perception.scene import SceneSnapshot
from .nav_memory import NavMemory

if TYPE_CHECKING:
    from .llm_client import FallbackOllamaClient

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    success: bool
    summary: str
    actions: list[str] = field(default_factory=list)
    steps_taken: int = 0


class Planner:
    """
    Multi-turn LLM planning loop with tool execution.

    One call to run() drives a complete goal → plan → execute → done cycle.
    The caller provides:
      tool_schemas  — OpenAI-compatible tool definitions sent to the LLM
      executor      — dict mapping tool_name → async callable
                      NOTE: 'done' is handled internally; do not include it
                            in executor.
      on_speak      — optional callback(text) for narration outside the tool set
    """

    def __init__(self, llm: "FallbackOllamaClient", memory: NavMemory) -> None:
        self._llm = llm
        self._memory = memory

    async def run(
        self,
        *,
        goal: str,
        scene: SceneSnapshot,
        system_prompt: str,
        tool_schemas: list[dict[str, Any]],
        executor: dict[str, Callable],
        max_steps: int = 10,
        on_speak: Optional[Callable[[str], None]] = None,
    ) -> PlanResult:
        """
        Drive the planning loop until the LLM calls done() or max_steps is hit.
        CancelledError propagates cleanly (caller's finally block handles cleanup).
        """
        episodes = self._memory.recall(scene.sonar_range(), goal, n=3)
        memory_ctx = self._memory.format_for_prompt(episodes)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Situation: {scene.to_text()}\n"
                    f"Goal: {goal}\n"
                    f"Past experiences:\n{memory_ctx}"
                ),
            },
        ]

        actions_taken: list[str] = []

        for step in range(max_steps):
            logger.debug("planner: step %d/%d goal=%r", step + 1, max_steps, goal)

            try:
                msg = await self._llm.chat(messages, tools=tool_schemas)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("planner: LLM call failed at step %d", step + 1)
                return PlanResult(success=False, summary="LLM error", steps_taken=step + 1, actions=actions_taken)

            # Echo assistant message into history so the LLM has context.
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or [],
            })

            tool_calls = msg.get("tool_calls") or []

            # No tool calls — LLM replied in text; treat as natural completion.
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                if text and on_speak:
                    on_speak(text)
                return PlanResult(
                    success=True,
                    summary=text or "completed",
                    steps_taken=step + 1,
                    actions=actions_taken,
                )

            tool_results: list[dict] = []
            done_result: Optional[PlanResult] = None

            for call in tool_calls:
                fn_info = call.get("function", {})
                name = fn_info.get("name", "")
                raw_args = fn_info.get("arguments") or {}

                # Some small models return arguments as a JSON string.
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                # Small models sometimes nest args under "parameters".
                if "parameters" in raw_args and isinstance(raw_args["parameters"], dict):
                    raw_args = raw_args["parameters"]

                if name == "done":
                    done_result = PlanResult(
                        success=bool(raw_args.get("success", True)),
                        summary=raw_args.get("summary", ""),
                        steps_taken=step + 1,
                        actions=actions_taken,
                    )
                    tool_results.append({"role": "tool", "content": "Navigation complete."})
                    continue

                if name not in executor:
                    logger.warning("planner: unknown tool %r — skipped", name)
                    tool_results.append({"role": "tool", "content": f"Error: unknown tool '{name}'"})
                    continue

                logger.debug("planner: executing %r(%r)", name, raw_args)
                try:
                    result = await executor[name](**raw_args)
                    action_str = f"{name}({', '.join(f'{k}={v!r}' for k, v in raw_args.items())})"
                    if name not in ("speak",):  # don't clutter memory with speak calls
                        actions_taken.append(action_str)
                    result_text = json.dumps(result) if isinstance(result, dict) else str(result or "ok")
                    tool_results.append({"role": "tool", "content": result_text})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("planner: tool %r raised", name)
                    tool_results.append({"role": "tool", "content": f"Error: {exc}"})

            messages.extend(tool_results)

            if done_result is not None:
                return done_result

        return PlanResult(
            success=False,
            summary="max steps reached without completing goal",
            steps_taken=max_steps,
            actions=actions_taken,
        )
