"""
Task Planner Tool for Research Assistant

This module provides a TaskPlannerTool that uses the OpenAI chat completion
API with function-calling tools to reliably return a structured JSON plan
containing the tasks required to answer a research question.

It can be used as a stand-alone utility or integrated into orchestration
layers such as the MCPSimulator to replace the heuristic prompt currently
found in `high_level_plan`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    # The 1.x OpenAI SDK is preferred. It exposes the `OpenAI` client class.
    from openai import OpenAI  # type: ignore

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover – handled gracefully at runtime
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TaskType(str, Enum):
    """Enumeration of supported research task types."""

    SEARCH = "search"
    EXTRACT = "extract"
    CSV_ANALYSIS = "csv_analysis"
    SUMMARIZE = "summarize"
    SYNTHESIZE = "synthesize"
    REPORT = "report"
    FOLLOW_UP = "follow_up"


@dataclass
class PlannedTask:
    """Dataclass representing a single planned task."""

    task_type: TaskType
    description: str
    priority: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedTask":
        """Create a PlannedTask from a raw dict (e.g. parsed JSON)."""
        return cls(
            task_type=TaskType(data["task_type"]),
            description=data["description"],
            priority=int(data.get("priority", 3)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the task."""
        return {
            "task_type": self.task_type.value,
            "description": self.description,
            "priority": self.priority,
        }


class TaskPlannerTool:
    """LLM-powered planning helper that outputs a JSON plan.

    The tool relies on the OpenAI function-calling protocol to ensure the
    response is a well-formed JSON payload that can be parsed deterministically.
    """

    DEFAULT_MODEL_NAME = "gpt-4.1-nano"  # small, inexpensive default

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not found. Install with `pip install openai`."
            )

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Function tool definition following the OpenAI 1.x schema under "tools"
        self._planning_tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "create_plan",
                "description": (
                    "Break a research question into a set of high-level tasks. "
                    "Return the tasks in the order they should be executed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_type": {
                                        "type": "string",
                                        "description": "Type of task to perform",
                                        "enum": [t.value for t in TaskType],
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "What the task should accomplish",
                                    },
                                    "priority": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "1 = highest priority, 5 = lowest",
                                    },
                                },
                                "required": ["task_type", "description", "priority"],
                            },
                        }
                    },
                    "required": ["tasks"],
                },
            },
        }

    def plan(self, question: str) -> List[PlannedTask]:
        """Generate a list of :class:`PlannedTask` objects for *question*."""
        logger.info("Planning tasks for question: %s", question)

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a research task planner. "
                    "Given a research question, break it down into actionable tasks."
                ),
            },
            {
                "role": "user",
                "content": f"Research question: {question}",
            },
        ]

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "tools": [self._planning_tool],
            "tool_choice": "auto",
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**params)

        # The assistant *should* respond with a function call. Fallback to
        # content-based JSON if the tool protocol is not followed.
        message = response.choices[0].message

        raw_tasks: List[Dict[str, Any]]
        if getattr(message, "tool_calls", None):
            arguments = message.tool_calls[0].function.arguments  # type: ignore[attr-defined]
            # `arguments` is a JSON-encoded string
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError as exc:
                logger.error("Failed to decode function arguments: %s", exc)
                raise
            raw_tasks = payload.get("tasks", [])
        else:
            try:
                raw_tasks = json.loads(message.content or "[]")
            except json.JSONDecodeError as exc:
                logger.error("Assistant did not return valid JSON: %s", exc)
                raise

        planned_tasks = [PlannedTask.from_dict(t) for t in raw_tasks]
        logger.info("%d tasks planned", len(planned_tasks))
        return planned_tasks
