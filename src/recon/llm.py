"""Thin Claude client wrapper.

Every LLM-backed feature degrades to a deterministic fallback when no API key is
configured, so the whole pipeline still runs end-to-end on a fresh clone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-4-20250514"


@dataclass
class LLMConfig:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = 700
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        )


class LLMUnavailable(RuntimeError):
    pass


class ClaudeClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self.available:
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        client = self._ensure_client()
        message = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    def complete_json(self, system: str, prompt: str) -> dict:
        raw = self.complete(system, prompt)
        return extract_json(raw)


def extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])
