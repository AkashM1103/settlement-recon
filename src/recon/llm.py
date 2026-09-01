"""Thin Groq client wrapper.

Every LLM-backed feature degrades to a deterministic fallback when no API key is
configured, so the whole pipeline still runs end-to-end on a fresh clone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass
class LLMConfig:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = 700
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        )


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self.available:
                raise LLMUnavailable("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=self.config.api_key)
        return self._client

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            response_format={"type": "json_object"} if json_mode else None,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def complete_json(self, system: str, prompt: str) -> dict:
        return extract_json(self.complete(system, prompt, json_mode=True))


def extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])
