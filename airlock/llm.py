"""
LLM backends. Everything in airlock talks to a model through one method:

    complete(messages: list[{"role", "content"}]) -> str

Two implementations:
  * OpenAICompatLLM: any server with an OpenAI-style /chat/completions
    endpoint (a local Ollama, or a hosted provider). Configured by env vars.
  * the scripted models in scripted.py, used by tests, the benchmark and the
    demo, so nothing here needs a paid API key.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class LLMError(Exception):
    pass


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatLLM | None":
        base, model = os.getenv("AIRLOCK_LLM_BASE_URL"), os.getenv("AIRLOCK_LLM_MODEL")
        if not (base and model):
            return None
        return cls(base, model, os.getenv("AIRLOCK_LLM_API_KEY"))

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM server returned {e.code}: {e.read()[:300]!r}")
        except urllib.error.URLError as e:
            raise LLMError(f"could not reach LLM server: {e.reason}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(data)[:300]}")


def parse_json(text: str):
    """Models often wrap JSON in prose or ``` fences. Take the first JSON
    object in the reply; anything else is a protocol error."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in model reply: {text[:200]!r}")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise LLMError(f"malformed JSON from model: {e}")
    return obj
