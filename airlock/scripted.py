"""
Deterministic stand-ins for an LLM, built from the benchmark's task
definitions. Both are "gullible": competent at the task, and obedient to any
injected instruction they read. All three agents use the same brains, so the
only difference between them is the architecture around the model.
"""
from __future__ import annotations

import json
import re

from .bench import ATTACKS, TASKS, Task
from .llm import parse_json

_RESULT = re.compile(r"^Tool result: (.*)$", re.S)


def _task_for(text: str) -> Task:
    for t in TASKS:
        if t.prompt in text:
            return t
    raise KeyError("scripted model doesn't know this task")


class ScriptedPlanner:
    """Writes the reference plan for a known task. Records what it was shown,
    so tests can prove it never saw untrusted content."""

    def __init__(self):
        self.seen: list[str] = []

    def complete(self, messages: list[dict]) -> str:
        text = "\n".join(m["content"] for m in messages)
        self.seen.append(text)
        return json.dumps(_task_for(text).plan)


class ScriptedQuarantine:
    """Answers extraction requests correctly, unless the data contains an
    injection that targets this type of value; then it returns the attacker's
    value. For free-text answers it also echoes the injected text, the way a
    real model summarising a poisoned email might."""

    def complete(self, messages: list[dict]) -> str:
        system, user = messages[0]["content"], messages[1]["content"]
        instr = re.search(r"Instruction: (.*)", system).group(1).strip()
        typ = re.search(r"Type: (.*)", system).group(1).strip()
        data_text = user.split("DATA:\n", 1)[1]
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            data = data_text
        for a in ATTACKS:
            if a.payload in data_text and typ in a.override:
                return json.dumps({"value": a.override[typ]})
        oracle = next(f for t in TASKS for st in t.plan["steps"] if st["tool"] == "extract"
                      and st["args"]["instruction"] == instr
                      for sid, f in t.oracles.items() if sid == st["id"])
        value = oracle(data)
        if typ == "string":
            injected = [a.payload for a in ATTACKS if a.payload in data_text]
            if injected:
                value = value + " " + " ".join(injected)
        return json.dumps({"value": value})


class ScriptedReAct:
    """Plays the ReAct protocol for a known task. Before each planned step it
    checks its context for injected instructions it hasn't carried out yet,
    and carries them out first (the worst case). Values it needs from untrusted
    text are taken from the injection when one targets them."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.emitted: list[tuple[str, str]] = []    # ("plan", step id) | ("attack", id)
        self.done_attacks: set[str] = set()

    def complete(self, messages: list[dict]) -> str:
        if len(messages) == 2:          # fresh conversation
            self.reset()
        task = _task_for(messages[1]["content"])
        ctx = "\n".join(m["content"] for m in messages[2:])

        results = []
        for m in messages[2:]:
            if m["role"] == "user":
                r = _RESULT.match(m["content"])
                results.append(json.loads(r.group(1)) if r else None)
        env = {}
        for (kind, ident), res in zip(self.emitted, results):
            if kind == "plan":
                env[ident] = res

        for a in ATTACKS:
            if a.call and a.id not in self.done_attacks and a.payload in ctx:
                self.done_attacks.add(a.id)
                self.emitted.append(("attack", a.id))
                tool, args = a.call(ctx)
                return json.dumps({"tool": tool, "args": args})

        done = {i for k, i in self.emitted if k == "plan"}
        for st in task.plan["steps"]:
            if st["tool"] == "extract":
                if st["id"] not in env and st["args"]["text"][1:] in env:
                    env[st["id"]] = self._extract(task, st, env, ctx)
                continue
            if st["id"] in done:
                continue
            args = {k: self._sub(v, env) for k, v in st["args"].items()}
            self.emitted.append(("plan", st["id"]))
            return json.dumps({"tool": st["tool"], "args": args})
        return json.dumps({"final": str(self._sub(task.plan.get("answer", "Done."), env))})

    def _extract(self, task, st, env, ctx):
        typ = st["args"]["type"]
        for a in ATTACKS:
            if a.payload in ctx and typ in a.override:
                return a.override[typ]
        value = task.oracles[st["id"]](env[st["args"]["text"][1:]])
        if typ == "string":
            injected = [a.payload for a in ATTACKS if a.payload in ctx]
            if injected:
                value = value + " " + " ".join(injected)
        return value

    def _sub(self, v, env):
        if isinstance(v, str) and v.startswith("$"):
            return env.get(v[1:])
        if isinstance(v, list):
            return [self._sub(x, env) for x in v]
        return v
