"""
Run the benchmark.

    python -m airlock.evaluate                 # scripted worst-case model
    python -m airlock.evaluate --model         # real model from AIRLOCK_LLM_* env vars

Reports, per agent:
  utility           tasks completed with no attack
  utility (attack)  tasks still completed while under attack
  ASR               attack success rate, over the (task, attack) pairs where
                    the attack can apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass

from .agents import AirlockAgent, ReActAgent
from .bench import ATTACKS, TASKS, build_workspace
from .llm import OpenAICompatLLM


@dataclass
class Cell:
    utility: bool
    attacked: bool
    blocked: bool


def make_agents(real: bool):
    if real:
        llm = OpenAICompatLLM.from_env()
        if llm is None:
            sys.exit("set AIRLOCK_LLM_BASE_URL and AIRLOCK_LLM_MODEL (and AIRLOCK_LLM_API_KEY if needed)")
        return {"react": lambda: ReActAgent(llm),
                "react+taint": lambda: ReActAgent(llm, enforce=True),
                "airlock": lambda: AirlockAgent(llm, llm)}
    from .scripted import ScriptedPlanner, ScriptedQuarantine, ScriptedReAct
    return {"react": lambda: ReActAgent(ScriptedReAct()),
            "react+taint": lambda: ReActAgent(ScriptedReAct(), enforce=True),
            "airlock": lambda: AirlockAgent(ScriptedPlanner(), ScriptedQuarantine())}


def evaluate(real: bool = False) -> dict:
    agents = make_agents(real)
    out = {}
    for name, factory in agents.items():
        clean, under, hits, pairs = 0, 0, 0, 0
        per_attack = defaultdict(lambda: [0, 0])
        per_task = {}
        for t in TASKS:
            ws = build_workspace(t, None)
            r = factory().run(t.prompt, ws)
            ok = t.check(ws, r.answer)
            clean += ok
            per_task[t.id] = ok
            for a in ATTACKS:
                if not a.applies_to(t):
                    continue
                ws = build_workspace(t, a)
                r = factory().run(t.prompt, ws)
                pairs += 1
                won = a.success(ws, r.answer)
                hits += won
                under += t.check(ws, r.answer)
                per_attack[a.id][0] += won
                per_attack[a.id][1] += 1
        out[name] = {
            "utility": clean / len(TASKS),
            "utility_under_attack": under / pairs,
            "asr": hits / pairs,
            "pairs": pairs,
            "per_attack": {k: v[0] / v[1] for k, v in per_attack.items()},
            "per_task_clean": per_task,
        }
    return out


def print_table(res: dict) -> None:
    pct = lambda x: f"{100 * x:.0f}%"
    print("| agent | utility (no attack) | utility (under attack) | attack success rate |")
    print("|---|---|---|---|")
    for name, r in res.items():
        print(f"| {name} | {pct(r['utility'])} | {pct(r['utility_under_attack'])} "
              f"| **{pct(r['asr'])}** ({r['pairs']} pairs) |")
    print("\nAttack success rate by attack:\n")
    names = list(res)
    print("| attack | kind | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for a in ATTACKS:
        cells = [pct(res[n]["per_attack"].get(a.id, 0)) for n in names]
        print(f"| {a.id} | {a.kind} | " + " | ".join(cells) + " |")
    print("\nTasks completed with no attack:\n")
    print("| task | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for t in TASKS:
        print(f"| {t.id} | " + " | ".join("yes" if res[n]["per_task_clean"][t.id] else "**no**"
                                          for n in names) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="use the real model from AIRLOCK_LLM_* env vars")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = evaluate(real=a.model)
    print(json.dumps(res, indent=2)) if a.json else print_table(res)
