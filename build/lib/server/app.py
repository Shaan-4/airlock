"""Demo site: pick a task and an attack, watch three agents handle it."""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from airlock.bench import ATTACKS, TASKS, build_workspace
from airlock.evaluate import evaluate, make_agents

app = FastAPI(title="Airlock", version="0.1.0")
INDEX = Path(__file__).with_name("index.html")


class RunReq(BaseModel):
    task: str
    attack: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX.read_text(encoding="utf-8")


@app.get("/api/scenarios")
def scenarios():
    return {
        "tasks": [{"id": t.id, "prompt": t.prompt, "reads_untrusted": t.inject is not None} for t in TASKS],
        "attacks": [{"id": a.id, "kind": a.kind, "payload": a.payload,
                     "applies_to": [t.id for t in TASKS if a.applies_to(t)]} for a in ATTACKS],
    }


@app.post("/api/run")
def run(req: RunReq):
    task = next((t for t in TASKS if t.id == req.task), None)
    if task is None:
        raise HTTPException(404, "unknown task")
    attack = next((a for a in ATTACKS if a.id == req.attack), None) if req.attack else None
    if req.attack and attack is None:
        raise HTTPException(404, "unknown attack")
    if attack and not attack.applies_to(task):
        raise HTTPException(400, "that attack doesn't apply to this task")
    out = {}
    for name, factory in make_agents(real=False).items():
        ws = build_workspace(task, attack)
        r = factory().run(task.prompt, ws)
        out[name] = {
            "task_done": task.check(ws, r.answer),
            "attack_succeeded": bool(attack and attack.success(ws, r.answer)),
            "blocked": r.blocked,
            "answer": r.answer,
            "trace": [asdict(e) for e in r.trace],
        }
    return {"task": task.prompt, "injected": attack.payload if attack else None,
            "injected_into": task.inject, "agents": out}


@lru_cache(maxsize=1)
def _benchmark():
    return evaluate()


@app.get("/api/benchmark")
def benchmark():
    return _benchmark()


@app.get("/healthz")
def healthz():
    return {"ok": True}
