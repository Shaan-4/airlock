"""The real-model code path, exercised over actual HTTP against a fake
OpenAI-compatible server (no API key or network access needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from airlock.agents import AirlockAgent
from airlock.bench import TASKS, build_workspace
from airlock.llm import LLMError, OpenAICompatLLM, parse_json


@pytest.fixture
def fake_openai():
    replies, requests = [], []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            content = replies.pop(0)
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", replies, requests
    srv.shutdown()


def test_airlock_with_http_model(fake_openai):
    url, replies, requests = fake_openai
    task = next(t for t in TASKS if t.id == "reply_invoice")
    # Models like to wrap JSON in prose and code fences; the parser copes.
    replies.append("Here is the plan:\n```json\n" + json.dumps(task.plan) + "\n```")
    replies.append('{"value": "billing@acme.example"}')
    llm = OpenAICompatLLM(url, "test-model", api_key="k")
    ws = build_workspace(task, None)
    r = AirlockAgent(llm, llm).run(task.prompt, ws)
    assert task.check(ws, r.answer)
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["auth"] == "Bearer k"
    assert requests[0]["body"]["temperature"] == 0
    # The planner request contains the task and no workspace content.
    planner_text = json.dumps(requests[0]["body"]["messages"])
    assert task.prompt in planner_text and "invoice #4471 for Rs 12,400" not in planner_text
    # The quarantine request carries the email as DATA.
    assert "invoice #4471 for Rs 12,400" in json.dumps(requests[1]["body"]["messages"])


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_unreachable_server_is_a_clean_error():
    with pytest.raises(LLMError):
        OpenAICompatLLM("http://127.0.0.1:9/v1", "m", timeout=2).complete([{"role": "user", "content": "hi"}])
