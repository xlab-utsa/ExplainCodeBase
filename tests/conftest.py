import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from explaincodebase.graph import CallGraph
from explaincodebase.llm import Completion, Provider
from explaincodebase.parsing import parse_codebase

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
POLYGLOT = os.path.join(HERE, "fixtures", "polyglot")
INHERITANCE = os.path.join(ROOT, "input", "inheritance")


@pytest.fixture(scope="session")
def polyglot_graph():
    return CallGraph(parse_codebase(POLYGLOT))


@pytest.fixture(scope="session")
def inheritance_graph():
    return CallGraph(parse_codebase(INHERITANCE))


def graph_from(tmp_path, files: dict[str, str]) -> CallGraph:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return CallGraph(parse_codebase(str(tmp_path)))


class RecordingProvider(Provider):
    """Answers with a fixed function of the prompt and records every prompt."""

    name = "recording"
    model = "recording"

    def __init__(self, answer=lambda prompt: "A summary."):
        self.answer = answer
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def complete(self, prompt, system):
        with self.lock:
            self.prompts.append(prompt)
        return Completion(self.answer(prompt), 10, 5)


class FakeVLLM:
    """A tiny OpenAI-compatible server standing in for ``vllm serve``."""

    def __init__(self, model="Qwen/Qwen3-8B", max_model_len=8192):
        self.model = model
        self.max_model_len = max_model_len
        self.requests: list[dict] = []
        self.fail_next = 0
        self.reply = lambda body: "<think>Let me look at the code.</think>\nIt adds two numbers."
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send(200, {"object": "list", "data": [{
                        "id": fake.model, "object": "model", "owned_by": "vllm",
                        "max_model_len": fake.max_model_len}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                if fake.fail_next:
                    fake.fail_next -= 1
                    self._send(429, {"error": {"message": "rate limited"}})
                    return
                content = fake.reply(body)
                self._send(200, {
                    "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


@pytest.fixture
def fake_vllm(monkeypatch):
    for key in ("VLLM_MODEL", "VLLM_API_BASE", "VLLM_MAX_MODEL_LEN", "VLLM_MAX_TOKENS",
                "VLLM_TEMPERATURE", "ECB_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    # Talk to the local fake server directly, not through any HTTP proxy.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with FakeVLLM() as server:
        monkeypatch.setenv("VLLM_API_BASE", server.url)
        yield server
