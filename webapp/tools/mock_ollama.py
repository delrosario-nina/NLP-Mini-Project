"""A fake Ollama server for testing the LLM path without installing Ollama.

HOW TO USE
    Terminal 1:
        python webapp/tools/mock_ollama.py --port 11435

    Terminal 2 (PowerShell):
        $env:OLLAMA_HOST = "http://localhost:11435"
        uvicorn app:app --app-dir webapp --reload --port 8000

    The health pill should read "Ollama ready" and LLM mode will stream a
    canned answer through the real ollama-python client and the real /api/chat
    code path. This is a protocol stand-in, not a language model.

Endpoints implemented: GET /api/tags, GET /api/version, POST /api/chat.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "llama3.2:3b"

FALLBACK = """MOCK OLLAMA — no real model is running.

This answer comes from webapp/tools/mock_ollama.py, so you can test the
streaming path, the answer renderer, and the error handling end to end.

1. Retrieval results above are real: they come from the recipe parquet.
2. The wording below the retrieved cards is canned, not generated.
3. Switch the answer mode to Demo to compare a deterministic answer."""


def answer_for(payload: dict) -> str:
    """Use the primary retrieved recipe name so the mock answer looks contextual."""
    messages = payload.get("messages") or []
    prompt = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )

    match = re.search(r"\[Primary recipe\]\s*(.+)", prompt) or re.search(
        r"\[Recipe 1\]\s*(.+)", prompt
    )
    if not match:
        return FALLBACK

    name = match.group(1).strip()
    return (
        "MOCK OLLAMA — no real model is running.\n\n"
        f"Recommended: {name}\n"
        "- This reply is streamed by webapp/tools/mock_ollama.py to test the UI.\n\n"
        "1. The retrieved recipe cards are real and come from the parquet dataset.\n"
        "2. The answer wording is canned; it does not read the recipe instructions.\n"
        "3. Switch to Demo mode to compare a deterministic, data-derived answer."
    )


def chunks(text: str, size: int = 10):
    for start in range(0, len(text), size):
        yield text[start : start + size]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console quiet
        return

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server naming)
        if self.path.startswith("/api/tags"):
            self._json(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "model": MODEL,
                            "modified_at": "2026-01-01T00:00:00Z",
                            "size": 123456789,
                            "digest": "mock",
                            "details": {"family": "mock", "parameter_size": "3B"},
                        }
                    ]
                }
            )
        elif self.path.startswith("/api/version"):
            self._json({"version": "0.0.0-mock"})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if not self.path.startswith("/api/chat"):
            self._json({"error": "not found"}, status=404)
            return

        text = answer_for(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        for piece in chunks(text):
            frame = {
                "model": MODEL,
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": piece},
                "done": False,
            }
            self.wfile.write((json.dumps(frame) + "\n").encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.01)

        final = {
            "model": MODEL,
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Ollama server for UI testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mock Ollama listening on http://{args.host}:{args.port}")
    print(f'Now run:  $env:OLLAMA_HOST = "http://{args.host}:{args.port}"')
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
