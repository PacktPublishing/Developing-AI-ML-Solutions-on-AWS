"""Anthropic-API shim for the offline mode.

Injects think="low" into every request: gpt-oss floods otherwise (measured
2,000+ hidden reasoning tokens per question, minutes at laptop decode
speed). Everything else passes through to Ollama untouched, including the
small utility model Claude Code uses alongside the main one.
Verified with the local stack 2026-08-01.
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://host.docker.internal:11434")


class Handler(BaseHTTPRequestHandler):
    """Forward Anthropic API calls to the local model server."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        """Forward a GET unchanged."""
        self._forward(None)

    def do_POST(self):
        """Forward a POST body unchanged."""
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(body)
            payload.pop("thinking", None)
            payload["think"] = "low"
            body = json.dumps(payload).encode()
        except (json.JSONDecodeError, AttributeError):
            pass
        self._forward(body)

    def _forward(self, body):
        req = urllib.request.Request(
            UPSTREAM + self.path, data=body, method=self.command
        )
        for header in ("Content-Type", "X-Api-Key", "Anthropic-Version", "Accept"):
            if self.headers.get(header):
                req.add_header(header, self.headers[header])
        try:
            upstream = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as exc:
            upstream = exc
        except urllib.error.URLError:
            data = json.dumps(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "model server down"},
                }
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = upstream.read()
        self.send_response(upstream.code)
        self.send_header(
            "Content-Type", upstream.headers.get("Content-Type", "application/json")
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        """Keep the proxy quiet."""
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8091), Handler).serve_forever()
