"""A from-source Redshift Data API shim over Postgres: the four operations the helper uses (ExecuteStatement, DescribeStatement, GetStatementResult, CancelStatement) against redshift-local, so the UNMODIFIED helper runs locally through AWS_ENDPOINT_URL_REDSHIFT_DATA.

Verified against the local stack 2026-07-30 (helper through the shim, masking, and a full Claude Code run); the cloud path is still unrun.
"""

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import redshift_connector

_STATEMENTS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _connect():
    return redshift_connector.connect(
        host=os.environ.get("PGHOST", "redshift-local"),
        port=int(os.environ.get("PGPORT", "5439")),
        user=os.environ.get("PGUSER", "bi_analyst"),
        password=os.environ.get("PGPASSWORD", "local"),
        database=os.environ.get("DEFAULT_DB", "analytics"),
        # redshift-local's wire proxy presents a self-signed cert that verify-ca would reject; plain TCP is fine for a loopback shim, and the real Data API path never opens a driver socket.
        ssl=False,
    )


def _run(sid: str, sql: str) -> None:
    try:
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            if cur.description:
                cols = [
                    c[0].decode() if isinstance(c[0], bytes) else str(c[0])
                    for c in cur.description
                ]
                rows = cur.fetchall()
            else:
                cols, rows = [], []
        with _LOCK:
            _STATEMENTS[sid].update(
                Status="FINISHED",
                _cols=cols,
                _rows=rows,
                HasResultSet=bool(cols),
            )
    except (redshift_connector.Error, OSError, ValueError) as exc:
        # surfaced through DescribeStatement, the way the real Data API does
        with _LOCK:
            _STATEMENTS[sid].update(Status="FAILED", Error=str(exc))


def _field(value):
    if value is None:
        return {"isNull": True}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"longValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class Handler(BaseHTTPRequestHandler):
    """Serve the Redshift Data API surface the assistant calls."""

    def do_GET(self):  # health check
        """Answer the health check."""
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        """Dispatch one Data API operation."""
        target = self.headers.get("X-Amz-Target", "").split(".")[-1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        handler = {
            "ExecuteStatement": self._execute,
            "DescribeStatement": self._describe,
            "GetStatementResult": self._result,
            "CancelStatement": self._cancel,
        }.get(target)
        if handler is None:
            self._send(400, {"message": f"unsupported operation {target}"})
            return
        self._send(200, handler(body))

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/x-amz-json-1.1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _execute(self, body: dict) -> dict:
        sid = str(uuid.uuid4())
        with _LOCK:
            _STATEMENTS[sid] = {"Status": "STARTED", "HasResultSet": False}
        threading.Thread(target=_run, args=(sid, body["Sql"]), daemon=True).start()
        return {"Id": sid}

    def _describe(self, body: dict) -> dict:
        with _LOCK:
            st = dict(
                _STATEMENTS.get(body["Id"], {"Status": "FAILED", "Error": "unknown id"})
            )
        return {
            "Id": body["Id"],
            "Status": st["Status"],
            "HasResultSet": st.get("HasResultSet", False),
            **({"Error": st["Error"]} if "Error" in st else {}),
        }

    def _result(self, body: dict) -> dict:
        with _LOCK:
            st = _STATEMENTS[body["Id"]]
        return {
            "ColumnMetadata": [{"name": c} for c in st.get("_cols", [])],
            "Records": [[_field(v) for v in row] for row in st.get("_rows", [])],
            "TotalNumRows": len(st.get("_rows", [])),
        }

    def _cancel(self, body: dict) -> dict:
        with _LOCK:
            _STATEMENTS.get(body["Id"], {}).update(Status="ABORTED")
        return {"Status": True}

    def log_message(self, format: str, *args) -> None:  # quiet
        """Keep the server quiet."""
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8002), Handler).serve_forever()
