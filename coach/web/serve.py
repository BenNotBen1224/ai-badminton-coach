"""Stdlib static server for the viewer. Range requests, no cache, never port 8765."""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MIMES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mp4": "video/mp4",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

ROOT = Path(__file__).resolve().parent


def _parse(argv):
    port = 8772
    root = ROOT
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--port":
            port = int(argv[i + 1])
            i += 2
            continue
        root = Path(arg).resolve()
        i += 1
    return port, root


def _under(root: Path, url_path: str):
    rel = unquote(urlparse(url_path).path).lstrip("/\\")
    if not rel:
        return None
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


def _range(header, size):
    if not header or not header.startswith("bytes="):
        return None
    spec = header.split("=", 1)[1].split(",")[0].strip()
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    if start_s == "":
        if not end_s:
            return None
        length = int(end_s)
        if length <= 0:
            return None
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._send(body=True)

    def do_HEAD(self):
        self._send(body=False)

    def _send(self, body):
        path = _under(self.root, self.path)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        mime = MIMES.get(path.suffix.lower(), "application/octet-stream")
        wanted = _range(self.headers.get("Range"), size)
        if self.headers.get("Range") and wanted is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if wanted is None:
            start, end = 0, size - 1
            status = 200
        else:
            start, end = wanted
            status = 206
        length = 0 if size == 0 else end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body or length == 0:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(262144, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        return


def main(argv=None):
    port, root = _parse(sys.argv[1:] if argv is None else argv)
    if port == 8765:
        print("refusing port 8765", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"missing root {root}", file=sys.stderr)
        return 2
    Handler.root = root
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    bound = server.server_address[1]
    print(f"port {bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
