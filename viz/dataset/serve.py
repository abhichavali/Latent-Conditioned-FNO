"""Serve the repo (viz pages) plus the dataset under /data/, with HTTP Range so the viewer fetches one IC at a time.

On the GPU box:   python3 viz/dataset/serve.py --data /workspace/pdegen/out
On your machine:  ssh -N -L 8000:localhost:8000 pdebox     then open http://localhost:8000/viz/dataset/
Binds to localhost only; the SSH tunnel is the way in.
"""

import argparse
import io
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, data, **kw):
        self.data = data
        super().__init__(*a, **kw)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("/data/"):
            return os.path.join(self.data, os.path.normpath("/" + path[len("/data/"):]).lstrip("/"))
        return super().translate_path(path)

    def send_head(self):
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
        if a > b:
            self.send_error(416)
            return None
        with open(path, "rb") as f:
            f.seek(a)
            body = f.read(b - a + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return io.BytesIO(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir (contains index.jsonl and eq_XXXX/)")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    handler = partial(Handler, data=os.path.abspath(args.data), directory=os.path.abspath(args.root))
    print(f"http://localhost:{args.port}/viz/dataset/  (data: {args.data})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
