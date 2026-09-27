"""
Serve web/ on localhost for the classroom demo.

    python tools/serve.py            # http://localhost:8000
    python tools/serve.py --port 9000 --no-open

A plain static server, with caching turned off so an edit to app.js shows up on
reload, and correct types for .wav and .json.
"""
import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".json": "application/json",
        ".js": "text/javascript",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "200" not in (args[1] if len(args) > 1 else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not (WEB / "index.html").exists():
        raise SystemExit("no {0}".format(WEB / "index.html"))
    if not (WEB / "clips.json").exists():
        print("! web/clips.json is missing -- the page will tell you how to build it")

    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(Handler, directory=str(WEB))
    url = "http://localhost:{0}".format(args.port)
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print("serving {0}\n  {1}\nctrl-c to stop".format(WEB, url))
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
