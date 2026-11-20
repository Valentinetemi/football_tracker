"""Small local HTTP server for the STEM2Connect replay dashboard."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    static_directory: Path
    video_path: Path
    replay_data_path: Path


class InvalidRange(ValueError):
    """Raised when an HTTP byte range cannot be satisfied."""


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range into an inclusive start/end pair."""
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise InvalidRange("Only one byte range is supported.")
    bounds = value.removeprefix("bytes=").split("-", maxsplit=1)
    if len(bounds) != 2:
        raise InvalidRange("Malformed byte range.")
    start_text, end_text = bounds
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise InvalidRange("Suffix byte range must be positive.")
            start = max(0, size - suffix_length)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as exc:
        raise InvalidRange("Malformed byte range.") from exc
    if start < 0 or start >= size or end < start:
        raise InvalidRange("Byte range is outside the file.")
    return start, min(end, size - 1)


class ReplayDashboardHandler(BaseHTTPRequestHandler):
    server_version = "STEM2ConnectReplay/1.0"

    def __init__(self, *args, settings: DashboardSettings, **kwargs) -> None:
        self.settings = settings
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=False)

    def _serve(self, *, send_body: bool) -> None:
        route = urlparse(self.path).path
        if route == "/health":
            payload = json.dumps({"status": "ok"}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if send_body:
                self.wfile.write(payload)
            return
        if route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        static_files = {
            "/": (self.settings.static_directory / "index.html", "text/html; charset=utf-8"),
            "/index.html": (
                self.settings.static_directory / "index.html",
                "text/html; charset=utf-8",
            ),
            "/app.css": (self.settings.static_directory / "app.css", "text/css; charset=utf-8"),
            "/app.mjs": (
                self.settings.static_directory / "app.mjs",
                "text/javascript; charset=utf-8",
            ),
            "/replay-state.mjs": (
                self.settings.static_directory / "replay-state.mjs",
                "text/javascript; charset=utf-8",
            ),
            "/match.mp4": (self.settings.video_path, "video/mp4"),
            "/replay-data.json": (self.settings.replay_data_path, "application/json"),
        }
        target = static_files.get(route)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path, content_type = target
        self._send_file(
            path,
            content_type=content_type,
            allow_range=route == "/match.mp4",
            send_body=send_body,
        )

    def _send_file(
        self,
        path: Path,
        *,
        content_type: str,
        allow_range: bool,
        send_body: bool,
    ) -> None:
        try:
            size = path.stat().st_size
            byte_range = parse_byte_range(self.headers.get("Range"), size) if allow_range else None
        except InvalidRange:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        start, end = byte_range if byte_range is not None else (0, size - 1)
        status = HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return

        remaining = end - start + 1
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


def create_server(
    settings: DashboardSettings,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    handler = partial(ReplayDashboardHandler, settings=settings)
    return ThreadingHTTPServer((host, port), handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local STEM2Connect replay dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="local interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="local port (default: 8765)")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("outputs/stem2connect-annotated-60s.mp4"),
        help="annotated match video",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/stem2connect-replay-data.json"),
        help="per-frame replay JSON",
    )
    parser.add_argument(
        "--static-directory",
        type=Path,
        default=Path("dashboard"),
        help="dashboard asset directory (default: dashboard)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = DashboardSettings(
        static_directory=args.static_directory.resolve(),
        video_path=args.video.resolve(),
        replay_data_path=args.data.resolve(),
    )
    required = (
        settings.static_directory / "index.html",
        settings.static_directory / "app.css",
        settings.static_directory / "app.mjs",
        settings.static_directory / "replay-state.mjs",
        settings.video_path,
        settings.replay_data_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing dashboard file(s): " + ", ".join(missing))
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")

    server = create_server(settings, host=args.host, port=args.port)
    actual_port = server.server_address[1]
    print(f"STEM2Connect replay dashboard: http://{args.host}:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
