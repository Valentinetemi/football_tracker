"""Small local HTTP server for the STEM2Connect replay dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC_ASSETS = ("index.html", "app.css", "app.mjs", "replay-state.mjs")
DEFAULT_VIDEO = Path("outputs/stem2connect-annotated-60s.mp4")
DEFAULT_REPLAY_DATA = Path("outputs/stem2connect-replay-data.json")
BROWSER_FRIENDLY_CODECS = frozenset({"avc1", "h264", "x264", "avc3"})


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    static_directory: Path
    video_path: Path
    replay_data_path: Path


class InvalidRange(ValueError):
    """Raised when an HTTP byte range cannot be satisfied."""


class DashboardInputError(Exception):
    """Raised when the dashboard cannot start because a required file is missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class MissingDashboardFiles:
    static_files: tuple[Path, ...]
    video_path: Path | None
    replay_data_path: Path | None

    @property
    def has_any(self) -> bool:
        return bool(self.static_files or self.video_path or self.replay_data_path)


def find_missing_dashboard_files(settings: DashboardSettings) -> MissingDashboardFiles:
    static_files = tuple(
        path
        for name in STATIC_ASSETS
        if not (path := settings.static_directory / name).is_file()
    )
    video_path = settings.video_path if not settings.video_path.is_file() else None
    replay_data_path = (
        settings.replay_data_path if not settings.replay_data_path.is_file() else None
    )
    return MissingDashboardFiles(static_files, video_path, replay_data_path)


def format_missing_dashboard_files(missing: MissingDashboardFiles) -> str:
    """Explain a fresh-clone failure and how to point the server at real files."""
    lines = [
        "The replay dashboard cannot start because required files are missing.",
        "",
        "Missing:",
    ]
    for path in missing.static_files:
        lines.append(f"  - {path} (dashboard asset)")
    if missing.video_path is not None:
        lines.append(f"  - {missing.video_path} (annotated match video)")
    if missing.replay_data_path is not None:
        lines.append(f"  - {missing.replay_data_path} (per-frame replay JSON)")

    if missing.video_path is not None or missing.replay_data_path is not None:
        lines.extend(
            [
                "",
                "A fresh clone does not include the demo video or replay JSON. Those files",
                "are gitignored. Download them from the GitHub Release the project owner",
                "uploads (README section \"Demo files\") and place them at the default paths:",
                "",
                f"  {DEFAULT_VIDEO}",
                f"  {DEFAULT_REPLAY_DATA}",
                "",
                "Or pass files you already have:",
                "",
                "  python -m football_tracker.dashboard \\",
                "    --video path/to/annotated.mp4 \\",
                "    --data path/to/replay-data.json",
                "",
                "OpenCV's default mp4v output often will not play in a browser. Prefer an",
                "H.264 file written with --codec avc1 when this machine's OpenCV build",
                "supports it.",
            ]
        )
    if missing.static_files:
        lines.extend(
            [
                "",
                "Dashboard HTML and scripts live in the dashboard directory at the",
                "repository root. Run this command from the repository root, or pass",
                "--static-directory.",
            ]
        )
    return "\n".join(lines)


def ensure_dashboard_files(settings: DashboardSettings) -> None:
    missing = find_missing_dashboard_files(settings)
    if missing.has_any:
        raise DashboardInputError(format_missing_dashboard_files(missing))


def browser_playback_notice(codec: str | None) -> str | None:
    """Warn when a video codec is unlikely to play in the dashboard browser."""
    if codec is None:
        return None
    normalized = "".join(char for char in codec if char.isprintable()).strip().lower()
    if normalized in BROWSER_FRIENDLY_CODECS:
        return None
    shown = normalized or "unknown"
    return (
        f"Note: this video's codec is {shown!r}. OpenCV's default mp4v output often "
        "will not play in a browser. Prefer H.264 (--codec avc1) for the dashboard "
        "file when this machine's OpenCV build can write it."
    )


def read_video_codec(path: Path) -> str | None:
    """Return the four-character codec OpenCV reports, or None if it cannot."""
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    characters: list[str] = []
    for shift in range(4):
        value = (fourcc >> (8 * shift)) & 0xFF
        if value < 32 or value > 126:
            return None
        characters.append(chr(value))
    return "".join(characters)


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
        default=DEFAULT_VIDEO,
        help=(
            "annotated match video served at /match.mp4 "
            f"(default: {DEFAULT_VIDEO}). Use an H.264 file; mp4v often will not "
            "play in browsers."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_REPLAY_DATA,
        help=(
            "per-frame replay JSON served at /replay-data.json "
            f"(default: {DEFAULT_REPLAY_DATA})"
        ),
    )
    parser.add_argument(
        "--static-directory",
        type=Path,
        default=Path("dashboard"),
        help="dashboard asset directory (default: dashboard)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    settings = DashboardSettings(
        static_directory=args.static_directory.resolve(),
        video_path=args.video.resolve(),
        replay_data_path=args.data.resolve(),
    )
    try:
        ensure_dashboard_files(settings)
    except DashboardInputError as exc:
        print(exc.message, file=sys.stderr)
        return 2

    server = create_server(settings, host=args.host, port=args.port)
    actual_port = server.server_address[1]
    print(f"STEM2Connect replay dashboard: http://{args.host}:{actual_port}", flush=True)
    notice = browser_playback_notice(read_video_codec(settings.video_path))
    if notice:
        print(notice, flush=True)
    print(
        "Open a prepared moment with ?track=<id>&time=<seconds> on that URL. "
        "The README explains how to choose those values from the replay file.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
