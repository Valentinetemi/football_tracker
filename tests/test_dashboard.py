from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from football_tracker.dashboard import (
    DashboardSettings,
    InvalidRange,
    browser_playback_notice,
    create_server,
    ensure_dashboard_files,
    main,
    parse_byte_range,
    read_video_codec,
)


class ByteRangeTests(unittest.TestCase):
    def test_parses_closed_open_and_suffix_ranges(self) -> None:
        self.assertEqual(parse_byte_range("bytes=2-5", 10), (2, 5))
        self.assertEqual(parse_byte_range("bytes=7-", 10), (7, 9))
        self.assertEqual(parse_byte_range("bytes=-4", 10), (6, 9))

    def test_rejects_unsatisfied_or_multiple_ranges(self) -> None:
        for value in ("bytes=10-", "bytes=5-2", "bytes=0-1,4-5", "items=0-1"):
            with self.subTest(value=value), self.assertRaises(InvalidRange):
                parse_byte_range(value, 10)


class DashboardServerTests(unittest.TestCase):
    def test_creates_a_local_threading_server_with_the_requested_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static = root / "dashboard"
            static.mkdir()
            video = root / "match.mp4"
            data = root / "replay.json"
            settings = DashboardSettings(static, video, data)
            with patch("football_tracker.dashboard.ThreadingHTTPServer") as server_type:
                server = create_server(settings, host="127.0.0.1", port=8123)

        self.assertIs(server, server_type.return_value)
        address, handler = server_type.call_args.args
        self.assertEqual(address, ("127.0.0.1", 8123))
        self.assertEqual(handler.keywords["settings"], settings)


class DashboardStartupTests(unittest.TestCase):
    def test_missing_demo_files_explain_how_to_recover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--video",
                        str(root / "missing.mp4"),
                        "--data",
                        str(root / "missing.json"),
                        "--static-directory",
                        str(root / "dashboard"),
                    ]
                )
            message = stderr.getvalue()

        self.assertEqual(code, 2)
        self.assertIn("missing.mp4", message)
        self.assertIn("missing.json", message)
        self.assertIn("index.html", message)
        self.assertIn("GitHub Release", message)
        self.assertIn("outputs/stem2connect-annotated-60s.mp4", message)
        self.assertIn("outputs/stem2connect-replay-data.json", message)
        self.assertIn("--video", message)
        self.assertIn("--data", message)
        self.assertIn("mp4v", message)
        self.assertIn("avc1", message)
        self.assertIn("--static-directory", message)

    def test_existing_files_pass_the_startup_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static = root / "dashboard"
            static.mkdir()
            for name in ("index.html", "app.css", "app.mjs", "replay-state.mjs"):
                (static / name).write_text("ok", encoding="utf-8")
            video = root / "match.mp4"
            data = root / "replay.json"
            video.write_bytes(b"not a real video")
            data.write_text("{}", encoding="utf-8")
            ensure_dashboard_files(DashboardSettings(static, video, data))

    def test_browser_codec_notice_accepts_h264_and_flags_mp4v(self) -> None:
        self.assertIsNone(browser_playback_notice("avc1"))
        self.assertIsNone(browser_playback_notice("H264"))
        notice = browser_playback_notice("mp4v")
        self.assertIsNotNone(notice)
        assert notice is not None
        self.assertIn("mp4v", notice)
        self.assertIn("avc1", notice)
        self.assertIsNone(browser_playback_notice(None))

    def test_mp4v_file_is_flagged_for_browser_playback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (160, 96))
            if not writer.isOpened():
                self.skipTest("Test environment cannot create an mp4v video.")
            writer.write(np.zeros((96, 160, 3), dtype=np.uint8))
            writer.release()
            codec = read_video_codec(path)

        self.assertIsNotNone(codec)
        self.assertIsNotNone(browser_playback_notice(codec))

    def test_help_documents_configurable_demo_paths(self) -> None:
        from football_tracker.dashboard import build_parser

        help_text = build_parser().format_help()
        self.assertIn("--video", help_text)
        self.assertIn("--data", help_text)
        self.assertIn("stem2connect-annotated-60s.mp4", help_text)
        self.assertIn("stem2connect-replay-data.json", help_text)
        self.assertIn("mp4v", help_text)


class ReadmeJudgeSectionsTests(unittest.TestCase):
    def test_readme_is_rewritten_for_judges(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("Deliberately deferred", readme)
        self.assertNotIn("first five milestones", readme)
        for heading in (
            "## Who it is for",
            "## Judge quickstart",
            "## Demo files",
            "## AI disclosure",
            "## What I learned",
            "## Limitations",
            "## Next steps",
            "## Hackathon track",
        ):
            self.assertIn(heading, readme)
        self.assertIn("Education", readme)
        self.assertIn("Community Impact", readme)
        self.assertIn("TODO", readme)
        self.assertIn("?track=", readme)
        self.assertIn("Ultralytics", readme)
        self.assertIn("ByteTrack", readme)
        self.assertIn("avc1", readme)


if __name__ == "__main__":
    unittest.main()
