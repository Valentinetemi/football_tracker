from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from football_tracker.dashboard import (
    DashboardSettings,
    InvalidRange,
    create_server,
    parse_byte_range,
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


if __name__ == "__main__":
    unittest.main()
