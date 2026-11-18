from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from football_tracker.cli import main


class CliTests(unittest.TestCase):
    def test_module_help_is_available_without_loading_ml_stack(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "football_tracker.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detect and track", result.stdout)

    def test_cli_builds_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "clip.mp4"
            input_path.touch()
            received = []

            exit_code = main(
                [str(input_path), "--output", str(Path(directory) / "tracked.mp4")],
                runner=received.append,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(received[0].input_path, input_path)
        self.assertEqual(received[0].model, "yolo11n.pt")


if __name__ == "__main__":
    unittest.main()
