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

    def test_pitch_model_enables_calibration_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "clip.mp4"
            pitch_model = root / "pitch.pt"
            input_path.touch()
            pitch_model.touch()
            received = []

            exit_code = main(
                [
                    str(input_path),
                    "--output",
                    str(root / "tracked.mp4"),
                    "--pitch-model",
                    str(pitch_model),
                ],
                runner=received.append,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(received[0].pitch_model, str(pitch_model))
        self.assertEqual(received[0].pitch_minimum_keypoints, 6)

    def test_roster_enables_optional_identity_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "clip.mp4"
            roster_path = root / "roster.csv"
            input_path.touch()
            roster_path.write_text(
                "team,number,name\nteam-a,10,Ada\n",
                encoding="utf-8",
            )
            received = []

            exit_code = main(
                [
                    str(input_path),
                    "--output",
                    str(root / "tracked.mp4"),
                    "--roster",
                    str(roster_path),
                ],
                runner=received.append,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(received[0].roster_path, roster_path)
        self.assertEqual(received[0].jersey_minimum_votes, 3)


if __name__ == "__main__":
    unittest.main()
