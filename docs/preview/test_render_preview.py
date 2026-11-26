"""Render the checked-in evaluation report and verify a newly created image."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RENDERER = Path("docs/preview/render_preview.py")
REPORT = Path("evaluation/reports/nine_class_test_20260930.json")


class PreviewRendererTests(unittest.TestCase):
    def test_renderer_writes_a_1600x900_rgb_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            for asset in (RENDERER, REPORT):
                target = project / asset
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / asset, target)
            subprocess.run(
                [sys.executable, str(RENDERER)],
                cwd=project,
                check=True,
                timeout=120,
                capture_output=True,
                text=True,
            )
            preview = project / RENDERER.with_name("nlp-project-preview.png")
            with Image.open(preview) as image:
                image.load()
                self.assertEqual(image.size, (1600, 900))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
