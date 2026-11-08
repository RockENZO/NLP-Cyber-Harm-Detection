"""Render the checked-in project preview and verify the generated artifact."""
import subprocess
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "docs/preview/render_preview.py"
PREVIEW = ROOT / "docs/preview/nlp-project-preview.png"


class PreviewRendererTests(unittest.TestCase):
    def test_renderer_writes_a_1600x900_rgb_preview(self):
        subprocess.run(
            [sys.executable, str(RENDERER)],
            cwd=ROOT,
            check=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
        self.assertTrue(PREVIEW.is_file())
        with Image.open(PREVIEW) as image:
            self.assertEqual(image.size, (1600, 900))
            self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
