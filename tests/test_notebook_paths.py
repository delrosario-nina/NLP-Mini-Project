"""Guard against the 'ingredient_utils not found' notebook error.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    "01_preprocessing.ipynb",
    "02_alias_map.ipynb",
    "03_retrieval.ipynb",
    "04_rag_pipeline.ipynb",
    "05_version_comparison.ipynb",
]


class TestNotebookPaths(unittest.TestCase):
    def test_every_notebook_adds_src_to_sys_path(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
                code = "\n".join(
                    "".join(cell["source"])
                    for cell in nb["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn('sys.path.insert(0, str(ROOT / "src"))', code)
                if name != "05_version_comparison.ipynb":  # 05 uses ROOT / paths throughout
                    self.assertIn("os.chdir(ROOT)", code)


if __name__ == "__main__":
    unittest.main()
