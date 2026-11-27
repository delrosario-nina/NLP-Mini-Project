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

    def test_retrieval_root_lookup_is_not_tied_to_a_named_cwd(self):
        nb = json.loads((ROOT / "notebooks" / "03_retrieval.ipynb").read_text(encoding="utf-8"))
        setup = next(
            "".join(cell["source"]) for cell in nb["cells"]
            if cell["cell_type"] == "code" and "ingredient_utils.py" in "".join(cell["source"])
        )
        self.assertNotIn('Path.cwd().name == "notebooks"', setup)
        self.assertIn("start.parents", setup)
        self.assertIn('"src" / "ingredient_utils.py"', setup)

    def test_retrieval_import_cell_can_bootstrap_itself(self):
        nb = json.loads((ROOT / "notebooks" / "03_retrieval.ipynb").read_text(encoding="utf-8"))
        import_cell = next(
            "".join(cell["source"]) for cell in nb["cells"]
            if cell["cell_type"] == "code"
            and "from ingredient_utils import" in "".join(cell["source"])
        )
        self.assertIn("except ModuleNotFoundError:", import_cell)
        self.assertIn('sys.path.insert(0, str(src))', import_cell)
        self.assertIn("from ingredient_utils import", import_cell)


if __name__ == "__main__":
    unittest.main()
