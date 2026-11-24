"""Checks that the manual demo renders a full simulated answer.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))

import demo_answer  # noqa: E402


class TestDemoAnswer(unittest.TestCase):
    def test_conversation_contains_comparison_choice_and_final_answer(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            demo_answer.main()
        text = buffer.getvalue()
        for phrase in (
            "SIMULATED CONVERSATION",
            "We found 3 versions of Chicken Inasal",
            "USER: pick the fastest",
            "Recommended: Chicken Inasal",
            "Matched ingredients (you have):",
            "Missing ingredients:",
            "Suggested substitutions:",
            "Simplified steps:",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
