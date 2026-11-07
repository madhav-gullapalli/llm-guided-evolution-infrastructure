"""Test response parsing without importing model clients or loading weights."""
import ast
from pathlib import Path
import re

import pytest


tree = ast.parse((Path(__file__).resolve().parents[1] / "src/llm_utils.py").read_text())
parser = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "clean_code_from_llm")
namespace = {"re": re}
exec(compile(ast.Module(body=[parser], type_ignores=[]), "llm_utils.py", "exec"), namespace)
clean = namespace["clean_code_from_llm"]


@pytest.mark.parametrize("position", [0, 1, 2])
def test_longest_block_wins_over_placeholder_and_example(position):
    full_code = "class Surrogate:\n    def fit(self, x, y):\n        return self.train(x, y)"
    blocks = ["#... rest of the module...", "x = 1"]
    blocks.insert(position, full_code)
    response = "Explanation\n" + "\nMore explanation\n".join(
        f"```python\n{code}\n```" for code in blocks)
    assert clean(response) == full_code


def test_unlabelled_fence_and_windows_newlines():
    assert clean("```\r\nx = 1\r\n``` ") == "x = 1"


@pytest.mark.parametrize("response", [None, "no fences", "```python\nx = 1", "```python\n\n```"])
def test_missing_complete_nonempty_block(response):
    assert clean(response) == "ERROR"
