"""Guard tests for the versioned Colab runner script (issue #5)."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v2.py",
]
SCRIPT = SCRIPTS[0]


def test_colab_runner_compiles() -> None:
    for script in SCRIPTS:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_stdout_reconfigure_is_guarded() -> None:
    """sys.stdout.reconfigure must sit inside try/except AttributeError.

    Jupyter/Colab replaces sys.stdout with ipykernel OutStream, which has no
    reconfigure(); an unguarded call crashes the run at startup (issue #5).
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        guards_attribute_error = any(
            isinstance(h.type, ast.Name) and h.type.id == "AttributeError" for h in node.handlers
        )
        if not guards_attribute_error:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "reconfigure"
            ):
                return
    raise AssertionError("no try/except AttributeError guard around sys.stdout.reconfigure")
