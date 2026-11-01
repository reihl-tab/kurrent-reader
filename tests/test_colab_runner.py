"""Guard tests for the versioned Colab runner script (issue #5)."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v2.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v4.py",
]


def test_colab_runner_compiles() -> None:
    for script in SCRIPTS:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_stdout_reconfigure_is_guarded() -> None:
    """sys.stdout.reconfigure must sit inside try/except AttributeError.

    Jupyter/Colab replaces sys.stdout with ipykernel OutStream, which has no
    reconfigure(); an unguarded call crashes the run at startup (issue #5).
    """
    for script in SCRIPTS:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        assert _has_guard(tree), (
            f"{script.name}: no try/except AttributeError guard around sys.stdout.reconfigure"
        )


def _has_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(isinstance(h.type, ast.Name) and h.type.id == "AttributeError" for h in node.handlers):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "reconfigure"
            ):
                return True
    return False
