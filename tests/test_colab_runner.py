"""Guard tests for the versioned Colab runner script (issue #5)."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v2.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v8.py",
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


V4_RUNNERS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v8.py",
]


def _load_fn(path: Path, name: str, ns: dict):
    """Exec a single top-level function out of a runner script into ns."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(path), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{path.name}: function {name} not found")


def test_combined_score_excludes_inf_from_divisor() -> None:
    """inf entries must not inflate the divisor (spurious NEW BEST, issue #24).

    Applies to the v4-lineage combined_score; kernel v8's frozen variant
    returns the names CER directly and has no divisor.
    """
    for script in V4_RUNNERS[:2]:
        ns = {"score12": lambda r: r["s"]}
        fn = _load_fn(script, "combined_score", ns)
        assert fn({"s": float("inf")}, {"n": 1, "cer": 0.5}) == 0.5
        assert abs(fn({"s": 0.2}, {"n": 1, "cer": 0.4}) - 0.3) < 1e-9
        assert fn({"s": float("inf")}, {}) == float("inf")


def test_held_out_leak_assert_normalizes_idnr() -> None:
    """The row-level held-out check must catch int or str IdNr (issue #24)."""
    for script in V4_RUNNERS:
        src = script.read_text(encoding="utf-8")
        assert '_held_idnrs = {"33739", "35238"}' in src, f"{script.name}: held-out set not normalized"
        assert 'str(s.get("IdNr"))' in src, f"{script.name}: IdNr not normalized to str in leak check"


def test_evaluate12_rejects_traversal() -> None:
    """evaluate12 must resolve eval file paths under DATA only (issue #24)."""
    import tempfile

    for script in V4_RUNNERS:
        ns: dict = {}
        fn = _load_fn(script, "_safe_data_path", ns)
        src = script.read_text(encoding="utf-8")
        assert '_safe_data_path(DATA, r["file"])' in src, f"{script.name}: evaluate12 not using the guard"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assert fn(root, "a/b.png") == (root / "a/b.png").resolve()
            for bad in ("../escape.png", "a/../../escape.png", "/etc/passwd"):
                try:
                    fn(root, bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{script.name}: traversal accepted: {bad}")


def _load_assignments(path: Path, names: set, ns: dict) -> dict:
    """Exec top-level assignments to the given names, in source order, into ns."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    found = {t.id for n in nodes for t in n.targets if isinstance(t, ast.Name)}
    missing = names - found
    if missing:
        raise AssertionError(f"{path.name}: assignments not found: {sorted(missing)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_held_out_leak_check_catches_int_and_str() -> None:
    """Behaviorally pin the row-level held-out check (issue #24).

    Execs the runner's own _held_idnrs/_leaked lines against synthetic
    samples: the held-out IdNrs must be caught whether the manifest stores
    them as int or str, and clean samples must pass.
    """
    for script in V4_RUNNERS:
        for idnr in (33739, "33739", 35238, "35238"):
            ns = {"samples": [{"IdNr": idnr}, {"IdNr": 1}, {"IdNr": "2"}, {"IdNr": None}]}
            _load_assignments(script, {"_held_idnrs", "_leaked"}, ns)
            assert ns["_leaked"] == [str(idnr)], f"{script.name}: missed held-out IdNr {idnr!r}"
        ns = {"samples": [{"IdNr": 1}, {"IdNr": "2"}, {"IdNr": None}]}
        _load_assignments(script, {"_held_idnrs", "_leaked"}, ns)
        assert ns["_leaked"] == [], f"{script.name}: false positive on clean samples"
