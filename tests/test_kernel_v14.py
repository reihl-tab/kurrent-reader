"""Boot-path behavioral tests for kernel v14 (wave-2 checkpoint run).

Pattern per the boot-crash rule: exec the REAL blocks from the script against
tiny stand-ins (torch module / synthetic dirs), not a hand-copied variant.
Pins the v14 edits: batch-2 lever, extended held-out IdNrs, lowlr default,
2000-step cap, and the v13 gentle block (requires_grad attribute hotfix).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

V14 = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v14.py"
SRC = V14.read_text(encoding="utf-8")
LINES = SRC.splitlines()


def _span(start_prefix: str, end_prefix: str) -> str:
    start = next(i for i, l in enumerate(LINES) if l.startswith(start_prefix))
    end = next(i for i, l in enumerate(LINES) if l.startswith(end_prefix))
    return "\n".join(LINES[start : end + 1])


def test_v14_compiles() -> None:
    compile(SRC, str(V14), "exec")


def test_stdout_reconfigure_is_guarded() -> None:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if any(isinstance(h.type, ast.Name) and h.type.id == "AttributeError" for h in node.handlers):
            if any(isinstance(i, ast.Call) and isinstance(i.func, ast.Attribute) and i.func.attr == "reconfigure"
                   for i in ast.walk(node)):
                return
    raise AssertionError("no try/except AttributeError guard around sys.stdout.reconfigure")


def test_short_checkpoint_pins() -> None:
    assert "MAX_STEPS = 2000" in SRC, "v14 must cap the approved short run at 2000 steps"
    for idnr in ("33739", "35238", "38666", "38678", "38693", "38696"):
        assert f'"{idnr}"' in SRC.split("_held_idnrs =")[1].splitlines()[0], f"held-out IdNr {idnr} missing"
    assert 'GENTLE = os.environ.get("KURRENT_GENTLE", "lowlr")' in SRC, "v14 default must be the v13a lowlr recipe"


def test_page_dirs_include_batch2(tmp_path) -> None:
    block = _span("PAGE_DIRS = sorted", "print(\"STAGE page-roots:\"")
    (tmp_path / "kurrent-train-v1").mkdir()
    (tmp_path / "kurrent-train-v1" / "s02_22.jpg").write_bytes(b"x")
    (tmp_path / "kurrent-batch2").mkdir(exist_ok=True)
    (tmp_path / "kurrent-batch2" / "g01_22.jpg").write_bytes(b"x")
    ns = {"DATA_ROOT": tmp_path, "Path": Path}
    exec(compile(block, str(V14), "exec"), ns)
    assert "g01_22.jpg" in ns["PAGE_ROOT"], "batch-2 mount must supply g01_22.jpg"
    assert "s02_22.jpg" in ns["PAGE_ROOT"]


def _cell(idnr, project="eschau1081-batch2"):
    return {"file": "g01_22.jpg", "bbox": [1, 2, 3, 4], "text": "abc", "IdNr": idnr, "project": project}


def _cells_block_ns(tmp_path, b2_cells):
    code = tmp_path / "kurrent-code"; code.mkdir(exist_ok=True)
    (code / "train_manifest_v4.jsonl").write_text(
        json.dumps({"file": "a.jpg", "text": "mm line", "project": "MM_1_001"}) + "\n" +
        json.dumps({"file": "s.jpg", "text": "seed line", "project": "eschau-seeds"}) + "\n")
    (tmp_path / "kurrent-batch1").mkdir(exist_ok=True)
    (tmp_path / "kurrent-batch1" / "batch1_manifest.jsonl").write_text(
        "".join(json.dumps(_cell(90000 + i, "eschau1081-batch1")) + "\n" for i in range(2)))
    (tmp_path / "kurrent-batch2").mkdir(exist_ok=True)
    (tmp_path / "kurrent-batch2" / "batch2_manifest.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in b2_cells))
    block = _span("_EMB = ", "    assert not _leaked")
    ns = {"DATA_ROOT": tmp_path, "CODE": code, "SMOKE": False, "json": json, "DATA": tmp_path}
    exec(compile(block, str(V14), "exec"), ns)
    return ns


def test_cells_block_loads_batch1_and_batch2(tmp_path) -> None:
    ns = _cells_block_ns(tmp_path, [_cell(80001), _cell(80002), _cell(80003)])
    assert len(ns["_b2"]) == 3
    assert len(ns["_cells"]) == 140 + 2 + 3, "embedded (148-8 contaminated) + batch1 + batch2"
    assert ns["REPEAT"] == 4
    # mix = MM lines (eschau-* projects excluded) + cells x4
    assert len(ns["samples"]) == 1 + (140 + 2 + 3) * 4


def test_cells_block_rejects_held_out_idnrs(tmp_path) -> None:
    for idnr in (33739, 35238, 38666, 38678, 38693, 38696):
        with pytest.raises(AssertionError):
            _cells_block_ns(tmp_path, [_cell(idnr)])


def _run_gentle_block(variant: str, model) -> dict:
    import os as _os
    block = _span("GENTLE = os.environ.get", "print(f\"STAGE gentle")
    ns: dict = {"os": _os, "model": model, "SMOKE": False, "LR": 5e-6}
    _os.environ["KURRENT_GENTLE"] = variant
    try:
        exec(compile(block, str(V14), "exec"), ns)
    finally:
        _os.environ.pop("KURRENT_GENTLE", None)
    return ns


def test_gentle_block_lowlr_default() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    ns = _run_gentle_block("lowlr", model)
    assert ns["LR"] == 1e-6
    assert ns["_frozen"] == 0
    assert all(p.requires_grad for p in model.parameters())
