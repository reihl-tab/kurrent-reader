"""Boot-path behavioral tests for kernel v15 (clean deduped merged manifest).

Pattern per the boot-crash rule: exec the REAL blocks from the script against
tiny stand-ins, not a hand-copied variant. Pins the v15 edits: single merged
load (no embedded blob / batch-1 mount), held-out IdNr decontam, text-level
leak filter ("Gerson Mosbacher" eval-target text), cell-count assert,
lowlr default, 2000-step cap.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

V15 = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v15.py"
SRC = V15.read_text(encoding="utf-8")
LINES = SRC.splitlines()


def _span(start_prefix: str, end_prefix: str) -> str:
    start = next(i for i, l in enumerate(LINES) if l.startswith(start_prefix))
    end = next(i for i, l in enumerate(LINES) if l.startswith(end_prefix))
    return "\n".join(LINES[start : end + 1])


def test_v15_compiles() -> None:
    compile(SRC, str(V15), "exec")


def test_no_embedded_blob_or_batch1_mount() -> None:
    assert '_EMB = "' not in SRC, "v15 must not carry the embedded seed blob"
    assert "kurrent-batch1" not in SRC, "v15 must not mount the superseded batch-1 dataset"


def test_checkpoint_pins() -> None:
    assert "MAX_STEPS = 2000" in SRC
    for idnr in ("33739", "35238", "38666", "38678", "38693", "38696"):
        assert f'"{idnr}"' in SRC.split("_held_idnrs =")[1].splitlines()[0]
    assert 'GENTLE = os.environ.get("KURRENT_GENTLE", "lowlr")' in SRC


def test_page_dirs_include_batch2(tmp_path) -> None:
    block = _span("PAGE_DIRS = sorted", "print(\"STAGE page-roots:\"")
    (tmp_path / "kurrent-train-v1").mkdir()
    (tmp_path / "kurrent-train-v1" / "s02_22.jpg").write_bytes(b"x")
    (tmp_path / "kurrent-batch2").mkdir(exist_ok=True)
    (tmp_path / "kurrent-batch2" / "g09_22.jpg").write_bytes(b"x")
    ns = {"DATA_ROOT": tmp_path, "Path": Path}
    exec(compile(block, str(V15), "exec"), ns)
    assert "g09_22.jpg" in ns["PAGE_ROOT"], "batch-2 v2 mount must supply g09_22.jpg"
    assert "s02_22.jpg" in ns["PAGE_ROOT"]


def _cell(idnr, project="eschau1081-batch2", text="abc"):
    return {"file": "g01_22.jpg", "bbox": [1, 2, 3, 4], "text": text, "IdNr": idnr, "project": project}


def _cells_block_ns(tmp_path, cells, expect=None):
    import os as _os
    code = tmp_path / "kurrent-code"; code.mkdir(exist_ok=True)
    (code / "train_manifest_v4.jsonl").write_text(
        json.dumps({"file": "a.jpg", "text": "mm line", "project": "MM_1_001"}) + "\n" +
        json.dumps({"file": "s.jpg", "text": "seed line", "project": "eschau-seeds"}) + "\n")
    (tmp_path / "kurrent-batch2").mkdir(exist_ok=True)
    (tmp_path / "kurrent-batch2" / "batch2_manifest.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in cells))
    block = _span("import random as _rnd", "    assert not _leaked")
    ns = {"DATA_ROOT": tmp_path, "CODE": code, "SMOKE": False, "json": json, "DATA": tmp_path,
          "os": _os, "Path": Path}
    if expect is not None:
        _os.environ["KURRENT_EXPECT_CELLS"] = str(expect)
    try:
        exec(compile(block, str(V15), "exec"), ns)
    finally:
        _os.environ.pop("KURRENT_EXPECT_CELLS", None)
    return ns


def test_cells_block_merged_load(tmp_path) -> None:
    ns = _cells_block_ns(tmp_path, [_cell(80001), _cell(80002), _cell(80003)], expect=3)
    assert len(ns["_cells"]) == 3
    assert ns["REPEAT"] == 4
    assert len(ns["samples"]) == 1 + 3 * 4, "MM lines + merged cells x4"


def test_cells_block_drops_held_out_idnrs(tmp_path) -> None:
    ns = _cells_block_ns(tmp_path, [_cell(80001), _cell(33739), _cell(38666)], expect=1)
    assert [c["IdNr"] for c in ns["_cells"]] == [80001], "held-out IdNrs must be dropped at load"


def test_cells_block_drops_eval_target_text(tmp_path) -> None:
    ns = _cells_block_ns(tmp_path, [_cell(80001), _cell(38750, text="Gerson Mosbacher"),
                                    _cell(38668, text="Gerson Mosbacher")], expect=1)
    assert [c["IdNr"] for c in ns["_cells"]] == [80001], "eval-target name text must be dropped"


def test_cell_count_assert_fires(tmp_path) -> None:
    with pytest.raises(AssertionError):
        _cells_block_ns(tmp_path, [_cell(80001)], expect=301)


def _run_gentle_block(variant: str, model) -> dict:
    import os as _os
    block = _span("GENTLE = os.environ.get", "print(f\"STAGE gentle")
    ns: dict = {"os": _os, "model": model, "SMOKE": False, "LR": 5e-6}
    _os.environ["KURRENT_GENTLE"] = variant
    try:
        exec(compile(block, str(V15), "exec"), ns)
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


def test_push_metadata_enables_internet() -> None:
    """2026-11-06 incident: a push with enable_internet:false failed at
    processor load (DNS blocked). The push metadata MUST keep internet on
    (needed to fetch dh-unibe/trocr-kurrent from HuggingFace)."""
    meta = json.loads(
        (Path(__file__).resolve().parents[1] / "scripts" / "kernel-metadata.v15.json").read_text()
    )
    assert meta["enable_internet"] is True
    assert meta["enable_gpu"] is True
    assert "reihlinstinct/kurrent-batch2" in meta["dataset_sources"]
