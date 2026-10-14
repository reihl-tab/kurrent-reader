"""Tests for the training-data helpers (no model downloads)."""

from __future__ import annotations

import json
from pathlib import Path

from kurrent_reader.train import cer, exclude_files, load_manifest, word_accuracy


def test_cer_perfect_and_empty() -> None:
    assert cer("abc", "abc") == 0.0
    assert cer("", "abc") == 1.0


def test_cer_substitution() -> None:
    assert cer("ist", "30ten May 1845") > 0.5


def test_word_accuracy() -> None:
    assert word_accuracy("Den Nov ist dem", "Den 23ten Nov ist dem") == 0.8
    assert word_accuracy("", "abc") == 0.0


def test_load_manifest_skips_blank_and_missing(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    rows = [
        {"file": "a.png", "text": "Gerson Mosbacher"},
        {"file": "b.png", "text": "  "},
        {"file": "c.png", "text": "missing image"},
    ]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    samples = load_manifest(manifest, tmp_path)
    assert [s.text for s in samples] == ["Gerson Mosbacher"]


def test_exclude_files(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "h01.png").write_bytes(b"x")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"file": "a.png", "text": "x"}) + "\n" + json.dumps({"file": "h01.png", "text": "y"}),
        encoding="utf-8",
    )
    samples = load_manifest(manifest, tmp_path)
    kept = exclude_files(samples, {"h01.png"})
    assert [s.file.name for s in kept] == ["a.png"]
