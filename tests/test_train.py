"""Tests for the training-data helpers (no model downloads)."""

from __future__ import annotations

import json
from pathlib import Path

from kurrent_reader.train import (
    cer,
    exclude_files,
    load_manifest,
    load_training_samples,
    make_pad_collator,
    word_accuracy,
)


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


def test_load_manifest_skips_path_escape(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"x")
    rows = [
        {"file": "a.png", "text": "inside"},
        {"file": "../outside.png", "text": "escapes base dir"},
    ]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    samples = load_manifest(manifest, tmp_path)
    assert [s.text for s in samples] == ["inside"]


def test_load_training_samples_enforces_blocklist(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "held.png").write_bytes(b"x")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"file": "a.png", "text": "train"})
        + "\n"
        + json.dumps({"file": "held.png", "text": "reserved"}),
        encoding="utf-8",
    )
    samples = load_training_samples(manifest, tmp_path, {"held.png"})
    assert [s.file.name for s in samples] == ["a.png"]


def test_word_accuracy_duplicate_words_need_duplicate_matches() -> None:
    assert word_accuracy("den den", "den den") == 1.0
    assert word_accuracy("den", "den den") == 0.5


def test_pad_collator_handles_variable_label_lengths() -> None:
    torch = __import__("torch")
    collate = make_pad_collator(pad_token_id=1)
    features = [
        {"pixel_values": torch.zeros(3, 4, 4), "labels": torch.tensor([5, 6, 7, 8])},
        {"pixel_values": torch.ones(3, 4, 4), "labels": torch.tensor([9])},
    ]
    batch = collate(features)
    assert batch["pixel_values"].shape == (2, 3, 4, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["labels"][0].tolist() == [5, 6, 7, 8]
    assert batch["labels"][1].tolist() == [9, -100, -100, -100]
