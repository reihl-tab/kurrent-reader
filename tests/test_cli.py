"""CLI smoke tests that avoid the heavy model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kurrent_reader.cli import main


def test_preprocess_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "p.png"
    cv2.imwrite(str(src), np.full((200, 300), 200, np.uint8))
    assert main(["preprocess", str(src), str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert '"steps"' in out


def test_help_lists_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    for word in ("preprocess", "segment", "read", "run"):
        assert word in out
