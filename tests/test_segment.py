"""Segmentation tests on synthetic pages (kraken classic segmenter)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kurrent_reader import segment


def _page_with_lines(tmp_path: Path, n_lines: int = 5) -> Path:
    img = np.full((900, 1200), 255, np.uint8)
    for i in range(n_lines):
        y = 100 + i * 150
        # text-like band: clustered blobs, not bare hairlines
        for x in range(100, 1000, 18):
            cv2.circle(img, (x, y + 15), 8, 0, -1)
    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), img)
    return path


def test_segment_finds_lines_in_order(tmp_path: Path) -> None:
    page = _page_with_lines(tmp_path)
    boxes = segment.segment_page(page)
    assert len(boxes) >= 4
    ys = [b.y0 for b in boxes]
    assert ys == sorted(ys)


def test_crop_lines_writes_files(tmp_path: Path) -> None:
    page = _page_with_lines(tmp_path)
    boxes = segment.segment_page(page)
    manifest = segment.crop_lines(page, boxes, tmp_path / "lines")
    assert len(manifest) == len(boxes)
    for entry in manifest:
        assert (tmp_path / "lines" / str(entry["file"])).exists()
    assert (tmp_path / "lines" / "lines.json").exists()


def test_unknown_mode_rejected(tmp_path: Path) -> None:
    page = _page_with_lines(tmp_path)
    try:
        segment.segment_page(page, mode="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
