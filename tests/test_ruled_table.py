"""Tests for ruled-table geometry on synthetic scans."""

from __future__ import annotations

import numpy as np

from kurrent_reader.ruled_table import align_chronological, detect_rules, row_bands


def _synthetic_table() -> np.ndarray:
    img = np.full((400, 600), 220, dtype=np.uint8)
    for y in (50, 150, 250, 350):
        img[y - 1 : y + 2, :] = 30  # horizontal rules
    for x in (80, 300, 520):
        img[:, x - 1 : x + 2] = 30  # vertical rules
    return img


def test_detect_rules_horizontal_and_vertical() -> None:
    img = _synthetic_table()
    assert detect_rules(img, axis=1) == [50, 150, 250, 350]
    assert detect_rules(img, axis=0) == [80, 300, 520]


def test_row_bands_columns() -> None:
    bands = row_bands(_synthetic_table())
    assert len(bands) == 3
    assert bands[0].y0 == 50 and bands[0].y1 == 150
    assert bands[0].columns == [(80, 300), (300, 520)]
    assert bands[0].crop_boxes() == [(80, 50, 300, 150), (300, 50, 520, 150)]


def test_align_skips_header_bands() -> None:
    bands = row_bands(_synthetic_table())
    aligned = align_chronological(bands, 2)
    assert aligned == bands[1:]


def test_align_missing_rows_gives_none() -> None:
    bands = row_bands(_synthetic_table())
    aligned = align_chronological(bands, 4)
    assert aligned[0] is None
    assert aligned[1:] == bands


def test_align_empty() -> None:
    assert align_chronological([], 0) == []
