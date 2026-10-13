"""Tests for table-rule detection and removal."""

from __future__ import annotations

import numpy as np

from kurrent_reader import derule


def _synthetic_table() -> np.ndarray:
    """White page, two ruled cells, text-like blobs inside each cell."""
    img = np.full((120, 200), 255, np.uint8)
    img[60, :] = 0  # horizontal rule
    img[:, 100] = 0  # vertical rule
    # text-like blobs (short strokes, far from rule length)
    img[20:26, 20:70] = 0
    img[30:34, 25:60] = 0
    img[80:86, 120:180] = 0
    img[90:94, 130:170] = 0
    return img


def test_detect_positions() -> None:
    img = _synthetic_table()
    mask = derule.detect_ruling_lines(img, line_fraction=0.5)
    vert = derule._rule_positions(mask, axis=0)
    horiz = derule._rule_positions(mask, axis=1)
    assert len(vert) == 1 and abs(vert[0] - 100) <= 3
    assert len(horiz) == 1 and abs(horiz[0] - 60) <= 3


def test_removal_erases_rules_keeps_text() -> None:
    img = _synthetic_table()
    result = derule.remove_ruling_lines(img, line_fraction=0.5)
    # rule pixels away from text become white again
    assert result.deruled[60, 195] > 200
    assert result.deruled[110, 100] > 200
    # text blobs survive (ink far from rules)
    assert result.deruled[22, 40] < 100
    assert result.deruled[82, 150] < 100
    assert result.steps


def test_split_columns() -> None:
    img = _synthetic_table()
    spans = derule.split_columns(img, [100], pad=3)
    assert spans == [(0, 97), (103, 200)]


def test_split_columns_no_rules() -> None:
    img = _synthetic_table()
    assert derule.split_columns(img, []) == [(0, 200)]


def test_no_rules_no_positions() -> None:
    img = np.full((120, 200), 255, np.uint8)
    img[20:26, 20:70] = 0  # text blob only, too short to be a rule
    result = derule.remove_ruling_lines(img, line_fraction=0.5)
    assert result.vertical_positions == []
    assert result.horizontal_positions == []


def test_split_words() -> None:
    img = np.full((40, 160), 255, np.uint8)
    img[10:30, 10:40] = 0  # word 1
    img[10:30, 70:110] = 0  # word 2
    spans = derule.split_words(img, gap=12)
    assert spans == [(10, 40), (70, 110)]


def test_split_words_merges_close_parts() -> None:
    img = np.full((40, 160), 255, np.uint8)
    img[10:30, 10:30] = 0
    img[10:30, 34:50] = 0  # 4 px gap: same word
    spans = derule.split_words(img, gap=12)
    assert spans == [(10, 50)]


def test_pad_to_strip_shape_and_ink() -> None:
    word = np.full((20, 40), 255, np.uint8)
    word[5:15, 10:30] = 0
    strip = derule.pad_to_strip(word, target_height=64)
    assert strip.shape == (64, 40 * ((64 - 16) / 20) + 32)
    assert strip.min() == 0  # ink survived
    assert strip[0, 0] == 255  # white border
