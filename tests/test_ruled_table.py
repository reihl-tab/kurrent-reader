"""Tests for ruled-table geometry on synthetic scans."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kurrent_reader.ruled_table import align_chronological, detect_rules, row_bands


def _synthetic_table() -> NDArray[np.uint8]:
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


def test_align_anchors_to_bottom_of_range() -> None:
    bands = row_bands(_synthetic_table())
    aligned = align_chronological(bands, 5)
    assert aligned[0] is None and aligned[1] is None
    assert aligned[2:] == bands  # the LAST entries of the range get the page's bands


def test_align_empty() -> None:
    assert align_chronological([], 0) == []


def _skewed_table(angle: float = 0.0) -> NDArray[np.uint8]:
    from PIL import Image

    img: NDArray[np.uint8] = np.full((800, 600), 220, dtype=np.uint8)
    for y in range(50, 750, 60):
        img[y - 1 : y + 2, :] = 30  # horizontal rules
    for x in range(80, 520, 110):
        img[:, x - 1 : x + 2] = 30  # vertical rules
    if angle:
        img = np.asarray(
            Image.fromarray(img).rotate(-angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255),
            dtype=np.uint8,
        )
    return img


def test_estimate_skew_zero_on_straight_table() -> None:
    from kurrent_reader.ruled_table import estimate_skew_deg

    assert abs(estimate_skew_deg(_skewed_table(0.0))) < 0.2


def test_estimate_skew_recovers_rotation() -> None:
    from kurrent_reader.ruled_table import estimate_skew_deg

    est = estimate_skew_deg(_skewed_table(1.5))
    assert abs(abs(est) - 1.5) < 0.4


def test_row_bands_deskewed_finds_rows_on_skewed_scan() -> None:
    from kurrent_reader.ruled_table import row_bands_deskewed

    bands, angle = row_bands_deskewed(_skewed_table(2.0))
    assert abs(abs(angle) - 2.0) < 0.5
    assert len(bands) >= 8  # deskew round trip restores detectability


def test_estimate_skew_recovers_known_angle() -> None:
    # _skewed_table(1.0) rotates content by -1.0 deg, so the estimator must
    # report about -1.0 (the skew present); deskew() then rotates by +1.0.
    from kurrent_reader.ruled_table import deskew, estimate_skew_deg, row_bands

    est = estimate_skew_deg(_skewed_table(1.0))
    assert -1.5 < est < -0.5  # sign and magnitude, not just non-zero
    assert 0.5 < estimate_skew_deg(_skewed_table(-1.0)) < 1.5
    fixed = deskew(_skewed_table(1.0), est)
    assert abs(estimate_skew_deg(fixed)) < 0.2  # round-trip removes the skew
    assert len(row_bands(fixed)) >= 10
