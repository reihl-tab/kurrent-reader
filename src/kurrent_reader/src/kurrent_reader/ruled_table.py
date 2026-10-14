"""Row and column extraction for ruled register tables (Eschau death books).

The 19th-century Eschau death registers (JUF source 1081, s01-s08 scans)
are hand-ruled tables: vertical rules split columns, horizontal rules split
entry rows. Rules are detected from dark-pixel projection profiles and
entries are aligned to rows chronologically, never by reading the ink.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _group_contiguous(indices: list[int], max_gap: int = 4) -> list[int]:
    """Collapse contiguous index runs into their mean positions."""
    groups: list[list[int]] = []
    for idx in indices:
        if groups and idx - groups[-1][-1] <= max_gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return [int(np.mean(g)) for g in groups if len(g) >= 2]


def detect_rules(
    gray: NDArray[np.uint8],
    axis: int,
    dark_threshold: int = 140,
    min_fraction: float = 0.2,
) -> list[int]:
    """Detect long straight rules in a grayscale image.

    axis=0 finds horizontal rules (dark rows), axis=1 vertical rules.
    A line qualifies when at least ``min_fraction`` of its pixels along the
    rule direction are darker than ``dark_threshold``.
    """
    dark = gray < dark_threshold
    profile = dark.mean(axis=axis)
    hits = [i for i, v in enumerate(profile) if v > min_fraction]
    return _group_contiguous(hits)


@dataclass
class RowBand:
    """One table row: pixel bounds and the column slices inside it."""

    y0: int
    y1: int
    columns: list[tuple[int, int]]

    def crop_boxes(self) -> list[tuple[int, int, int, int]]:
        return [(x0, self.y0, x1, self.y1) for x0, x1 in self.columns]


def row_bands(
    gray: NDArray[np.uint8],
    dark_threshold: int = 140,
    min_fraction: float = 0.2,
    min_height: int = 40,
) -> list[RowBand]:
    """Split a ruled table image into row bands with column slices."""
    h_rules = detect_rules(gray, axis=1, dark_threshold=dark_threshold, min_fraction=min_fraction)
    v_rules = detect_rules(gray, axis=0, dark_threshold=dark_threshold, min_fraction=min_fraction)
    columns = [(a, b) for a, b in zip(v_rules, v_rules[1:])]
    bands = [
        RowBand(y0=a, y1=b, columns=columns)
        for a, b in zip(h_rules, h_rules[1:])
        if b - a >= min_height
    ]
    return bands


def align_chronological(bands: list[RowBand], n_entries: int) -> list[RowBand | None]:
    """Assign the ``n_entries`` chronologically ordered entries to row bands.

    Register rows are written in date order top to bottom. With more bands
    than entries the header band(s) are skipped; with fewer bands than
    entries the surplus entries get ``None`` and must be looked for on the
    next scan.
    """
    if n_entries <= 0:
        return []
    if len(bands) >= n_entries:
        selected: list[RowBand | None] = list(bands[len(bands) - n_entries :])
        return selected
    padded: list[RowBand | None] = [None] * (n_entries - len(bands)) + list(bands)
    return padded
