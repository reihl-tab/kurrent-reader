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

    axis=0 finds vertical rules (the profile is averaged over rows, so hits
    are x positions); axis=1 finds horizontal rules (hits are y positions).
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

    Entries are anchored to the BOTTOM of the page's band list. With more
    bands than entries, the top band(s) are skipped (headers and any earlier
    rows). With fewer bands than entries, the EARLIEST entries get ``None``:
    the page is assumed to hold the tail of the requested chronological
    range, with the missing earlier entries on earlier scans. Callers that
    need the opposite page-position assumption must realign beforehand.
    """
    if n_entries <= 0:
        return []
    if len(bands) >= n_entries:
        selected: list[RowBand | None] = list(bands[len(bands) - n_entries :])
        return selected
    padded: list[RowBand | None] = [None] * (n_entries - len(bands)) + list(bands)
    return padded


def estimate_skew_deg(
    gray: NDArray[np.uint8],
    n_bands: int = 5,
    dark_threshold: int = 140,
    min_fraction: float = 0.2,
) -> float:
    """Estimate page skew in degrees from vertical-rule drift across bands.

    Full-page projection fails on slightly rotated scans because a rule
    drifts across columns over long y-ranges. Detect vertical rules per
    horizontal band, track each rule across bands by nearest-x matching,
    and fit each rule's x drift against band y; the median slope gives the
    skew angle. Returns 0.0 when too few rules can be tracked.
    """
    import math

    h, _w = gray.shape
    band_h = h // n_bands
    strips: list[tuple[float, list[int]]] = []
    for b in range(n_bands):
        strip = gray[b * band_h : (b + 1) * band_h]
        rules = detect_rules(strip, axis=0, dark_threshold=dark_threshold, min_fraction=min_fraction)
        strips.append((b * band_h + band_h / 2, rules))
    ref_y, ref_rules = max(strips, key=lambda s: len(s[1]))
    if len(ref_rules) < 2:
        return 0.0
    slopes: list[float] = []
    for ref_x in ref_rules:
        pts = [(ref_y, float(ref_x))]
        for y, rules in strips:
            if not rules:
                continue
            nearest = min(rules, key=lambda x: abs(x - pts[-1][1]))
            if abs(nearest - pts[-1][1]) <= band_h * 0.25 + 8:
                pts.append((y, float(nearest)))
        if len(pts) >= 3:
            ys = np.array([p[0] for p in pts])
            xs = np.array([p[1] for p in pts])
            slope, _intercept = np.polyfit(ys, xs, 1)
            slopes.append(float(slope))
    if not slopes:
        return 0.0
    return math.degrees(math.atan(float(np.median(slopes))))


def deskew(gray: NDArray[np.uint8], angle_deg: float) -> NDArray[np.uint8]:
    """Rotate a grayscale image by ``angle_deg`` (PIL, no canvas expansion)."""
    from PIL import Image

    if abs(angle_deg) < 0.05:
        return gray
    img = Image.fromarray(gray)
    rotated = img.rotate(-angle_deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255)
    return np.asarray(rotated, dtype=np.uint8)


def row_bands_deskewed(
    gray: NDArray[np.uint8],
    dark_threshold: int = 140,
    min_fraction: float = 0.2,
    min_height: int = 40,
    n_bands: int = 5,
) -> tuple[list[RowBand], float]:
    """Deskew-tolerant row detection: estimate skew, deskew, then row_bands."""
    angle = estimate_skew_deg(gray, n_bands=n_bands, dark_threshold=dark_threshold, min_fraction=min_fraction)
    fixed = deskew(gray, angle)
    return row_bands(fixed, dark_threshold=dark_threshold, min_fraction=min_fraction, min_height=min_height), angle
