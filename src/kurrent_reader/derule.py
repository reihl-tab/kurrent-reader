"""Table-rule removal for tabular register pages.

Many 19th-century registers are printed tables: the text sits inside ruled
cells, and the ruling lines cut through handwriting strokes. Line-level
readers such as TrOCR Kurrent then see a mixture of ink and rule and
produce garbage. This module detects the printed ruling lines with
morphological opening, erases them by inpainting, and cuts ruled bands
into cell images that read like ordinary text lines.

The input image is never modified; all outputs are new arrays/files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


@dataclass
class DeruleResult:
    """Output of rule removal plus what was detected."""

    deruled: GrayImage
    line_mask: GrayImage
    vertical_positions: list[int]
    horizontal_positions: list[int]
    steps: list[str] = field(default_factory=list)


def _binarize_inv(gray: GrayImage) -> GrayImage:
    """Adaptive inverted binarisation (ink/rules white on black)."""
    return cast(
        GrayImage,
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
        ),
    )


def detect_ruling_lines(
    gray: GrayImage,
    *,
    line_fraction: float = 0.2,
    min_length: int = 15,
    dilate: int = 2,
) -> GrayImage:
    """Return a mask (uint8, 255 = ruling line) of long straight rules.

    A rule must run for at least ``line_fraction`` of the image width
    (horizontal rules) or height (vertical rules) to count, so ordinary
    stroke connections inside handwriting are not eaten.
    """
    inv = _binarize_inv(gray)
    h, w = gray.shape
    horiz_len = max(min_length, int(w * line_fraction))
    vert_len = max(min_length, int(h * line_fraction))
    horiz = cv2.morphologyEx(
        inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    )
    vert = cv2.morphologyEx(
        inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    )
    mask = cast(GrayImage, cv2.bitwise_or(horiz, vert))
    if dilate > 0:
        mask = cast(
            GrayImage,
            cv2.dilate(mask, np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)),
        )
    return mask


def _rule_positions(mask: GrayImage, axis: int, *, coverage: float = 0.5) -> list[int]:
    """Centres of detected rules along ``axis`` (0 = vertical rules -> x).

    A column/row qualifies when its mask-pixel count reaches ``coverage``
    of the strongest column/row; adjacent qualifying lines merge into one
    position.
    """
    profile = mask.sum(axis=axis).astype(np.float64) / 255.0
    if profile.size == 0 or profile.max() == 0:
        return []
    threshold = float(profile.max()) * coverage
    hits = np.where(profile >= threshold)[0]
    positions: list[int] = []
    group = [int(hits[0])]
    for idx in hits[1:]:
        if int(idx) - group[-1] <= 3:
            group.append(int(idx))
        else:
            positions.append(int(round(sum(group) / len(group))))
            group = [int(idx)]
    positions.append(int(round(sum(group) / len(group))))
    return positions


def remove_ruling_lines(
    gray: GrayImage,
    *,
    line_fraction: float = 0.2,
    min_length: int = 15,
    dilate: int = 2,
    inpaint_radius: int = 3,
) -> DeruleResult:
    """Erase detected ruling lines and inpaint the gaps.

    Inpainting (rather than whitening) keeps handwriting strokes continuous
    where a printed rule crossed them.
    """
    mask = detect_ruling_lines(
        gray, line_fraction=line_fraction, min_length=min_length, dilate=dilate
    )
    deruled = cast(
        GrayImage, cv2.inpaint(gray, mask, inpaint_radius, cv2.INPAINT_TELEA)
    )
    vertical = _rule_positions(mask, axis=0)
    horizontal = _rule_positions(mask, axis=1)
    steps = [
        f"detect-{len(vertical)}v-{len(horizontal)}h",
        f"inpaint-telea-r{inpaint_radius}",
    ]
    return DeruleResult(
        deruled=deruled,
        line_mask=mask,
        vertical_positions=vertical,
        horizontal_positions=horizontal,
        steps=steps,
    )


def split_columns(
    gray: GrayImage, vertical_positions: list[int], *, pad: int = 3, min_width: int = 12
) -> list[tuple[int, int]]:
    """Column x-ranges between vertical rules, padded away from the rules."""
    w = gray.shape[1]
    bounds = [0, *vertical_positions, w]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        x0 = left if left == 0 else min(left + pad, right)
        x1 = right if right == w else max(right - pad, x0)
        if x1 - x0 >= min_width:
            spans.append((x0, x1))
    return spans


def split_words(
    gray: GrayImage, *, min_width: int = 8, ink_threshold: int = 128, gap: int = 12
) -> list[tuple[int, int]]:
    """Word x-ranges inside one de-ruled cell or line.

    Column projection of the binarised image; runs of ink columns separated
    by gaps narrower than ``gap`` pixels merge into one word. Returns
    ``(x0, x1)`` spans in reading order.
    """
    binary = (gray < ink_threshold).astype(np.uint8)
    profile = binary.sum(axis=0)
    ink_cols = np.where(profile > 0)[0]
    if ink_cols.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(ink_cols[0])
    for col in ink_cols[1:]:
        col = int(col)
        if col - prev > gap:
            spans.append((start, prev + 1))
            start = col
        prev = col
    spans.append((start, prev + 1))
    return [(x0, x1) for x0, x1 in spans if x1 - x0 >= min_width]


def pad_to_strip(
    word: GrayImage, *, target_height: int = 64, pad_x: int = 16, pad_y: int = 8
) -> GrayImage:
    """Pad/resize a word crop into a line-like strip for the line reader.

    The word keeps its aspect ratio, is scaled to ``target_height`` minus
    vertical padding, and sits on a white canvas, mimicking the line crops
    the reader was trained on.
    """
    h, w = word.shape
    scale = (target_height - 2 * pad_y) / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(word, (new_w, target_height - 2 * pad_y), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((target_height, new_w + 2 * pad_x), 255, np.uint8)
    canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + new_w] = resized
    return canvas


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    line_fraction: float = 0.2,
    split: bool = False,
) -> dict[str, object]:
    """De-rule one image and write ``<stem>.deruled.png`` + rule mask.

    With ``split=True`` also writes one ``<stem>.cell-NN.png`` crop per
    column between the detected vertical rules. The source file is opened
    read-only and never overwritten.
    """
    src = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cast(GrayImage, cv2.imread(str(src), cv2.IMREAD_GRAYSCALE))
    if image is None:
        raise ValueError(f"cannot read image: {src}")
    result = remove_ruling_lines(image, line_fraction=line_fraction)
    deruled_path = out_dir / f"{src.stem}.deruled.png"
    mask_path = out_dir / f"{src.stem}.rulemask.png"
    cv2.imwrite(str(deruled_path), result.deruled)
    cv2.imwrite(str(mask_path), result.line_mask)
    cells: list[str] = []
    if split:
        for i, (x0, x1) in enumerate(split_columns(image, result.vertical_positions)):
            cell_path = out_dir / f"{src.stem}.cell-{i:02d}.png"
            cv2.imwrite(str(cell_path), result.deruled[:, x0:x1])
            cells.append(str(cell_path))
    return {
        "source": str(src),
        "deruled": str(deruled_path),
        "rule_mask": str(mask_path),
        "vertical_positions": result.vertical_positions,
        "horizontal_positions": result.horizontal_positions,
        "cells": cells,
        "steps": result.steps,
    }
