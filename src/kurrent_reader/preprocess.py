"""Scan cleanup for historical register pages.

Removes photographic/scan noise before segmentation and reading:
contrast normalisation, denoising, deskewing and binarisation. The input
image is never modified; cleaned variants are returned as new arrays and
written to new files by :func:`process_file`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_sauvola

GrayImage = NDArray[np.uint8]


@dataclass
class CleanResult:
    """Output of the cleanup chain plus what was done."""

    cleaned_gray: GrayImage
    binarized: GrayImage
    deskew_angle_deg: float
    steps: list[str] = field(default_factory=list)


def to_gray(image: NDArray[np.uint8]) -> GrayImage:
    """Convert any uint8 image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cast(GrayImage, cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY))
    return cast(GrayImage, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def normalize_contrast(gray: GrayImage, clip_limit: float = 2.0, tile: int = 16) -> GrayImage:
    """Even out illumination with CLAHE so stained areas stay readable."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return cast(GrayImage, clahe.apply(gray))


def denoise(gray: GrayImage, strength: int = 7) -> GrayImage:
    """Suppress scan noise/grain while keeping stroke edges.

    A median pre-pass removes salt-and-pepper specks cheaply; non-local
    means then smooths the remaining grain. NLM is the slow step, so it runs
    on a copy shrunk to at most ~1500 px on the long side and is blended
    back at full resolution by reusing the estimated noise level: for our
    documents the median pass alone removes most of the damaging speckle,
    so NLM only runs when the image is small enough.
    """
    med = cast(GrayImage, cv2.medianBlur(gray, 3))
    if max(med.shape) <= 2000:
        return cast(GrayImage, cv2.fastNlMeansDenoising(med, None, strength, 7, 21))
    return med


def estimate_skew_angle(gray: GrayImage, max_angle: float = 10.0) -> float:
    """Estimate page skew in degrees via the minimum-area rectangle of ink.

    Positive angle means the text is rotated clockwise and the correction
    must rotate counter-clockwise by the returned angle.
    """
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if coords.shape[0] < 100:
        return 0.0
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew(gray: GrayImage, angle_deg: float) -> GrayImage:
    """Rotate the image to cancel the measured skew, canvas kept constant."""
    if abs(angle_deg) < 0.05:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cast(
        GrayImage,
        cv2.warpAffine(
            gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        ),
    )


def binarize(gray: GrayImage, window: int = 41, k: float = 0.2) -> GrayImage:
    """Sauvola binarisation; robust to the brown/gray stains of old scans."""
    win = window if window % 2 == 1 else window + 1
    thresh = threshold_sauvola(gray.astype(np.float64) / 255.0, window_size=win, k=k)  # type: ignore[no-untyped-call]
    binary = (gray.astype(np.float64) / 255.0 > thresh).astype(np.uint8) * 255
    return cast(GrayImage, binary)


def clean(
    image: NDArray[np.uint8],
    *,
    do_denoise: bool = True,
    do_deskew: bool = True,
    do_contrast: bool = True,
) -> CleanResult:
    """Run the full cleanup chain on a copy of ``image``."""
    steps: list[str] = []
    gray = to_gray(image.copy())
    if do_contrast:
        gray = normalize_contrast(gray)
        steps.append("contrast-clahe")
    if do_denoise:
        gray = denoise(gray)
        steps.append("denoise-median-nlm")
    angle = estimate_skew_angle(gray) if do_deskew else 0.0
    if do_deskew and not math.isclose(angle, 0.0, abs_tol=0.05):
        gray = deskew(gray, angle)
        steps.append(f"deskew-{angle:+.2f}deg")
    binary = binarize(gray)
    steps.append("binarize-sauvola")
    return CleanResult(cleaned_gray=gray, binarized=binary, deskew_angle_deg=angle, steps=steps)


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    **kwargs: bool,
) -> dict[str, object]:
    """Clean one scan and write ``<stem>.clean.png`` + ``<stem>.bin.png``.

    The source file is opened read-only and never overwritten.
    """
    src = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cast(NDArray[np.uint8], cv2.imread(str(src), cv2.IMREAD_UNCHANGED))
    if image is None:
        raise ValueError(f"cannot read image: {src}")
    result = clean(image, **kwargs)
    gray_path = out_dir / f"{src.stem}.clean.png"
    bin_path = out_dir / f"{src.stem}.bin.png"
    cv2.imwrite(str(gray_path), result.cleaned_gray)
    cv2.imwrite(str(bin_path), result.binarized)
    return {
        "source": str(src),
        "cleaned": str(gray_path),
        "binarized": str(bin_path),
        "deskew_angle_deg": result.deskew_angle_deg,
        "steps": result.steps,
    }
