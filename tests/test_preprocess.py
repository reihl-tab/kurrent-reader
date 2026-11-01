"""Tests for the cleanup chain; all synthetic, no external data needed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from kurrent_reader import preprocess


def _lined_page(angle_deg: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """A white page with horizontal ink lines, optionally rotated and noisy."""
    rng = np.random.default_rng(7)
    img = np.full((600, 800), 235, np.uint8)
    for y in range(80, 560, 60):
        cv2.line(img, (60, y), (740, y), color=20, thickness=3)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D((400, 300), angle_deg, 1.0)
        img = cv2.warpAffine(img, matrix, (800, 600), borderValue=(235, 235, 235))
    if noise:
        specks = rng.random(img.shape) < noise
        img[specks] = 0
    return img


def test_deskew_recovers_rotation() -> None:
    rotated = _lined_page(angle_deg=3.0)
    angle = preprocess.estimate_skew_angle(rotated)
    assert abs(angle + 3.0) < 0.6
    fixed = preprocess.deskew(rotated, angle)
    assert abs(preprocess.estimate_skew_angle(fixed)) < 0.6


def test_binarize_returns_two_values() -> None:
    img = _lined_page(noise=0.002)
    binary = preprocess.binarize(img)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})


def test_denoise_removes_speckle() -> None:
    img = _lined_page(noise=0.01)
    cleaned = preprocess.denoise(img)
    specks_before = int((img == 0).sum())
    specks_after = int((cleaned == 0).sum())
    assert specks_after < specks_before // 3


def test_clean_does_not_mutate_input() -> None:
    img = _lined_page()
    before = img.copy()
    preprocess.clean(img)
    assert np.array_equal(img, before)


def test_process_file_keeps_original_bytes(tmp_path: Path) -> None:
    src = tmp_path / "page.jpg"
    cv2.imwrite(str(src), _lined_page(angle_deg=2.0))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    meta = preprocess.process_file(src, tmp_path / "out")
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest
    assert Path(str(meta["cleaned"])).exists()
    assert Path(str(meta["binarized"])).exists()
    assert meta["cleaned"] != str(src)


def test_estimate_skew_rejects_blank_pages() -> None:
    blank = np.full((400, 400), 255, np.uint8)
    assert preprocess.estimate_skew_angle(blank) == 0.0


def test_to_gray_handles_color() -> None:
    color = np.full((10, 10, 3), 128, np.uint8)
    gray = preprocess.to_gray(color)
    assert gray.ndim == 2
