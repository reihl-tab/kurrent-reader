"""Line segmentation with kraken.

Default mode uses kraken's classic box segmenter, which needs no downloaded
model. A neural baseline model (kraken ``-bl`` mode, any segmentation model
from the kraken model repository) can be selected for harder layouts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class LineBox:
    """One detected text line as an absolute (x0, y0, x1, y1) box."""

    index: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def segment_page(
    image_path: str | Path,
    *,
    mode: str = "boxes",
    model: str | None = None,
    pad: int = 4,
) -> list[LineBox]:
    """Detect text lines in reading order.

    ``mode='boxes'`` runs the model-free classic segmenter; ``mode='baseline'``
    runs the neural segmenter and requires ``model`` (path or kraken repo id).
    """
    im = Image.open(image_path).convert("L")
    if mode == "boxes":
        from kraken import binarization, pageseg
        from kraken.lib.exceptions import KrakenInputException

        # The classic segmenter only accepts bi-level images, and noisy
        # photocopies can exceed its connected-component limit; for those,
        # smooth and shrink the image and scale the boxes back up.
        scale = 1.0
        bilevel = binarization.nlbin(im)
        try:
            seg = pageseg.segment(bilevel, pad=(0, pad))
        except (KrakenInputException, ValueError) as exc:
            if "connected components" not in str(exc):
                raise
            import cv2
            import numpy as np

            arr = np.asarray(im)
            smoothed = cv2.medianBlur(arr, 5)
            scale = 0.5 if max(arr.shape) > 1600 else 1.0
            if scale != 1.0:
                smoothed = cv2.resize(smoothed, None, fx=scale, fy=scale)
            small = Image.fromarray(smoothed)
            seg = pageseg.segment(binarization.nlbin(small), pad=(0, pad))
        raw = [[v / scale for v in line.bbox] for line in seg.lines]
    elif mode == "baseline":
        if model is None:
            raise ValueError("baseline mode needs a segmentation model")
        from kraken import blla
        from kraken.lib import vgsl

        net = vgsl.TorchVGSLModel.load_model(model)
        seg = blla.segment(im, model=net)
        raw = [list(line.bbox) for line in seg.lines]
    else:
        raise ValueError(f"unknown mode: {mode}")
    w, h = im.size
    boxes: list[LineBox] = []
    for i, (x0, y0, x1, y1) in enumerate(raw):
        box = LineBox(
            index=i,
            x0=max(0, int(x0)),
            y0=max(0, int(y0)),
            x1=min(w, int(x1)),
            y1=min(h, int(y1)),
        )
        if box.width >= 8 and box.height >= 8:
            boxes.append(box)
    return boxes


def crop_lines(
    image_path: str | Path,
    boxes: list[LineBox],
    output_dir: str | Path,
    *,
    pad: int = 4,
) -> list[dict[str, object]]:
    """Write one PNG per detected line plus a ``lines.json`` manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(Image.open(image_path).convert("L"))
    manifest: list[dict[str, object]] = []
    for box in boxes:
        y0 = max(0, box.y0 - pad)
        y1 = min(arr.shape[0], box.y1 + pad)
        x0 = max(0, box.x0 - pad)
        x1 = min(arr.shape[1], box.x1 + pad)
        crop = arr[y0:y1, x0:x1]
        name = f"line-{box.index:03d}.png"
        Image.fromarray(crop).save(out_dir / name)
        manifest.append({**asdict(box), "file": name})
    (out_dir / "lines.json").write_text(json.dumps(manifest, indent=2))
    return manifest
