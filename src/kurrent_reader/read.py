"""Line reading with the TrOCR Kurrent model (dh-unibe/trocr-kurrent).

The model is line-level: one cropped text line in, one string out. CPU
inference is the default; ``quantize='int8'`` applies dynamic quantization
so the 1.3 GB fp32 model fits and runs on small machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

DEFAULT_MODEL = "dh-unibe/trocr-kurrent"


@dataclass
class ReadResult:
    """Transcription of one line image with a coarse confidence score."""

    file: str
    text: str
    confidence: float


class LineReader:
    """Lazy-loaded TrOCR reader for batches of line images."""

    def __init__(self, model_id: str = DEFAULT_MODEL, quantize: str | None = None) -> None:
        self.model_id = model_id
        self.quantize = quantize
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self.model_id)
        available_mb = _available_memory_mb()
        if self.quantize == "int8" and available_mb < 2600:
            # fp32 weights (1.3 GB) plus runtime overhead do not fit on small
            # machines, and torch cannot quantize fp16 tensors, so fall back
            # to plain fp16 inference (~670 MB weights).
            model = VisionEncoderDecoderModel.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_id, dtype=torch.float16, low_cpu_mem_usage=True
            )
        else:
            model = VisionEncoderDecoderModel.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_id, low_cpu_mem_usage=True
            )
            if self.quantize == "int8":
                model = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif self.quantize is not None:
                raise ValueError(f"unknown quantize mode: {self.quantize}")
        model.eval()
        self._model = model

    def read(self, image_path: str | Path) -> ReadResult:
        """Read a single line image."""
        return self.read_many([image_path])[0]

    def read_many(self, image_paths: Sequence[str | Path], max_new_tokens: int = 64) -> list[ReadResult]:
        """Read several line images, one at a time (small-memory machines)."""
        import torch

        self._load()
        assert self._processor is not None and self._model is not None
        results: list[ReadResult] = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            pixels = self._processor(images=image, return_tensors="pt").pixel_values
            with torch.inference_mode():
                generated = self._model.generate(
                    pixels,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            ids = generated.sequences[0]
            text = self._processor.batch_decode([ids], skip_special_tokens=True)[0]
            confidence = _mean_token_probability(ids, generated.scores)
            results.append(
                ReadResult(file=str(path), text=text.strip(), confidence=round(confidence, 4))
            )
        return results


def _available_memory_mb() -> int:
    """MemAvailable from /proc/meminfo; a large sentinel when unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def _mean_token_probability(ids: Any, scores: Any) -> float:
    """Mean softmax probability of the emitted tokens; 0 when unavailable."""
    import torch

    if not scores:
        return 0.0
    probs: list[float] = []
    # ids[0] is the decoder start token; generated ids[1:] align with scores.
    for token_id, step_scores in zip(ids[1:], scores):
        prob = torch.softmax(step_scores[0].float(), dim=-1)[int(token_id)]
        probs.append(float(prob))
    if not probs:
        # Scores can be missing (e.g. early EOS under inference_mode); an
        # honest "unknown" is -1.0, never a fake zero-confidence reading.
        return -1.0
    return math.exp(sum(math.log(max(p, 1e-9)) for p in probs) / len(probs))
