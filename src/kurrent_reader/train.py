"""Fine-tune dh-unibe/trocr-kurrent on labeled Kurrent line crops.

Designed for weekly Kaggle runs: free-tier GPU, quota reset, resume from the
previous week's checkpoint. Data is a folder of PNG line crops plus a JSONL
manifest ({"file": ..., "text": ...}); labels must come from verified
transcriptions only (see issue #3 label rules). The held-out eval manifest
(quorum-scored crops) is loaded separately and never enters training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "dh-unibe/trocr-kurrent"


@dataclass
class LineSample:
    file: Path
    text: str


def load_manifest(path: Path, image_dir: Path) -> list[LineSample]:
    """Load a JSONL manifest of {"file", "text"} rows, skipping blank labels."""
    samples: list[LineSample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            image = image_dir / str(row["file"])
            if text and image.exists():
                samples.append(LineSample(file=image, text=text))
    return samples


def exclude_files(samples: list[LineSample], blocked: set[str]) -> list[LineSample]:
    """Drop any sample whose filename is in the held-out blocklist."""
    return [s for s in samples if s.file.name not in blocked]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate: Levenshtein distance over reference length."""
    rows = len(hypothesis) + 1
    cols = len(reference) + 1
    dist: NDArray[np.int64] = np.zeros((rows, cols), dtype=np.int64)
    dist[:, 0] = np.arange(rows)
    dist[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = int(hypothesis[i - 1] != reference[j - 1])
            dist[i, j] = min(dist[i - 1, j] + 1, dist[i, j - 1] + 1, dist[i - 1, j - 1] + cost)
    return float(dist[rows - 1, cols - 1]) / max(1, len(reference))


def word_accuracy(hypothesis: str, reference: str) -> float:
    """Share of reference words present verbatim in the hypothesis."""
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0
    return sum(1 for w in ref_words if w in hyp_words) / len(ref_words)


def build_trainer(
    train_manifest: Path,
    image_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL,
    resume_from: Path | None = None,
    epochs: float = 3.0,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
) -> Any:
    """Assemble the HF Trainer (heavy imports deferred to call time)."""
    import torch
    from PIL import Image
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        default_data_collator,
    )

    processor = TrOCRProcessor.from_pretrained(model_id)
    source = str(resume_from) if resume_from is not None else model_id
    model = VisionEncoderDecoderModel.from_pretrained(source)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    samples = load_manifest(train_manifest, image_dir)
    if not samples:
        raise ValueError(f"no labeled samples in {train_manifest}")

    class LineDataset(torch.utils.data.Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            sample = samples[idx]
            image = Image.open(sample.file).convert("RGB")
            pixels = processor(images=image, return_tensors="pt").pixel_values[0]
            labels = processor.tokenizer(sample.text, return_tensors="pt").input_ids[0]
            return {"pixel_values": pixels, "labels": labels}

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=LineDataset(),
        data_collator=default_data_collator,
    )
