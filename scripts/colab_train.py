"""Kaggle GPU run: fine-tune dh-unibe/trocr-kurrent on line-level Kurrent crops.

Data: line_manifest.jsonl rows {file, bbox:[x,y,w,h], text, project} + page JPGs
+ eval PNGs + eval_manifest.jsonl, all flat in the attached dataset.
Kernel crops lines from page images on the fly (per-worker page cache, pages
processed contiguously in a fixed page-level permutation).
MAX_STEPS=50 -> short timing probe; None -> full run.
"""
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Jupyter/Colab OutStream has no reconfigure
print("STAGE boot", flush=True)
import transformers, torch as _t
print("versions:", transformers.__version__, _t.__version__, flush=True)
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

T0 = time.time()
import glob as _glob
_hits = _glob.glob("/content/data/**/eval_manifest.jsonl", recursive=True)
print("STAGE manifest-search:", _hits, flush=True)
DATA = Path(_hits[0]).parent if _hits else Path("/content/data/kurrent-train-v1")
# pages are split across multiple dataset mounts (Kaggle file-count limit):
PAGE_DIRS = sorted(Path("/content/data").glob("kurrent-train-v1*"))
PAGE_ROOT = {}
for _d in PAGE_DIRS:
    for _f in _d.iterdir():
        if _f.suffix == ".jpg":
            PAGE_ROOT[_f.name] = _f
print("STAGE page-roots:", [str(d) for d in PAGE_DIRS], "pages:", len(PAGE_ROOT), flush=True)
OUT = Path("/content/working")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "dh-unibe/trocr-kurrent"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = None  # full epoch (~8,991 steps ~3h35m T4); 50 for a validation probe
PAD = 6
print(f"STAGE imports-done device={DEVICE} gpu={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'}", flush=True)

print("STAGE processor-load", flush=True)
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
print("STAGE model-download", flush=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
print("STAGE model-ready", flush=True)
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id


def cer(hyp: str, ref: str) -> float:
    rows, cols = len(hyp) + 1, len(ref) + 1
    d = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        d[i][0] = i
    for j in range(cols):
        d[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]))
    return d[-1][-1] / max(1, len(ref))


def word_acc(hyp: str, ref: str) -> float:
    hw, rw = hyp.split(), ref.split()
    return sum(1 for w in rw if w in hw) / max(1, len(rw))


def read_crop(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    pixels = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixels, max_length=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def evaluate(tag: str) -> dict:
    rows = [json.loads(l) for l in (DATA / "eval_manifest.jsonl").open()]
    results = {}
    model.eval()
    for r in rows:
        hyp = read_crop(DATA / r["file"])
        entry = {"hyp": hyp}
        gold = r.get("scored_gold") or ""
        if r["id"] in ("h01", "h07") and gold:
            entry["cer"] = round(cer(hyp, gold), 3)
            entry["word_acc"] = round(word_acc(hyp, gold), 3)
        elif gold:
            if isinstance(gold, list):
                entry["span_hit"] = {g: (g in hyp) for g in gold}
            else:
                entry["span_hit"] = gold in hyp
        results[r["id"]] = entry
    print(f"EVAL {tag}: {json.dumps(results, ensure_ascii=False)}", flush=True)
    return results


print("STAGE eval-before", flush=True)
base = evaluate("before")

samples = [json.loads(l) for l in (DATA / "line_manifest.jsonl").open()]
samples = [s for s in samples if 2 <= len(s.get("text", "").strip()) <= 480]
print(f"train line-samples: {len(samples)}", flush=True)

# --- seed sanity (2026-10-19; verifies ESCHAU seeds landed) ---
from collections import Counter as _Counter
_proj = _Counter(s.get("project", "?") for s in samples)
print("DATA dir:", str(DATA), flush=True)
print("project counts:", dict(_proj), flush=True)
_eschau = sum(1 for s in samples if s.get("project") == "eschau-seeds")
print(f"ESCHAU seed lines: {_eschau}", flush=True)
_seed_files = sorted(set(s["file"] for s in samples if s.get("project") == "eschau-seeds"))
print("ESCHAU seed pages:", _seed_files, flush=True)
_missing = sorted(f for f in set(s["file"] for s in samples) if f not in PAGE_ROOT)
print("missing pages:", _missing if _missing else "none", flush=True)
if _seed_files:
    _im = Image.open(PAGE_ROOT[_seed_files[0]])
    print("seed page resolution:", _seed_files[0], _im.size, flush=True)
    assert _im.size[0] > 1000, "seed page resolution too small - wrong image?"

# Group line indices by page, then iterate pages in a fixed permutation so each
# page image is decoded once per worker per epoch.
by_page = {}
for idx, s in enumerate(samples):
    by_page.setdefault(s["file"], []).append(idx)
pages = sorted(by_page)
rng = random.Random(42)
rng.shuffle(pages)
order = [i for p in pages for i in by_page[p]]
print(f"train pages: {len(pages)}", flush=True)


class LineDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._cache = {}

    def __len__(self):
        return len(order)

    def __getitem__(self, i):
        s = samples[order[i]]
        img = self._cache.get(s["file"])
        if img is None:
            img = Image.open(PAGE_ROOT[s["file"]]).convert("RGB")
            if len(self._cache) >= 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[s["file"]] = img
        x, y, w, h = s["bbox"]
        x0 = max(0, x - PAD)
        y0 = max(0, y - PAD)
        x1 = min(img.width, x + w + PAD)
        y1 = min(img.height, y + h + PAD)
        crop = img.crop((x0, y0, x1, y1))
        pixels = processor(images=crop, return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(s["text"], return_tensors="pt", truncation=True, max_length=500).input_ids[0]
        return {"pixel_values": pixels, "labels": labels}


def collate(batch):
    pixels = torch.stack([b["pixel_values"] for b in batch])
    maxlen = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = b["labels"]
    return {"pixel_values": pixels, "labels": labels}


args = Seq2SeqTrainingArguments(
    output_dir=str(OUT / "ckpt"),
    num_train_epochs=1,
    max_steps=MAX_STEPS if MAX_STEPS else -1,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    fp16=DEVICE == "cuda",
    save_strategy="epoch" if not MAX_STEPS else "no",
    save_total_limit=1,
    logging_steps=10,
    report_to=[],
    dataloader_num_workers=2,
    remove_unused_columns=False,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=LineDataset(), data_collator=collate)
print("STAGE train-start", flush=True)
t_train = time.time()
trainer.train()
train_secs = time.time() - t_train
print(f"train_seconds={train_secs:.0f}", flush=True)
trainer.save_model(str(OUT / "checkpoint-final"))

print("STAGE eval-after", flush=True)
after = evaluate("after")
summary = {"before": base, "after": after, "train_seconds": round(train_secs), "total_seconds": round(time.time() - T0), "samples": len(samples), "pages": len(pages), "max_steps": MAX_STEPS}
(OUT / "timing_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
