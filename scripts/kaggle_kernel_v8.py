"""Kaggle GPU run v4: Gerson-eval + collapse-guard round, resuming from the best available checkpoint.

v4 changes (2026-10-20, after the v2-best checkpoint proved date-collapsed on
Eschau cell crops - register inference produced date-junk):
- checkpoint source is chosen by the NOTEBOOK (v3 best if its timing_result is
  healthy, else v2 best) and extracted to /content/ckpt-best as before
- new held-out eval: eval_names_v4.jsonl = the 4 occupation crops + the Gerson
  Mosbacher row cells (name "Gerson Mosbacher", occupation "Handelsmann",
  date scored as span-hit on ["28","1845"] because the cell is narrative);
  IdNrs 33739/35238 stay out of training (row-level hold-out, pixel-checked)
- per-row eval detail printed (not just means) so collapse is visible
- seeds-fit diagnostic after training: score 12 TRAINED seed cells (fit-check;
  catches a model that cannot reproduce its own training data)

v8 MIX-OVERSAMPLE: MM + 148 clean target cells x40. Data: train_manifest_v4.jsonl + embedded cells (falls back to v3, then v2) + eval_mm006.jsonl +
eval_names_v4.jsonl (falls back to eval_names.jsonl) from the code bundle;
page JPGs from the data mounts; eval_manifest.jsonl + h-item PNGs in DATA.
Set KURRENT_SMOKE=1 for a tiny CPU smoke test (synthetic dirs)."""
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
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

T0 = time.time()
import glob as _glob

SMOKE = os.environ.get("KURRENT_SMOKE") == "1"
CODE = Path(os.environ.get("KURRENT_CODE_DIR", "/kaggle/input/kurrent-code"))
DATA_ROOT = Path(os.environ.get("KURRENT_DATA_ROOT", "/kaggle/input"))

_hits = _glob.glob(str(DATA_ROOT / "**/eval_manifest.jsonl"), recursive=True)
print("STAGE manifest-search:", _hits, flush=True)
DATA = Path(_hits[0]).parent if _hits else DATA_ROOT / "kurrent-train-v1"
# pages are split across multiple dataset mounts (Kaggle file-count limit):
PAGE_DIRS = sorted(DATA_ROOT.glob("kurrent-train-v1*"))
PAGE_ROOT = {}
for _d in PAGE_DIRS:
    for _f in _d.iterdir():
        if _f.suffix == ".jpg":
            PAGE_ROOT[_f.name] = _f
print("STAGE page-roots:", [str(d) for d in PAGE_DIRS], "pages:", len(PAGE_ROOT), flush=True)
OUT = Path(os.environ.get("KURRENT_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "dh-unibe/trocr-kurrent"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if SMOKE:
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 1, 1, int(os.environ.get("KURRENT_SMOKE_EVAL_EVERY", "1")), 0, 1e-5
    MAX_STEPS = int(os.environ.get("KURRENT_SMOKE_STEPS", "1"))
else:
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 1, 8, 500, 100, 5e-6
    MAX_STEPS = None  # full run
PAD = 6
print(f"STAGE imports-done device={DEVICE} gpu={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'} smoke={SMOKE}", flush=True)

print("STAGE processor-load", flush=True)
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
CKPT_SRC = os.environ.get("KURRENT_CKPT_SRC", "/kaggle/input/__nonexistent_no_checkpoint__")  # from-base by design (v4)
print("STAGE model-download", flush=True)
if os.path.exists(os.path.join(CKPT_SRC, "config.json")):
    model = VisionEncoderDecoderModel.from_pretrained(CKPT_SRC).to(DEVICE)
    print(f"STAGE model-ready src={CKPT_SRC} (resume from v2 best)", flush=True)
else:
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"STAGE model-ready src={MODEL_ID} (BASE - checkpoint missing at {CKPT_SRC}!)", flush=True)
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


def read_pil(img: Image.Image) -> str:
    pixels = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixels, max_length=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


_page_cache: dict = {}

def crop_from_page(row: dict) -> Image.Image:
    img = _page_cache.get(row["file"])
    if img is None:
        img = Image.open(PAGE_ROOT[row["file"]]).convert("RGB")
        if len(_page_cache) >= 4:
            _page_cache.pop(next(iter(_page_cache)))
        _page_cache[row["file"]] = img
    x, y, w, h = row["bbox"]
    x0 = max(0, x - PAD)
    y0 = max(0, y - PAD)
    x1 = min(img.width, x + w + PAD)
    y1 = min(img.height, y + h + PAD)
    return img.crop((x0, y0, x1, y1))


def evaluate12(tag: str) -> dict:
    rows = [json.loads(l) for l in (DATA / "eval_manifest.jsonl").open()]
    results = {}
    model.eval()
    for r in rows:
        hyp = read_pil(Image.open(DATA / r["file"]))
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
    print(f"EVAL12 {tag}: {json.dumps(results, ensure_ascii=False)}", flush=True)
    return results


def score12(results: dict) -> float:
    vals = [results[k]["cer"] for k in ("h01", "h07") if "cer" in results.get(k, {})]
    return sum(vals) / len(vals) if vals else float("inf")


def evaluate_names(tag: str) -> dict:
    path = CODE / "eval_names_v4.jsonl"
    if not path.exists():
        path = CODE / "eval_names.jsonl"
    if not path.exists():
        print(f"EVAL_NAMES {tag}: skipped (no eval_names file)", flush=True)
        return {}
    rows = [json.loads(l) for l in path.open()]
    if SMOKE:
        rows = rows[:1]
    model.eval()
    cers, waccs, detail = [], [], {}
    for r in rows:
        hyp = read_pil(crop_from_page(r))
        key = f'{r.get("kind","?")}:{r.get("IdNr","?")}'
        entry = {"hyp": hyp}
        gold_spans = r.get("scored_gold")
        if gold_spans:
            entry["span_hit"] = {g: (g in hyp) for g in gold_spans}
        else:
            c, w = cer(hyp, r["text"]), word_acc(hyp, r["text"])
            entry["cer"] = round(c, 3)
            entry["word_acc"] = round(w, 3)
            cers.append(c)
            waccs.append(w)
        detail[key] = entry
    out = {"n": len(rows), "cer": round(sum(cers) / max(1, len(cers)), 4), "word_acc": round(sum(waccs) / max(1, len(waccs)), 4), "detail": detail}
    print(f"EVAL_NAMES {tag}: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def evaluate500(tag: str) -> dict:
    rows = [json.loads(l) for l in (CODE / "eval_mm006.jsonl").open()]
    if SMOKE:
        rows = rows[:1]
    model.eval()
    cers, waccs = [], []
    for r in rows:
        hyp = read_pil(crop_from_page(r))
        cers.append(cer(hyp, r["text"]))
        waccs.append(word_acc(hyp, r["text"]))
    out = {"n": len(rows), "cer": round(sum(cers) / max(1, len(cers)), 4), "word_acc": round(sum(waccs) / max(1, len(waccs)), 4)}
    print(f"EVAL500 {tag}: {json.dumps(out)}", flush=True)
    return out


print("STAGE eval-before", flush=True)
base12 = evaluate12("before")
base500 = evaluate500("before")
basenames = evaluate_names("before")

_EMB = "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"
import base64 as _b64, gzip as _gz, random as _rnd
_cells = [json.loads(l) for l in _gz.decompress(_b64.b64decode(_EMB)).decode().splitlines()]
print(f"embedded clean target cells: {len(_cells)}", flush=True)
_train_path = CODE / "train_manifest_v4.jsonl"
print("train manifest:", _train_path.name, flush=True)
samples = [json.loads(l) for l in _train_path.open()]
_mm = [s2 for s2 in samples if s2.get("project") not in ("eschau-seeds", "eschau-occ", "eschau-gbirths")]
REPEAT = 40
samples = _mm + _cells * REPEAT
_rnd.seed(42); _rnd.shuffle(samples)
print(f"mix: MM lines {len(_mm)} + target {len(_cells)}x{REPEAT} = {len(samples)}", flush=True)
samples = [s for s in samples if 2 <= len(s.get("text", "").strip()) <= 480]
print(f"train line-samples: {len(samples)}", flush=True)

# --- sanity (2026-10-19 v2: seeds present, MM_1_006 excluded, eval pages resolve) ---
from collections import Counter as _Counter
_proj = _Counter(s.get("project", "?") for s in samples)
print("DATA dir:", str(DATA), flush=True)
print("project counts:", dict(_proj), flush=True)
_held_idnrs = {33739, 35238}
_leaked = sorted({s.get("IdNr") for s in samples if s.get("IdNr") in _held_idnrs})
if not SMOKE:
    assert "MM_1_006" not in _proj, "MM_1_006 leaked into training manifest!"
    assert not _leaked, f"held-out IdNrs leaked into training: {_leaked}"
_eschau = sum(1 for s in samples if s.get("project") == "eschau-seeds")
print(f"ESCHAU seed lines: {_eschau}", flush=True)
_seed_files = sorted(set(s["file"] for s in samples if s.get("project") == "eschau-seeds"))
print("ESCHAU seed pages:", _seed_files, flush=True)
_missing = sorted(f for f in set(s["file"] for s in samples) if f not in PAGE_ROOT)
print("missing pages:", _missing if _missing else "none", flush=True)
_eval_rows = [json.loads(l) for l in (CODE / "eval_mm006.jsonl").open()]
_eval_missing = sorted(set(r["file"] for r in _eval_rows) - set(PAGE_ROOT))
print("eval500 rows:", len(_eval_rows), "eval pages missing:", _eval_missing if _eval_missing else "none", flush=True)
if not SMOKE:
    assert len(_eval_rows) == 500, f"eval set should be 500 rows, got {len(_eval_rows)}"
    assert not _missing, f"training pages missing: {_missing[:5]}"
    assert not _eval_missing, f"eval pages missing: {_eval_missing[:5]}"
if _seed_files:
    _im = Image.open(PAGE_ROOT[_seed_files[0]])
    print("seed page resolution:", _seed_files[0], _im.size, flush=True)
    if not SMOKE:
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
        crop = crop_from_page(s)
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


BEST = {"score": float("inf"), "step": -1}
DRIVE_BEST = os.environ.get("KURRENT_DRIVE_BEST", "")  # e.g. /content/drive/MyDrive/kurrent-colab-results/v3-checkpoint-best


def combined_score(r12: dict, rnames: dict) -> float:
    # FROZEN 2026-10-23: keep-best = held-out Eschau EVAL_NAMES CER only (validation set).
    if rnames and rnames.get("n"):
        return rnames["cer"]
    return score12(r12)


def drive_save_best():
    """Atomic-ish copy of checkpoint-best to Drive: build tmp dir, then rename."""
    if not DRIVE_BEST:
        return
    import shutil
    tmp = DRIVE_BEST + ".tmp"
    try:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(str(OUT / "checkpoint-best"), tmp)
        if os.path.exists(DRIVE_BEST):
            shutil.rmtree(DRIVE_BEST)
        os.rename(tmp, DRIVE_BEST)
        print("DRIVE checkpoint-best updated", flush=True)
    except Exception as e:  # Drive hiccup must not kill training
        print(f"DRIVE best-save failed: {type(e).__name__}: {e}", flush=True)


class KeepBestCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step > 0 and state.global_step % EVAL_EVERY == 0:
            try:
                r = evaluate12(f"step{state.global_step}")
                rn = evaluate_names(f"step{state.global_step}")
                s = combined_score(r, rn)
                if s < BEST["score"]:
                    BEST.update(score=s, step=state.global_step)
                    trainer.save_model(str(OUT / "checkpoint-best"))
                    print(f"NEW BEST step={state.global_step} combinedCER={s:.3f} -> checkpoint-best saved", flush=True)
                    drive_save_best()
            except Exception as e:  # never let keep-best kill a run
                print(f"keep-best eval failed at step {state.global_step}: {type(e).__name__}: {e}", flush=True)
            finally:
                model.train()


args = Seq2SeqTrainingArguments(
    output_dir=str(OUT / "ckpt"),
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS if MAX_STEPS else -1,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=WARMUP,
    fp16=DEVICE == "cuda",
    save_strategy="no",
    logging_steps=10,
    report_to=[],
    dataloader_num_workers=2,
    remove_unused_columns=False,
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=LineDataset(), data_collator=collate, callbacks=[KeepBestCallback()])
print("STAGE train-start", flush=True)
t_train = time.time()
trainer.train()
train_secs = time.time() - t_train
print(f"train_seconds={train_secs:.0f}", flush=True)
trainer.save_model(str(OUT / "checkpoint-final"))

print("STAGE eval-after-final", flush=True)
final12 = evaluate12("after-final")

print("STAGE eval-best", flush=True)
if BEST["step"] > 0:
    try:
        model = VisionEncoderDecoderModel.from_pretrained(str(OUT / "checkpoint-best")).to(DEVICE)
        best12 = evaluate12(f"best-step{BEST['step']}")
        best500 = evaluate500(f"best-step{BEST['step']}")
        bestnames = evaluate_names(f"best-step{BEST['step']}")
    except Exception as e:  # fall back to scoring the final model
        print(f"best-checkpoint load failed: {type(e).__name__}: {e} - scoring final model", flush=True)
        model = trainer.model
        best12 = evaluate12("best-load-failed-final")
        best500 = evaluate500("best-load-failed-final")
        bestnames = evaluate_names("best-load-failed-final")
else:
    print("no improvement over base during training - best = base", flush=True)
    best12, best500, bestnames = evaluate12("best-is-base"), base500, basenames

# seeds-fit diagnostic: can the model reproduce TRAINED seed cells? (fit-check, not generalization)
def evaluate_seeds_fit():
    import random as _rnd
    _rnd.seed(7)
    seed_rows = [s for s in samples if s.get("project") == "eschau-seeds"]
    pick = _rnd.sample(seed_rows, min(12, len(seed_rows)))
    model.eval()
    cers = []
    for r in pick:
        hyp = read_pil(crop_from_page(r))
        cers.append(cer(hyp, r["text"]))
    out = {"n": len(pick), "cer": round(sum(cers) / max(1, len(cers)), 4)}
    print(f"EVAL_SEEDS_FIT: {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out

seeds_fit = evaluate_seeds_fit()

summary = {
    "config": {"lr": LR, "warmup": WARMUP, "epochs": EPOCHS, "eval_every": EVAL_EVERY, "batch": BATCH, "base_model": CKPT_SRC, "resume": True, "left_out": "MM_1_006"},
    "before": {"h12": base12, "mm006_500": base500, "names": basenames},
    "after_final": {"h12": final12},
    "best": {"step": BEST["step"], "combined_cer": round(BEST["score"], 4) if BEST["step"] > 0 else None, "h12": best12, "mm006_500": best500, "names": bestnames},
    "train_seconds": round(train_secs),
    "total_seconds": round(time.time() - T0),
    "samples": len(samples),
    "seeds_fit": seeds_fit,
    "pages": len(pages),
    "max_steps": MAX_STEPS,
}
(OUT / "timing_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("STAGE done", json.dumps(summary["best"]["mm006_500"]), flush=True)
