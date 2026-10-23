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

DIAG target-only overfit test. Data: train_manifest_v5_diag.jsonl (falls back to v3, then v2) + eval_mm006.jsonl +
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
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 3, 8, 200, 20, 5e-6
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
import base64 as _b64, gzip as _gz
Path("/kaggle/working/train_manifest_v5_diag.jsonl").write_bytes(_gz.decompress(_b64.b64decode(_EMB)))
_train_path = None
for _mn in ("train_manifest_v5_diag.jsonl", "train_manifest_v4.jsonl"):
    for _base in (Path("/kaggle/working"), CODE):
        _cand = _base / _mn
        if _cand.exists():
            _train_path = _cand
            break
    if _train_path is not None:
        break
print("train manifest:", _train_path.name, flush=True)
samples = [json.loads(l) for l in _train_path.open()]
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
    vals = [score12(r12)]
    if rnames and rnames.get("n"):
        vals.append(rnames["cer"])
    return sum(v for v in vals if v != float("inf")) / max(1, len(vals))


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
