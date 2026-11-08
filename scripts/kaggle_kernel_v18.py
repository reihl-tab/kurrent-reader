"""
v18 = v15/v16 pipeline (clean 301-cell manifest unchanged), from-base,
REPEAT 4 (v16 showed REPEAT 6 regressed names 0.8894->0.9276), MAX_STEPS 500,
EVAL_EVERY 125 (v15 only measured 500/1000/1500/2000 - the 0-500 interval is
unmapped; this run maps the early trajectory to locate the names optimum).
Checkpoint selection: EVAL_NAMES (7-person dev) CER only, now SEEDED with the
init (step-0) score so a trained checkpoint must strictly beat the base model
(v17's selector had no init baseline and picked a names-collapsed step).
"""
import os
import json
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
_b2_dir = DATA_ROOT / "kurrent-batch2"  # v14: also carries g01_22.jpg (not in the v1 mounts)
if _b2_dir.exists():
    PAGE_DIRS.append(_b2_dir)
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
    EPOCHS, BATCH, EVAL_EVERY, WARMUP, LR = 2, 8, 125, 100, 5e-6
    MAX_STEPS = 500  # v16: v15 proved best@500 then overfit (names 0.889->1.080); cap at the productive regime
PAD = 6
print(f"STAGE imports-done device={DEVICE} gpu={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'} smoke={SMOKE}", flush=True)

# --- run-progress heartbeat (issue #30): best-effort, never raises ---
PROGRESS_DATASET = os.environ.get("KURRENT_PROGRESS_DATASET", "reihlinstinct/kurrent-run-progress")
PROGRESS_ENABLED = os.environ.get("KURRENT_PROGRESS", "1") == "1"
_progress_state = {"pushes": 0, "last_error": None}


def progress_payload(stage: str, **fields) -> dict:
    payload = {"stage": stage, "elapsed_sec": round(time.time() - T0, 1)}
    payload.update(fields)
    return payload


def _progress_api():
    """Authenticated Kaggle API: in-kernel session first, then KAGGLE_API_TOKEN
    (e.g. injected as a Kaggle User Secret)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _json_safe(value):
    """Replace non-finite floats with None so json.dumps never emits Infinity/NaN."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def progress_push(payload: dict) -> bool:
    """Version the PRIVATE progress dataset with this payload as progress.json.

    Best-effort: returns False instead of raising so training never dies on a
    heartbeat failure. Payload policy (issue #30 review): aggregates only, no
    per-row hypotheses or gold text.
    """
    if not PROGRESS_ENABLED:
        return False
    import shutil
    import tempfile
    d = None
    try:
        d = Path(tempfile.mkdtemp(prefix="kurrent-progress-"))
        payload = _json_safe(payload)
        (d / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "dataset-metadata.json").write_text(json.dumps({
            "title": PROGRESS_DATASET.split("/")[-1],
            "id": PROGRESS_DATASET,
            "licenses": [{"name": "other"}],
        }), encoding="utf-8")
        _progress_api().dataset_create_version(str(d), str(payload.get("stage", "progress")), dir_mode="skip")
        _progress_state["pushes"] += 1
        print(f"PROGRESS pushed: {payload.get('stage')} (total {_progress_state['pushes']})", flush=True)
        return True
    except Exception as e:
        _progress_state["last_error"] = type(e).__name__
        print(f"PROGRESS push failed (non-fatal): {_progress_state['last_error']}", flush=True)
        return False
    finally:
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)



print("STAGE processor-load", flush=True)
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
CKPT_SRC = os.environ.get("KURRENT_CKPT_SRC", "/kaggle/input/__nonexistent_no_checkpoint__")  # from-base by design (v8)
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

# --- v13 gentle-training block (issue #37): exactly one variant per run ---
GENTLE = os.environ.get("KURRENT_GENTLE", "lowlr")
if GENTLE not in ("lowlr", "freeze-encoder"):
    raise ValueError(f"unknown KURRENT_GENTLE variant: {GENTLE!r}")
if not SMOKE:
    if GENTLE == "lowlr":
        LR = 1e-6  # was 5e-6 in v8-v12
    else:  # freeze-encoder
        for _p in model.encoder.parameters():
            _p.requires_grad_(False)
_frozen = sum(1 for _p in model.parameters() if not _p.requires_grad)
print(f"STAGE gentle mode={GENTLE} lr={LR} frozen_params={_frozen}", flush=True)


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


def _safe_data_path(root, rel):
    """Resolve rel under root, rejecting paths that escape it."""
    base = root.resolve()
    p = (base / rel).resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"unsafe path escapes data root: {rel!r}")
    return p


def evaluate12(tag: str) -> dict:
    rows = [json.loads(l) for l in (DATA / "eval_manifest.jsonl").open()]
    results = {}
    model.eval()
    for r in rows:
        hyp = read_pil(Image.open(_safe_data_path(DATA, r["file"])))
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

import random as _rnd

_held_idnrs = {"33739", "35238", "38666", "38678", "38693", "38696"}  # row-level hold-out: Gerson row + EVAL_NAMES persons
# v15: single clean merged manifest (repo labeling/batch2/merged_v15_manifest.json)
# replaces the embedded blob + batch-1 mount + batch-2 dataset v1 (see header).
_merged_path = DATA_ROOT / "kurrent-batch2" / "batch2_manifest.jsonl"
_cells = [json.loads(l) for l in _merged_path.open()]
print(f"merged v15 target cells: {len(_cells)} ({_merged_path})", flush=True)
_drop = [c for c in _cells if str(c.get("IdNr")) in _held_idnrs]
if _drop:
    print(f"held-out decontamination: dropping {len(_drop)} cells for persons {sorted({str(c['IdNr']) for c in _drop})}", flush=True)
    _cells = [c for c in _cells if str(c.get("IdNr")) not in _held_idnrs]
_gm = [c for c in _cells if str(c.get("text", "")).strip() == "Gerson Mosbacher"]
if _gm:
    print(f"text-level held-out leak: dropping {len(_gm)} cells carrying eval-target text", flush=True)
    _cells = [c for c in _cells if str(c.get("text", "")).strip() != "Gerson Mosbacher"]
if not SMOKE:
    assert len(_cells) == int(os.environ.get("KURRENT_EXPECT_CELLS", "301")), f"expected 301 merged v15 cells, got {len(_cells)}"
_train_path = CODE / "train_manifest_v4.jsonl"
print("train manifest:", _train_path.name, flush=True)
samples = [json.loads(l) for l in _train_path.open()]
_mm = [s2 for s2 in samples if s2.get("project") not in ("eschau-seeds", "eschau-occ", "eschau-gbirths")]
REPEAT = 4  # v18: back to v15 weight (v16 x6 regressed names 0.8894->0.9276)  # v12 (issue #32): was 40 in v8-v10; x40 drove memorization (seeds_fit 0.0, held-out names worse than base)
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
# _held_idnrs is defined at the embedded-cells load above (v14); assert below still guards batch-1/batch-2 cells
_leaked = sorted({str(s.get("IdNr")) for s in samples if str(s.get("IdNr")) in _held_idnrs})
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
progress_push(progress_payload("boot", train_samples=len(samples), target_cells=len(_cells),
                               missing_pages=len(_missing), eval_pages_missing=len(_eval_missing)))
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


BEST = {"score": basenames.get("cer", float("inf")) if "basenames" in dir() else float("inf"), "step": 0}  # v18: init baseline - a trained ckpt must beat step-0 names CER (v17 had no baseline and picked a names-collapsed step)
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
                    print(f"NEW BEST step={state.global_step} namesCER={s:.3f} -> checkpoint-best saved", flush=True)
                    drive_save_best()
                progress_push(progress_payload(
                    f"step{state.global_step}", step=state.global_step,
                    score12=round(score12(r), 4),
                    names_n=rn.get("n"), names_cer=rn.get("cer"), names_word_acc=rn.get("word_acc"),
                    names_cer_sel=round(s, 4), best_step=BEST["step"], best_score=BEST["score"]))
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
    "best": {"step": BEST["step"], "names_cer": round(BEST["score"], 4) if BEST["step"] > 0 else None, "h12": best12, "mm006_500": best500, "names": bestnames},
    "train_seconds": round(train_secs),
    "total_seconds": round(time.time() - T0),
    "samples": len(samples),
    "seeds_fit": seeds_fit,
    "pages": len(pages),
    "max_steps": MAX_STEPS,
}
(OUT / "timing_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
progress_push(progress_payload("train-done", result=str(OUT / "timing_result.json")))
print("STAGE done", json.dumps(summary["best"]["mm006_500"]), flush=True)
