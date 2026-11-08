"""v20 = v18/v19 pipeline with the arm-B enlarged target set: 306 cells (301 merged
v15 + 5 second-hand right-page date cells: g10/38752, g10/3664, g07/38779,
g07/38653, g07/38765). Rule extension approved by user 2026-11-08 08:35
(corroborated time words + parent-name cells at medium; applied: all time words
here uncorroborated -> excluded; weekdays only where calendar-verified + legible).
Recipe: lowlr LR 1e-6, REPEAT 4, from base, MAX_STEPS 500, EVAL_EVERY 125,
init-seeded names-CER selection (v18 trajectory: peak @250, overfit @375+).
Manifest embedded below (merged_armb_manifest.json); held-out asserts unchanged.
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
GENTLE = os.environ.get("KURRENT_GENTLE", "lowlr")  # v20: arm-B data, lowlr (v18 recipe; v19 freeze-encoder regressed)
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
_merged_path = "EMBEDDED arm-B manifest (repo labeling/batch2/merged_armb_manifest.json)"
_ARMB_BLOB = '''{"file": "g01_22.jpg", "bbox": [310, 512, 300, 48], "text": "Beile Marx", "IdNr": 38736, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#1"}
{"file": "g01_22.jpg", "bbox": [1055, 815, 155, 60], "text": "Joseph Mosbacher", "IdNr": 33729, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [460, 640, 330, 52], "text": "23.11.1821", "IdNr": 33729, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [560, 815, 185, 60], "text": "Merjem Mosbacher", "IdNr": 38747, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [460, 640, 330, 52], "text": "23.11.1821", "IdNr": 38747, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1821#2"}
{"file": "g01_22.jpg", "bbox": [940, 1105, 170, 43], "text": "Beile Michel", "IdNr": 38737, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#1"}
{"file": "g01_22.jpg", "bbox": [320, 1000, 325, 56], "text": "11.06.1822", "IdNr": 38737, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#1"}
{"file": "g01_22.jpg", "bbox": [320, 1290, 1080, 130], "text": "Julia Schönlein Strauß", "IdNr": 38314, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#2"}
{"file": "g01_22.jpg", "bbox": [400, 1230, 300, 60], "text": "28.05.1822", "IdNr": 38314, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#2"}
{"file": "g01_22.jpg", "bbox": [760, 1635, 250, 53], "text": "Wolf Strauß", "IdNr": 38813, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#3"}
{"file": "g01_22.jpg", "bbox": [420, 1543, 347, 55], "text": "29.07.1822", "IdNr": 38813, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#3"}
{"file": "g01_22.jpg", "bbox": [660, 1853, 340, 62], "text": "Vrommet Michel", "IdNr": 38742, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "1822#4"}
{"file": "g01_22.jpg", "bbox": [420, 1773, 313, 55], "text": "29.11.1822", "IdNr": 38742, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1822#4"}
{"file": "g02_22.jpg", "bbox": [140, 543, 205, 118], "text": "Elias Mosbacher", "IdNr": 34023, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n34023", "note": "given name 'Elias' legible, surname line faint/abbreviated in ink; label from index. TRIPLE dup index persons 34023/35707/38748 = same entry, labeled once (34023). Date pairing unique (25.06.1823); bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 703, 195, 95], "text": "Hajum Strauß", "IdNr": 37361, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n37361", "note": "ink faint; index 'Strauss, Hajum (Chaim Sohn des David Halevi)'; date pairing 18.08.1823; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 913, 180, 87], "text": "Jakob Michel", "IdNr": 38015, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38015", "note": "small mark line below name (possible death-note start, d.18.07.1825 per index) excluded from bbox; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1073, 180, 90], "text": "Kindle Michel", "IdNr": 38018, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38018", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1273, 180, 110], "text": "Vromeln Michel", "IdNr": 38022, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n38022", "note": "given name reads 'Vromeln'?/'Vromme'?; index 'Michel, NN, Vromeln'; date pairing 27.02.1825; bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1453, 180, 100], "text": "Abraham Strauß", "IdNr": 38784, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38784", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1603, 180, 95], "text": "Vromel Michel", "IdNr": 38738, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38738", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [140, 1813, 180, 120], "text": "Isaak Strasburger", "IdNr": 38031, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38031", "note": "bbox width clipped at col1/col2 rule (father column excluded)"}
{"file": "g02_22.jpg", "bbox": [1840, 538, 405, 115], "text": "den 25ten Juni 1823, 11 Uhr", "IdNr": 34023, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d34023", "note": "hour qualifier word illegible - omitted; no weekday words in this register; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 683, 405, 100], "text": "den 18ten August 1823, Morgens", "IdNr": 37361, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d37361", "note": "day numeral ambiguous in ink (reads '19ten'?; index 18.08.1823, label from index); hour numeral illegible - omitted; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 893, 405, 80], "text": "den 3ten October 1824, 11 Uhr", "IdNr": 38015, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38015", "note": "hour qualifier word illegible - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1078, 405, 115], "text": "den 1ten November 1824, Morgens", "IdNr": 38018, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38018", "note": "hour numeral weak - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1273, 405, 80], "text": "den 27ten Februar 1825, Mittags 1 Uhr", "IdNr": 38022, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38022", "note": "month word weak; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1448, 405, 125], "text": "den 18ten November 1825, Morgens", "IdNr": 38784, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38784", "note": "hour numeral illegible - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1595, 405, 158], "text": "den 24ten Juni 1826, Morgens", "IdNr": 38738, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38738", "note": "hour numeral weak - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g02_22.jpg", "bbox": [1840, 1822, 405, 156], "text": "den 7ten August 1826, 5 Uhr", "IdNr": 38031, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38031", "note": "hour qualifier word faint - omitted; numerals from index; bbox x1840-2245: excludes left col bleed and right col rule (~x2250)"}
{"file": "g04_22.jpg", "bbox": [150, 365, 280, 135], "text": "Veila Mosbacher", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38037", "note": "ink reads 'Veile', index 'Veila'; label from index"}
{"file": "g04_22.jpg", "bbox": [150, 540, 280, 142], "text": "Mendel Mosbacher", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n3763", "note": "duplicate index person 3764 = same entry, labeled once (3763); the g12/g13 father as a CHILD here (b.18.10.1831)"}
{"file": "g04_22.jpg", "bbox": [150, 741, 280, 148], "text": "Jakob Michel", "IdNr": 38043, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38043"}
{"file": "g04_22.jpg", "bbox": [150, 1255, 280, 145], "text": "Moses Strauß", "IdNr": 38794, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-n38794", "note": "surname heavy flourish, semi-legible"}
{"file": "g04_22.jpg", "bbox": [150, 1450, 280, 115], "text": "Joseph Mosbacher", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6403", "note": "duplicate index person 38095 = same entry, labeled once (6403); later-hand death note '+ d. 9 Aug. 1835' below name EXCLUDED from bbox"}
{"file": "g04_22.jpg", "bbox": [150, 1700, 280, 120], "text": "Isaak Strauß", "IdNr": 38097, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38097", "note": "later-hand small repeat annotation below excluded from bbox"}
{"file": "g04_22.jpg", "bbox": [150, 1880, 280, 137], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38098", "note": "d.08.04.1837 per index (also s03 death register person)"}
{"file": "g04_22.jpg", "bbox": [2100, 370, 400, 146], "text": "23.04.1831", "IdNr": 38037, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38037", "note": "'d 23t Apr: 1831 Morgens 1 Uhr' legible; no weekday words in this register"}
{"file": "g04_22.jpg", "bbox": [2100, 541, 400, 139], "text": "18.10.1831", "IdNr": 3763, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d3763", "note": "'d 18t Oct: 1831 Morgens 8 Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 712, 400, 130], "text": "19.12.1831", "IdNr": 38043, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38043", "note": "'d 19t Dec 1831 Mittags 4 Uhr' - day numeral weak, month/year legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1148, 400, 96], "text": "14.06.1833", "IdNr": 38794, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38794", "note": "'d 14ten Jun 1833 Abends 4 Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1252, 400, 152], "text": "18.08.1833", "IdNr": 6403, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6403", "note": "'d 18t Aug 1833 Abends 6(?) Uhr' legible"}
{"file": "g04_22.jpg", "bbox": [2100, 1660, 400, 200], "text": "15.01.1835", "IdNr": 38097, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38097", "note": "'d 15t Jan ...5 Abends 6 Uhr' - year tail weak"}
{"file": "g04_22.jpg", "bbox": [2100, 1924, 400, 176], "text": "28.11.1835", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38098", "note": "'d 28t Nov 1835 Abends 9 Uhr' legible"}
{"file": "g05_22.jpg", "bbox": [160, 171, 240, 382], "text": "Rebecca Mosbacher", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.0-1"}
{"file": "g05_22.jpg", "bbox": [2050, 170, 420, 200], "text": "10.01.1836", "IdNr": 38099, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.1-2"}
{"file": "g05_22.jpg", "bbox": [2050, 572, 420, 179], "text": "15.09.1837", "IdNr": 38100, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "g05_22.jpg", "bbox": [160, 744, 240, 191], "text": "Emanuel Mosbacher", "IdNr": 38101, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "g05_22.jpg", "bbox": [2050, 751, 420, 179], "text": "07.04.1838", "IdNr": 38101, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.4"}
{"file": "g05_22.jpg", "bbox": [160, 935, 240, 191], "text": "Josef Leopold Mosbacher", "IdNr": 6407, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.4"}
{"file": "g05_22.jpg", "bbox": [2050, 930, 420, 179], "text": "08.04.1841", "IdNr": 6407, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g05_22.jpg", "bbox": [160, 1126, 240, 191], "text": "Therese Regensburger", "IdNr": 38103, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.5"}
{"file": "g05_22.jpg", "bbox": [2050, 1109, 420, 179], "text": "06.06.1841", "IdNr": 38103, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.6"}
{"file": "g05_22.jpg", "bbox": [160, 1317, 240, 191], "text": "Aaron Strauß", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g05_22.jpg", "bbox": [2050, 1288, 420, 179], "text": "14.09.1841", "IdNr": 38106, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.7"}
{"file": "g05_22.jpg", "bbox": [160, 1508, 240, 191], "text": "Resel Mosbacher", "IdNr": 38107, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "g05_22.jpg", "bbox": [2050, 1467, 420, 179], "text": "21.10.1841", "IdNr": 38107, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.8"}
{"file": "g05_22.jpg", "bbox": [160, 1699, 240, 191], "text": "Emanuel Mosbacher", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.8"}
{"file": "g05_22.jpg", "bbox": [2050, 1646, 420, 179], "text": "28.01.1843", "IdNr": 38113, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "g05_22.jpg", "bbox": [160, 1890, 240, 191], "text": "Ernestine Regensburger", "IdNr": 38198, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.9"}
{"file": "g05_22.jpg", "bbox": [2050, 1825, 420, 179], "text": "28.01.1843", "IdNr": 38198, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "g05_22.jpg", "bbox": [160, 2025, 240, 162], "text": "Abraham Mosbacher", "IdNr": 38199, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "g05_22.jpg", "bbox": [2050, 1950, 420, 225], "text": "29.02.1844", "IdNr": 38199, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11"}
{"file": "g06_22.jpg", "bbox": [160, 219, 240, 183], "text": "Sara Strauß", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.1"}
{"file": "g06_22.jpg", "bbox": [2050, 230, 420, 190], "text": "02.05.1844", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.1"}
{"file": "g06_22.jpg", "bbox": [160, 402, 240, 183], "text": "Isaak Mosbacher", "IdNr": 38309, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.2"}
{"file": "g06_22.jpg", "bbox": [2050, 420, 420, 190], "text": "09.01.1845", "IdNr": 38309, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.2"}
{"file": "g06_22.jpg", "bbox": [160, 585, 240, 183], "text": "David Regensburger", "IdNr": 38763, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "g06_22.jpg", "bbox": [2050, 610, 420, 190], "text": "22.02.1845", "IdNr": 38763, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "g06_22.jpg", "bbox": [160, 768, 240, 183], "text": "Koppel Strauß", "IdNr": 38814, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.4"}
{"file": "g06_22.jpg", "bbox": [2050, 800, 420, 190], "text": "02.05.1845", "IdNr": 38814, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.4"}
{"file": "g06_22.jpg", "bbox": [160, 951, 240, 183], "text": "Babeta Mosbacher", "IdNr": 38646, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.5"}
{"file": "g06_22.jpg", "bbox": [2050, 990, 420, 190], "text": "02.06.1846", "IdNr": 38646, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g06_22.jpg", "bbox": [160, 1134, 240, 183], "text": "Edel Strauß", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g06_22.jpg", "bbox": [2050, 1180, 420, 190], "text": "11.11.1846", "IdNr": 38796, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6"}
{"file": "g06_22.jpg", "bbox": [160, 1317, 240, 183], "text": "Abraham Regensburger", "IdNr": 38764, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "g06_22.jpg", "bbox": [2050, 1370, 420, 190], "text": "03.02.1847", "IdNr": 38764, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.7"}
{"file": "g06_22.jpg", "bbox": [160, 1500, 240, 183], "text": "Daniel Strauß", "IdNr": 38815, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.8"}
{"file": "g06_22.jpg", "bbox": [2050, 1560, 420, 190], "text": "18.05.1847", "IdNr": 38815, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.8"}
{"file": "g06_22.jpg", "bbox": [160, 1683, 240, 183], "text": "Fanny Mosbacher", "IdNr": 23186, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.9"}
{"file": "g06_22.jpg", "bbox": [2050, 1750, 420, 190], "text": "21.06.1847", "IdNr": 23186, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.9"}
{"file": "g06_22.jpg", "bbox": [160, 1866, 240, 183], "text": "Leopold Rothschild", "IdNr": 38766, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "g06_22.jpg", "bbox": [2050, 1940, 420, 190], "text": "23.06.1847", "IdNr": 38766, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "g07_22.jpg", "bbox": [81, 690, 271, 190], "text": "Karoline Mosbacher", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3-top"}
{"file": "g07_22.jpg", "bbox": [1908, 690, 348, 190], "text": "12.07.1849", "IdNr": 38744, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "0.3-top"}
{"file": "g08_22.jpg", "bbox": [150, 270, 355, 65], "text": "Jette Michel", "IdNr": 38739, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38739"}
{"file": "g08_22.jpg", "bbox": [150, 415, 355, 55], "text": "Fanny Strauß", "IdNr": 38785, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38785"}
{"file": "g08_22.jpg", "bbox": [150, 530, 355, 90], "text": "Gittel Mosbacher", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38745", "note": "index name 'Mosbacher, Gitta,Getta (Gittel Gattin des Menachem)'; label matches ink 'Gittel'"}
{"file": "g08_22.jpg", "bbox": [150, 750, 355, 150], "text": "Moses Regensburger", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38664", "note": "scribe's struck false-start line above included in bbox; ink order 'Regensburger / Moses'; person also appears on s05"}
{"file": "g08_22.jpg", "bbox": [150, 985, 355, 130], "text": "Gidel Mosbacher", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38751", "note": "first line struck through by scribe (false start reading '...Strauss...'); label 'Gidel Mosbacher'"}
{"file": "g08_22.jpg", "bbox": [150, 1205, 355, 90], "text": "Elias Michel", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38740"}
{"file": "g08_22.jpg", "bbox": [150, 1525, 355, 95], "text": "Mendel Mosbacher", "IdNr": 38756, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38756"}
{"file": "g08_22.jpg", "bbox": [150, 1760, 355, 110], "text": "Lazarus Strauß", "IdNr": 38799, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38799"}
{"file": "g08_22.jpg", "bbox": [2100, 268, 560, 120], "text": "den 7ten März 1852, früh 8 Uhr", "IdNr": 38739, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38739", "note": "weekday word illegible, omitted from label (no single-reader guesses); 07.03.1852 was a Sunday; numerals from index; 'März' and 'früh 8 Uhr' legible"}
{"file": "g08_22.jpg", "bbox": [2100, 398, 560, 100], "text": "Freitag den 8ten Mai 1852, Mittags 3 Uhr", "IdNr": 38785, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38785", "note": "weekday 'Freitag' legible but 08.05.1852 was a Saturday - scribe error, documented; numerals from index"}
{"file": "g08_22.jpg", "bbox": [2100, 542, 560, 130], "text": "Montag den 21. Juni 1852, Mittags 4 Uhr", "IdNr": 38745, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38745", "note": "weekday cross-check PASS (21.06.1852 = Monday)"}
{"file": "g08_22.jpg", "bbox": [2100, 668, 560, 88], "text": "Dienstag den 30ten November 1852, Mittags 2 Uhr", "IdNr": 38750, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38750", "note": "weekday cross-check PASS (30.11.1852 = Tuesday); NAME CELL EXCLUDED: near-duplicate of held-out eval name 'Gerson Mosbacher' (33739); flagged for parent ruling"}
{"file": "g08_22.jpg", "bbox": [2100, 798, 560, 135], "text": "Am Sonnabend den 27ten May 1854, Morgens 5 Uhr", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38664", "note": "weekday cross-check PASS (27.05.1854 = Saturday); year '1854' on separate line"}
{"file": "g08_22.jpg", "bbox": [2100, 1042, 560, 150], "text": "am Freitag d. 18ten August 1854, Morgens 4 Uhr", "IdNr": 38751, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38751", "note": "weekday cross-check PASS (18.08.1854 = Friday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1208, 560, 195], "text": "am Donnerstag d. 28ten September 1854, Morgens 4 Uhr", "IdNr": 38740, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38740", "note": "weekday cross-check PASS (28.09.1854 = Thursday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1448, 560, 315], "text": "Freitag den 29ten December 1854, Vormittags 10 Uhr", "IdNr": 38756, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38756", "note": "certificate header line ('...No 10') inside bbox, not transcribed; hour qualifier faint (reads 'Vormittags'?); weekday+date cross-check PASS (29.12.1854 = Friday)"}
{"file": "g08_22.jpg", "bbox": [2100, 1762, 560, 140], "text": "am Mittwoch d. 30ten May 1855, Morgens", "IdNr": 38799, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38799", "note": "hour numeral illegible, omitted from label; weekday cross-check PASS (30.05.1855 = Wednesday)"}
{"file": "g09_22.jpg", "bbox": [252, 840, 145, 125], "text": "Gina Michel", "IdNr": 38741, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38741", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; two-line 'Gina / Michel'"}
{"file": "g09_22.jpg", "bbox": [250, 1250, 130, 70], "text": "David", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace (cert N.104), twins with 38787"}
{"file": "g09_22.jpg", "bbox": [245, 1340, 175, 70], "text": "Jeanette", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked; surname Strauss in shared parents brace, twins with 38786"}
{"file": "g09_22.jpg", "bbox": [210, 1595, 280, 100], "text": "Maier Joseph Mosbacher", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; two-line 'Maier Joseph / Mosbacher'"}
{"file": "g09_22.jpg", "bbox": [210, 1838, 180, 52], "text": "Babette", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; given name only as inked ('posthuma' entry); surname Strauss from index/parent phrase"}
{"file": "g09_22.jpg", "bbox": [1900, 855, 700, 155], "text": "12.10.1856", "IdNr": 38741, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38741", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; no weekday word written; 'Am 12ten ... October 1856 ... 6 Uhr' read; 12.10.1856 = Sunday (no cross-check possible)"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38786, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38786", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38787; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS"}
{"file": "g09_22.jpg", "bbox": [1900, 1215, 700, 240], "text": "20.10.1856", "IdNr": 38787, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38787", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; SHARED date cell with twin 38786; 'Am Montag den zwanzigsten October 1856' - Montag legible, 20.10.1856 = Monday PASS"}
{"file": "g09_22.jpg", "bbox": [1900, 1610, 700, 190], "text": "09.03.1857", "IdNr": 38746, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38746", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Montag ... Maerz 1857, Nachts 12 Uhr' - Montag legible, 09.03.1857 = Monday PASS; day numeral from index, pinned by weekday+month+year"}
{"file": "g09_22.jpg", "bbox": [1900, 1825, 750, 125], "text": "08.11.1857", "IdNr": 38800, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38800", "pairing": "cross-book", "pairing_note": "name cell from left birth register (no dates there); date cell from right parallel cert book (weekday/date/time); paired via index person + weekday cross-check where weekday word written; 'Am Sonntag d. 8ten Nov. 1857 Abends' - Sonntag legible, 08.11.1857 = Sunday PASS"}
{"file": "g10_22.jpg", "bbox": [140, 350, 330, 110], "text": "Isaak Mosbacher", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38752"}
{"file": "g10_22.jpg", "bbox": [140, 560, 330, 95], "text": "Mina Strauß", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38675", "note": "index 'Strauß, Wilhelmine, Wilhelmine (Mina)'; ink reads 'Mina Strauß' (call name documented in index)"}
{"file": "g10_22.jpg", "bbox": [140, 845, 330, 95], "text": "Jendele Marx", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n3664", "note": "index 'Marx, Jette, Jendele'; ink reads 'Jendele Marx'; DUP index person 35455 = same entry, labeled once; small-print naming note below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [140, 1015, 330, 85], "text": "Manuel Mosbacher", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38674", "note": "death note '+ d. 6 Jan. 1860' below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [140, 1315, 330, 95], "text": "Elieser Marx", "IdNr": 38733, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38733"}
{"file": "g10_22.jpg", "bbox": [140, 1495, 330, 90], "text": "Mendel Strauß", "IdNr": 38788, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38788"}
{"file": "g10_22.jpg", "bbox": [140, 1610, 330, 90], "text": "Abraham Mosbacher", "IdNr": 38753, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38753"}
{"file": "g10_22.jpg", "bbox": [140, 1862, 330, 78], "text": "Veist Strauß", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38677", "note": "index 'Strauß, Veist (Faist, Feist)'; death note '+ 31 Marz 1862' below excluded from bbox"}
{"file": "g10_22.jpg", "bbox": [1745, 382, 325, 240], "text": "den 17ten Januar 1858, 10 Uhr", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38752", "note": "weekday word illegible - omitted (17.01.1858 was a Sunday); hour qualifier faint; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 693, 325, 137], "text": "Am Mittwoch den 21ten July 1858", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d38675", "note": "weekday cross-check PASS (21.07.1858 = Wednesday)"}
{"file": "g10_22.jpg", "bbox": [1745, 853, 325, 160], "text": "Am Mittwoch den 15ten September 1858, Morgens 4 Uhr", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "L-d3664", "note": "weekday cross-check PASS (15.09.1858 = Wednesday)"}
{"file": "g10_22.jpg", "bbox": [1745, 1038, 325, 185], "text": "den 19ten December 1859, Mittags 11 Uhr", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38674", "note": "weekday word illegible - omitted (19.12.1859 was a Monday); numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1293, 325, 155], "text": "den 4ten Februar 1860, Morgens 2 Uhr", "IdNr": 38733, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38733", "note": "weekday word illegible - omitted (04.02.1860 was a Saturday); ink word order '...2 Uhr / Morgens'; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1483, 325, 145], "text": "den 4ten April 1860, Morgens 3 Uhr", "IdNr": 38788, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38788", "note": "weekday word illegible - omitted (04.04.1860 was a Wednesday); ink word order '...3 Uhr / Morgens'; numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1640, 325, 115], "text": "den 8ten December 1860, Nachmittags 3 Uhr", "IdNr": 38753, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38753", "note": "weekday word illegible - omitted (08.12.1860 was a Saturday); hour qualifier faint ('Nachmittags'?); numerals from index"}
{"file": "g10_22.jpg", "bbox": [1745, 1862, 325, 140], "text": "den 30ten Januar 1862, 6 Uhr", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "L-d38677", "note": "weekday and hour-qualifier words faint - omitted (30.01.1862 was a Thursday); numerals from index"}
{"file": "g11_22.jpg", "bbox": [225, 460, 220, 110], "text": "Sara Strauß", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r1", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [235, 720, 200, 95], "text": "Felix Mosbacher", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r2", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [235, 890, 300, 140], "text": "Jeanette Strauß", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r3", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [210, 1200, 230, 135], "text": "Louis Mosbacher", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r4", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [185, 1475, 220, 135], "text": "David Strauß", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r5", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [210, 1620, 250, 130], "text": "Amalie Mosbacher", "IdNr": 38757, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r6", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [195, 1825, 290, 90], "text": "Jeanette Strauß", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L-r7", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [230, 1935, 240, 100], "text": "Fanny Strauß", "IdNr": 38808, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-r8", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 490, 420, 210], "text": "15.09.1863", "IdNr": 38789, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1863", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 735, 420, 150], "text": "18.03.1864", "IdNr": 3766, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1864", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 955, 420, 240], "text": "30.08.1865", "IdNr": 38806, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1865", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1210, 420, 165], "text": "14.09.1866", "IdNr": 6446, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1866", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1505, 420, 150], "text": "06.07.1867", "IdNr": 38807, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-1867", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1780, 420, 105], "text": "28.01.1869", "IdNr": 38757, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869a", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1880, 420, 95], "text": "25.01.1869", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869b", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g11_22.jpg", "bbox": [1740, 1960, 420, 140], "text": "12.02.1869", "IdNr": 38808, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-1869c", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 495, 390, 105], "text": "Siegmund Strauß", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38697", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 670, 390, 95], "text": "Siegmund Strauß", "IdNr": 38790, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38790", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 795, 390, 95], "text": "Isaak Strauß", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38699", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 905, 390, 95], "text": "Rickchen Mosbacher", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1065, 390, 100], "text": "Abraham Strauß", "IdNr": 38781, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38781", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1185, 390, 115], "text": "Jeanette Strauß", "IdNr": 38791, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38791", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1380, 390, 90], "text": "Sara Strauß", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38698", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1615, 390, 85], "text": "Zerline Mosbacher", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1735, 390, 90], "text": "Regine Strauß", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [70, 1850, 390, 135], "text": "Lina Mosbacher", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag"}
{"file": "g12_22.jpg", "bbox": [1560, 555, 900, 125], "text": "15.03.1870", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38697", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Dienstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 690, 900, 125], "text": "05.04.1870", "IdNr": 38790, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38790", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday word illegible; numeral 5 + April clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 830, 900, 120], "text": "04.06.1870", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38699", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Samstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 950, 900, 120], "text": "02.06.1870", "IdNr": 6449, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6449", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1140, 900, 115], "text": "24.03.1871", "IdNr": 38781, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38781", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday reads 'Sonntag' but 24.03.1871=Friday - clerk error or misread; numeral 24 + März clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 1255, 900, 145], "text": "21.09.1871", "IdNr": 38791, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R-d38791", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday semi-legible (best read Donnerstag, matches calendar Thursday); numeral 21 + September clear, matches index"}
{"file": "g12_22.jpg", "bbox": [1560, 1421, 900, 139], "text": "14.05.1872", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38698", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Dienstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1614, 900, 115], "text": "15.08.1872", "IdNr": 6412, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6412", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Donnerstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1743, 900, 102], "text": "05.10.1872", "IdNr": 38809, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38809", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; weekday cross-check Samstag OK"}
{"file": "g12_22.jpg", "bbox": [1560, 1865, 900, 130], "text": "18.06.1872", "IdNr": 6500, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d6500", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right parallel cert book; paired via index person + weekday cross-check or medium flag; cell reads 'Dienstag den 8ten[struck] achtzehnten Juni' + 1897 margin correction confirming 18; Dienstag OK for 18.06.1872"}
{"file": "g13_22.jpg", "bbox": [230, 630, 350, 140], "text": "Jeanette Strauß", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 815, 350, 115], "text": "Jeta Mosbacher", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1020, 350, 120], "text": "Moses Strauß", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1195, 350, 110], "text": "Ruben Strauß", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1365, 350, 125], "text": "Selma Strauß", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1520, 350, 125], "text": "Felix Mosbacher", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [230, 1670, 350, 115], "text": "Friederike Strauß", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "L-n38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check"}
{"file": "g13_22.jpg", "bbox": [1520, 700, 940, 140], "text": "21.10.1873", "IdNr": 38783, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38783", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 840, 940, 140], "text": "21.12.1873", "IdNr": 38754, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38754", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1100, 940, 130], "text": "06.01.1874", "IdNr": 38792, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38792", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Dienstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1250, 940, 100], "text": "26.02.1874", "IdNr": 38810, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38810", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Donnerstag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1455, 940, 110], "text": "07.05.1875", "IdNr": 38782, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38782", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Freitag OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1575, 940, 110], "text": "25.08.1875", "IdNr": 38755, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38755", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Mittwoch OK"}
{"file": "g13_22.jpg", "bbox": [1520, 1695, 940, 110], "text": "19.09.1875", "IdNr": 38793, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R-d38793", "pairing": "cross-book", "pairing_note": "name cell from left summary register (no dates there); date cell from right cert book (weekday+date+time); paired via index person + weekday cross-check; weekday cross-check Sonntag OK"}
{"file": "s01_22.jpg", "bbox": [373, 425, 118, 72], "text": "Strauß", "IdNr": 38312, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "s01-n1", "pairing": "narrative", "pairing_note": "narrative death entry 1821; full name line-broken (Kop-pel Strauß), surname word labeled; date not in index, unlabeled per single-reader rule"}
{"file": "s02_22.jpg", "bbox": [157, 495, 235, 164], "text": "Giedle Mosbacher", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R3@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 495, 474, 164], "text": "28.06.1823", "IdNr": 38315, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R3"}
{"file": "s02_22.jpg", "bbox": [157, 659, 235, 164], "text": "Salomon Mosbacher", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R4@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 659, 474, 164], "text": "04.10.1823", "IdNr": 38622, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R4"}
{"file": "s02_22.jpg", "bbox": [157, 823, 235, 164], "text": "Riefke Rothschild", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R5@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 823, 474, 164], "text": "09.04.1825", "IdNr": 38623, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R5"}
{"file": "s02_22.jpg", "bbox": [157, 987, 235, 164], "text": "Löw Rothschild", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R6@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 987, 474, 164], "text": "11.05.1825", "IdNr": 38624, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R6"}
{"file": "s02_22.jpg", "bbox": [157, 1315, 235, 164], "text": "Ruchme Mosbacher", "IdNr": 38625, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "R8@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1315, 474, 164], "text": "08.04.1827", "IdNr": 38625, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R8"}
{"file": "s02_22.jpg", "bbox": [157, 1479, 235, 164], "text": "Gundle Mosbacher", "IdNr": 38626, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "R9@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1479, 474, 164], "text": "07.12.1827", "IdNr": 38626, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "R9"}
{"file": "s02_22.jpg", "bbox": [157, 1807, 235, 164], "text": "Lorenz Mosbacher", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R11@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1807, 474, 164], "text": "29.03.1829", "IdNr": 38627, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R11"}
{"file": "s02_22.jpg", "bbox": [157, 1971, 235, 164], "text": "Michael Michel", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "R12@sameY"}
{"file": "s02_22.jpg", "bbox": [1296, 1971, 474, 164], "text": "21.12.1829", "IdNr": 38628, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "R12"}
{"file": "s03_22.jpg", "bbox": [147, 526, 226, 164], "text": "Gudle Straus", "IdNr": 38633, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "s03_22.jpg", "bbox": [1292, 501, 442, 167], "text": "26.11.1831", "IdNr": 38633, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "s03_22.jpg", "bbox": [147, 1018, 226, 164], "text": "Jakob Mehnl", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "s03_22.jpg", "bbox": [1292, 1002, 442, 167], "text": "22.05.1834", "IdNr": 38636, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6"}
{"file": "s03_22.jpg", "bbox": [147, 1510, 226, 164], "text": "Jonas Mosbacher", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.9"}
{"file": "s03_22.jpg", "bbox": [1292, 1503, 442, 167], "text": "09.08.1835", "IdNr": 38639, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "s03_22.jpg", "bbox": [147, 1674, 226, 164], "text": "Michelle Michael", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.10"}
{"file": "s03_22.jpg", "bbox": [1292, 1670, 442, 167], "text": "26.06.1836", "IdNr": 38640, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.10"}
{"file": "s03_22.jpg", "bbox": [147, 1838, 226, 164], "text": "Madel Mosbacher", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.11"}
{"file": "s03_22.jpg", "bbox": [1292, 1837, 442, 167], "text": "08.04.1837", "IdNr": 38098, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.11"}
{"file": "s05_22.jpg", "bbox": [150, 190, 390, 105], "text": "Isaak Michel", "IdNr": 35242, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 155, 370, 135], "text": "06.05.1850", "IdNr": 35242, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 385, 390, 65], "text": "Jette Marx", "IdNr": 38649, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 295, 370, 170], "text": "22.05.1850", "IdNr": 38649, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 490, 390, 80], "text": "Jette Michel", "IdNr": 38651, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 485, 370, 140], "text": "22.07.1850", "IdNr": 38651, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 635, 390, 80], "text": "Eva Rothschild", "IdNr": 38652, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 635, 370, 117], "text": "20.10.1850", "IdNr": 38652, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 720, 390, 115], "text": "Mina Mosbacher", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 755, 370, 130], "text": "28.04.1853", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 945, 390, 80], "text": "Simson Michel", "IdNr": 38654, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 950, 370, 155], "text": "08.09.1853", "IdNr": 38654, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1055, 390, 130], "text": "Vaibel Mosbacher", "IdNr": 38041, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1150, 370, 185], "text": "30.12.1853", "IdNr": 38041, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1330, 390, 105], "text": "David Strauß", "IdNr": 38655, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1340, 370, 140], "text": "20.07.1854", "IdNr": 38655, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1445, 390, 125], "text": "Moses Regensburger", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1490, 370, 135], "text": "13.12.1854", "IdNr": 38664, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s05_22.jpg", "bbox": [150, 1595, 390, 110], "text": "Mendel Mosbacher", "IdNr": 38665, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "L"}
{"file": "s05_22.jpg", "bbox": [1330, 1650, 370, 185], "text": "29.12.1854", "IdNr": 38665, "project": "eschau1081-batch2", "kind": "date", "conf": "high", "band": "R"}
{"file": "s06_22.jpg", "bbox": [140, 134, 420, 220], "text": "Kehle Mosbacher", "IdNr": 38667, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.0"}
{"file": "s06_22.jpg", "bbox": [1290, 250, 240, 178], "text": "28.01.1856", "IdNr": 38667, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.0-1"}
{"file": "s06_22.jpg", "bbox": [1290, 428, 240, 149], "text": "14.04.1856", "IdNr": 38668, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.2"}
{"file": "s06_22.jpg", "bbox": [140, 574, 420, 220], "text": "Jeanette Strauß", "IdNr": 38669, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.2"}
{"file": "s06_22.jpg", "bbox": [1290, 577, 240, 149], "text": "26.10.1856", "IdNr": 38669, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3"}
{"file": "s06_22.jpg", "bbox": [140, 794, 420, 220], "text": "Alexander Strauß", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.3"}
{"file": "s06_22.jpg", "bbox": [1290, 900, 240, 124], "text": "19.10.1857", "IdNr": 38035, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.5"}
{"file": "s06_22.jpg", "bbox": [140, 1234, 420, 220], "text": "Sara Strauß", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.5"}
{"file": "s06_22.jpg", "bbox": [1290, 1090, 240, 170], "text": "19.03.1858", "IdNr": 38671, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.6-7"}
{"file": "s06_22.jpg", "bbox": [140, 1454, 420, 220], "text": "Seligmann Hirsch", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "s06_22.jpg", "bbox": [1290, 1322, 240, 149], "text": "11.08.1858", "IdNr": 38672, "project": "eschau1081-batch2", "kind": "date", "conf": "verified", "band": "1.8"}
{"file": "s06_22.jpg", "bbox": [140, 1674, 420, 220], "text": "Mattche Marx", "IdNr": 38673, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "s06_22.jpg", "bbox": [1290, 1620, 240, 149], "text": "27.02.1859", "IdNr": 38673, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "s06_22.jpg", "bbox": [140, 1894, 420, 220], "text": "Manuel Mosbacher", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.8"}
{"file": "s06_22.jpg", "bbox": [1290, 1910, 270, 230], "text": "06.01.1860", "IdNr": 38674, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11-12"}
{"file": "s07_22.jpg", "bbox": [150, 142, 430, 160], "text": "Rachel Strauß", "IdNr": 38676, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.0"}
{"file": "s07_22.jpg", "bbox": [1300, 145, 320, 155], "text": "31.05.1860", "IdNr": 38676, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.0"}
{"file": "s07_22.jpg", "bbox": [150, 302, 430, 160], "text": "Wilhelmine Strauß", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.1"}
{"file": "s07_22.jpg", "bbox": [1300, 430, 320, 180], "text": "24.03.1861", "IdNr": 38675, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.1-2"}
{"file": "s07_22.jpg", "bbox": [150, 622, 430, 160], "text": "Veist Strauß", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.3"}
{"file": "s07_22.jpg", "bbox": [1300, 615, 320, 160], "text": "31.03.1862", "IdNr": 38677, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.3-4"}
{"file": "s07_22.jpg", "bbox": [150, 1262, 430, 160], "text": "Jeanette Strauß", "IdNr": 38036, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.7"}
{"file": "s07_22.jpg", "bbox": [1300, 1225, 320, 185], "text": "22.11.1864", "IdNr": 38036, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.7"}
{"file": "s07_22.jpg", "bbox": [150, 1742, 430, 160], "text": "Jeanette Strauß", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.10"}
{"file": "s07_22.jpg", "bbox": [1300, 1685, 320, 160], "text": "08.02.1869", "IdNr": 38694, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.10"}
{"file": "s07_22.jpg", "bbox": [150, 1902, 430, 160], "text": "Jeanette Strauß", "IdNr": 38695, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "0.11"}
{"file": "s07_22.jpg", "bbox": [1300, 1845, 320, 160], "text": "11.02.1869", "IdNr": 38695, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "1.11"}
{"file": "s08_22.jpg", "bbox": [150, 500, 410, 165], "text": "Siegmund Strauß", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "2"}
{"file": "s08_22.jpg", "bbox": [1300, 490, 870, 190], "text": "31.05.1870", "IdNr": 38697, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "2"}
{"file": "s08_22.jpg", "bbox": [150, 705, 410, 110], "text": "Charlotte Mosbacher", "IdNr": 38042, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "3"}
{"file": "s08_22.jpg", "bbox": [1300, 695, 870, 185], "text": "08.07.1871", "IdNr": 38042, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "3"}
{"file": "s08_22.jpg", "bbox": [150, 925, 410, 150], "text": "Sara Strauß", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "4"}
{"file": "s08_22.jpg", "bbox": [1300, 900, 870, 215], "text": "05.06.1872", "IdNr": 38698, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "4"}
{"file": "s08_22.jpg", "bbox": [150, 1225, 410, 180], "text": "Clara Mosbacher", "IdNr": 38039, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "5"}
{"file": "s08_22.jpg", "bbox": [1300, 1180, 870, 235], "text": "02.04.1874", "IdNr": 38039, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "5"}
{"file": "s08_22.jpg", "bbox": [150, 1430, 410, 145], "text": "Isaak Strauß", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "name", "conf": "medium", "band": "6"}
{"file": "s08_22.jpg", "bbox": [1300, 1425, 870, 135], "text": "15.11.1875", "IdNr": 38699, "project": "eschau1081-batch2", "kind": "date", "conf": "medium", "band": "6"}
{"file": "g07_22.jpg", "bbox": [81, 150, 271, 181], "text": "Regina Mosbacher", "IdNr": 38749, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.1"}
{"file": "g07_22.jpg", "bbox": [81, 331, 271, 334], "text": "Josef Rothschild", "IdNr": 1945, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.2"}
{"file": "g07_22.jpg", "bbox": [81, 1096, 271, 174], "text": "Madel Strauß", "IdNr": 38816, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.4"}
{"file": "g07_22.jpg", "bbox": [81, 1407, 271, 199], "text": "Mina Mosbacher", "IdNr": 38653, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.6"}
{"file": "g07_22.jpg", "bbox": [81, 1606, 271, 138], "text": "Sarah Rothschild", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "name", "conf": "verified", "band": "0.7"}
{"file": "g07_22.jpg", "bbox": [1908, 162, 348, 161], "text": "25.03.1848", "IdNr": 38749, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.1"}
{"file": "g07_22.jpg", "bbox": [1908, 323, 348, 333], "text": "25.01.1849", "IdNr": 1945, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.2"}
{"file": "g07_22.jpg", "bbox": [1908, 1086, 348, 173], "text": "25.10.1849", "IdNr": 38816, "project": "eschau1081-batch1", "kind": "date", "conf": "medium", "band": "1.5"}
{"file": "g07_22.jpg", "bbox": [1908, 1394, 348, 200], "text": "28.03.1851", "IdNr": 38653, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.7"}
{"file": "g07_22.jpg", "bbox": [1908, 1594, 348, 137], "text": "08.04.1851", "IdNr": 38780, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.8"}
{"file": "g07_22.jpg", "bbox": [1908, 1731, 348, 156], "text": "05.01.1852", "IdNr": 38765, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.9"}
{"file": "g07_22.jpg", "bbox": [1908, 1887, 348, 253], "text": "08.02.1852", "IdNr": 38798, "project": "eschau1081-batch1", "kind": "date", "conf": "verified", "band": "1.10"}
{"file": "s04_22.jpg", "bbox": [150, 92, 280, 232], "text": "Rachel Strauß", "project": "eschau-seeds", "kind": "name", "IdNr": 38641}
{"file": "s04_22.jpg", "bbox": [150, 296, 280, 165], "text": "Raphael Simon", "project": "eschau-seeds", "kind": "name", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [890, 322, 360, 157], "text": "30.03.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [150, 605, 280, 173], "text": "Wolf Mehnl", "project": "eschau-seeds", "kind": "name", "IdNr": 38644}
{"file": "s04_22.jpg", "bbox": [890, 631, 360, 165], "text": "14.04.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38644}
{"file": "s04_22.jpg", "bbox": [150, 433, 280, 200], "text": "Gudle Straus", "project": "eschau-seeds", "kind": "name", "IdNr": 38643}
{"file": "s04_22.jpg", "bbox": [890, 459, 360, 192], "text": "17.04.1840", "project": "eschau-seeds", "kind": "date", "IdNr": 38643}
{"file": "s04_22.jpg", "bbox": [150, 750, 280, 195], "text": "Isaak Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 38309}
{"file": "s04_22.jpg", "bbox": [890, 776, 360, 173], "text": "19.01.1845", "project": "eschau-seeds", "kind": "date", "IdNr": 38309, "conf": "medium", "note": "faint/partially illegible ink; band+row verified 2026-11-06; numerals index-anchored"}
{"file": "s04_22.jpg", "bbox": [150, 1309, 280, 176], "text": "Babeta Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 38646}
{"file": "s04_22.jpg", "bbox": [890, 1335, 360, 168], "text": "14.06.1846", "project": "eschau-seeds", "kind": "date", "IdNr": 38646}
{"file": "s04_22.jpg", "bbox": [150, 1457, 280, 208], "text": "Jakob Solinger", "project": "eschau-seeds", "kind": "name", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [890, 1483, 360, 200], "text": "03.02.1848", "project": "eschau-seeds", "kind": "date", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [150, 1637, 280, 178], "text": "Sara Rothschild", "project": "eschau-seeds", "kind": "name", "IdNr": 38648}
{"file": "s04_22.jpg", "bbox": [890, 1663, 360, 170], "text": "13.04.1848", "project": "eschau-seeds", "kind": "date", "IdNr": 38648}
{"file": "s04_22.jpg", "bbox": [150, 1787, 280, 189], "text": "Berez Löb Mosbacher", "project": "eschau-seeds", "kind": "name", "IdNr": 6400}
{"file": "s04_22.jpg", "bbox": [890, 1813, 360, 181], "text": "20.04.1849", "project": "eschau-seeds", "kind": "date", "IdNr": 6400, "conf": "medium", "note": "faint ink; band verified 2026-11-06; numerals index-anchored"}
{"file": "s04_22.jpg", "bbox": [150, 1948, 280, 158], "text": "Jette Michel", "project": "eschau-seeds", "kind": "name", "IdNr": 35241}
{"file": "s04_22.jpg", "bbox": [890, 1974, 360, 150], "text": "04.05.1849", "project": "eschau-seeds", "kind": "date", "IdNr": 35241, "conf": "medium", "note": "very faint ink; band verified 2026-11-06; numerals index-anchored"}
{"file": "s02_22.jpg", "bbox": [180, 1108, 220, 259], "text": "Jakob Michel", "project": "eschau-seeds", "kind": "name", "IdNr": 38015}
{"file": "s04_22.jpg", "bbox": [434, 296, 166, 165], "text": "Lehrer, Vorsänger", "project": "eschau-occ", "kind": "occupation", "IdNr": 38642}
{"file": "s04_22.jpg", "bbox": [434, 1457, 166, 208], "text": "Handelsmann", "project": "eschau-occ", "kind": "occupation", "IdNr": 38647}
{"file": "s04_22.jpg", "bbox": [434, 1787, 166, 189], "text": "Händler", "project": "eschau-occ", "kind": "occupation", "IdNr": 6400}
{"file": "s05_22.jpg", "bbox": [474, 173, 152, 201], "text": "Handelsmann", "project": "eschau-occ", "kind": "occupation", "IdNr": 35242}
{"file": "s05_22.jpg", "bbox": [458, 1036, 166, 138], "text": "Krämer, Ökonom", "project": "eschau-occ", "kind": "occupation", "IdNr": 38041}
{"file": "s05_22.jpg", "bbox": [458, 1254, 166, 178], "text": "Viehhändler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38655}
{"file": "s05_22.jpg", "bbox": [458, 1630, 166, 106], "text": "Händler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38665}
{"file": "s06_22.jpg", "bbox": [450, 1671, 166, 266], "text": "Schreibmaterialienhändler", "project": "eschau-occ", "kind": "occupation", "IdNr": 38673}
{"file": "g10_22.jpg", "bbox": [1745, 448, 330, 128], "text": "d. 17ten siebzehnten Jan. 1858", "IdNr": 38752, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g10_22.jpg", "bbox": [1745, 872, 330, 80], "text": "den Mittwoch 15ten fünfzehnten", "IdNr": 3664, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1880, 545, 270, 105], "text": "fünfundzwanzigsten Jan. 1849", "IdNr": 38779, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1880, 1465, 280, 130], "text": "Freitag den 28. achtundzwanzigsten März 1851", "IdNr": 38653, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
{"file": "g07_22.jpg", "bbox": [1900, 1770, 280, 115], "text": "Montag den 5. fünften Jan. 1852", "IdNr": 38765, "project": "eschau1081-batch2", "kind": "date", "conf": "medium"}
'''
_cells = [json.loads(l) for l in _ARMB_BLOB.strip().splitlines()]
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
    assert len(_cells) == int(os.environ.get("KURRENT_EXPECT_CELLS", "306")), f"expected 306 merged arm-B cells, got {len(_cells)}"
_train_path = CODE / "train_manifest_v4.jsonl"
print("train manifest:", _train_path.name, flush=True)
samples = [json.loads(l) for l in _train_path.open()]
_mm = [s2 for s2 in samples if s2.get("project") not in ("eschau-seeds", "eschau-occ", "eschau-gbirths")]
REPEAT = 4  # v19: v15/v18 weight (v16 x6 regressed names 0.8894->0.9276)  # v12 (issue #32): was 40 in v8-v10; x40 drove memorization (seeds_fit 0.0, held-out names worse than base)
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
