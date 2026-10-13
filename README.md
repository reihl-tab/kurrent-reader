# kurrent-reader

Free, fully local reader for 19th-century German Kurrent handwriting in
archival register scans. Built to serve as one more independent "blind
reader" for contested dates in the Eschau/Kochendorf Jewish registers.

- **Reading**: [dh-unibe/trocr-kurrent](https://huggingface.co/dh-unibe/trocr-kurrent)
  (MIT), a TrOCR base fine-tune from Digital Humanities @ University of Bern,
  test CER 2.66% on 19th-century Kurrent. The model is **line-level**: one
  cropped text line in, one string out.
- **Segmentation**: [kraken](https://github.com/mittagessen/kraken), classic
  box segmenter by default (no model download); a neural baseline model can
  be selected for harder layouts.
- **Preprocessing**: OpenCV/scikit-image cleanup of scan noise (CLAHE
  contrast, median + non-local-means denoise, min-area-rect deskew, Sauvola
  binarisation). The original scan is never modified; cleaned variants are
  written next to it and **both variants are segmented and read** so the two
  transcripts can be compared line by line (`comparison.json`).

Everything runs offline on CPU. No accounts, no paid services.

## Status and honest limits

First real runs (2026-10-13, 2 GB RAM / 2 vCPU sandbox, CPU fp16):

- **Works well: full narrative date lines.** On the Eschau 1821 birth
  register the reader produced, unprompted:
  `2.) Den 23ten Nov ist dem hiesigen Schulzuden` (twins entry, conf 0.95),
  `4.) Den 28. August Abend 10 Uhr ist dem Grie-` (conf 0.87),
  `von seiner Ehefrau Kula...` (the twins' parents' line).
- **Fails: isolated table-cell crops, even after table-rule removal.**
  Single date cells from the 1845 death-register table ("28ten May",
  "30ten May") came back as garbage (`zuſchlag`, `Bſtron`, `922`). The
  model needs a full line of context; cell fragments are out of
  distribution. Erasing the printed table rules first (`kurrent-reader
  derule`: morphological rule detection + inpainting) improved honesty and
  partial content - the misleading `922` at 0.84 became `geometrisch` at
  0.13, and the year strip reads `1845` at 0.996 - but the day/month
  words are still not recovered.
- **Per-word strips: partial.** De-ruled cells split into word crops and
  padded to line-like strips read `1845` at 0.996 and flag a rule stub as
  a mark (`+` at 0.92), but merged cursive spans stay garbage at near-zero
  confidence (`Das Meeres` at 0.004). The Gerson death/burial day and
  month remain unreadable here; the quorum owns those fields.
- **Segmentation fragments tables.** Classic boxes on the death page:
  187 boxes original vs 120 cleaned; birth page 118 vs 77. Cleaned
  segmentation is less fragmented, but box indexes do not align between
  variants, so the per-line original-vs-cleaned comparison is only
  meaningful when both variants segment alike (a known caveat).
- **Noisy photocopies can break the classic segmenter** (kraken
  connected-component limit); `segment.py` retries with a smoothed,
  downscaled image. On the Kochendorf family-book photocopy that retry
  completed but found 0 lines; the recipe that works is the full
  preprocess filter first, then segmentation of the cleaned image at
  0.5x scale (82 lines).
- **Memory.** fp32 weights are ~1.3 GB and `torch` cannot quantize fp16
  tensors, so machines with < ~2.6 GB free fall back to fp16 inference
  (~670 MB). Throughput on 2 vCPU: ~30-120 s per line. A confidence of
  -1.0 means the score was unavailable, not a zero-confidence reading.
- This tool is an **extra reader only**. It does not replace the blind
  multi-model quorum and must not be used alone to change any Geni field.
- The model card warns the model "has not been extensively tested"; treat
  every output as a reading proposal, not ground truth.

## Usage

Docker (recommended):

```bash
docker build -t kurrent-reader .
docker run --rm -v "$PWD/data:/data" kurrent-reader run /data/page.jpg /data/out
```

Local:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install .
kurrent-reader --help
```

Commands:

```bash
kurrent-reader preprocess page.jpg out/            # writes page.clean.png + page.bin.png
kurrent-reader segment page.jpg out/lines          # line crops + lines.json
kurrent-reader read out/lines/line-*.png           # transcripts with confidence
kurrent-reader run page.jpg out/                   # all of the above, original vs cleaned
kurrent-reader run page.jpg out/ --no-read         # skip the model (preprocessing/segmentation only)
```

The model downloads from Hugging Face on first use (~1.3 GB).

## Development

```bash
pip install ".[dev]"
python -m pytest tests/        # synthetic data only; no model download
python -m mypy src/
```

## Layout

```
src/kurrent_reader/
  preprocess.py   scan cleanup (denoise/contrast/deskew/binarise)
  segment.py      kraken line segmentation + crop writing
  read.py         TrOCR line reader (CPU, optional int8 quantization)
  pipeline.py     original-vs-cleaned end-to-end run + comparison
  cli.py          kurrent-reader command
Dockerfile        python:3.11-slim, CPU torch, package + tests
tests/            pytest suite on synthetic images (fast, offline)
```

## Data

Test pages are public archival scans from the JUF/Stadtarchiv-Digital
collection (source 1081, Eschau Jewish registers) and a family-book scan of
the Kochendorf register. They are not committed to this repository; place
them under `data/` (gitignored).

## Intended safeguards

Preserve original scans. Keep cleaned derivatives and proposed readings
separate, retain uncertain alternatives and require human source review before
promoting a reading to a genealogical fact. Do not infer missing people or facts
from OCR output. Keep private scans and identifying material in their approved
private audience, not public examples or logs.

Changes should start with an issue and a small PR, with tests where code
exists and honest limits where it does not. Independent review is required
before merge; a successful test or merge is not approval to train, publish data.
