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

- First real runs were on a 2 GB / 2 vCPU sandbox. The fp32 model is
  ~1.3 GB, so the default is `--quantize int8` (dynamic quantization,
  ~350 MB weights, well supported for CPU Linear layers).
- The classic box segmenter handles clean book pages but can split or merge
  lines on table-heavy register pages; cropped regions (a single register
  row or entry) read far more reliably than whole table pages. This matches
  the model card: it was trained on text lines, not table cells.
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
