"""End-to-end page runs: clean, segment, read, compare original vs cleaned."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import preprocess, segment
from .read import LineReader, ReadResult


def run_page(
    page_path: str | Path,
    output_dir: str | Path,
    *,
    reader: LineReader | None = None,
    read: bool = True,
    seg_mode: str = "boxes",
    seg_model: str | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline on one page.

    Writes under ``output_dir``:
      ``cleaned/``           the cleaned + binarised variants of the page
      ``lines-original/``    line crops from the untouched page
      ``lines-cleaned/``     line crops from the cleaned page
      ``transcript-original.json`` / ``transcript-cleaned.json``
      ``comparison.json``    both transcripts aligned by line index

    The original file is never modified.
    """
    page = Path(page_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clean_meta = preprocess.process_file(page, out / "cleaned")
    variants: dict[str, str] = {
        "original": str(page),
        "cleaned": str(clean_meta["cleaned"]),
    }
    report: dict[str, Any] = {"page": str(page), "clean_meta": clean_meta, "variants": {}}
    manifests: dict[str, list[dict[str, object]]] = {}
    for name, path in variants.items():
        boxes = segment.segment_page(path, mode=seg_mode, model=seg_model)
        manifest = segment.crop_lines(path, boxes, out / f"lines-{name}")
        manifests[name] = manifest
        report["variants"][name] = {"line_count": len(manifest)}

    if read:
        active = reader if reader is not None else LineReader(quantize="int8")
        transcripts: dict[str, list[ReadResult]] = {}
        for name, manifest in manifests.items():
            paths = [str(out / f"lines-{name}" / str(m["file"])) for m in manifest]
            results = active.read_many(paths)
            transcripts[name] = results
            (out / f"transcript-{name}.json").write_text(
                json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False)
            )
        comparison = _compare(transcripts["original"], transcripts["cleaned"])
        (out / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False)
        )
        report["comparison"] = comparison
    return report


def _compare(
    original: list[ReadResult], cleaned: list[ReadResult]
) -> list[dict[str, object]]:
    """Align both transcripts by index; counts differ when segmentation does."""
    rows: list[dict[str, object]] = []
    for i in range(max(len(original), len(cleaned))):
        o = original[i] if i < len(original) else None
        c = cleaned[i] if i < len(cleaned) else None
        rows.append(
            {
                "index": i,
                "original": None if o is None else {"text": o.text, "confidence": o.confidence},
                "cleaned": None if c is None else {"text": c.text, "confidence": c.confidence},
                "agree": bool(o and c and o.text == c.text),
            }
        )
    return rows
