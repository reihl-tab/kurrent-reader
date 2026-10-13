"""Command line interface: kurrent-reader <preprocess|segment|read|run>."""

from __future__ import annotations

import argparse
import json
import sys

from . import preprocess, segment
from .pipeline import run_page
from .read import DEFAULT_MODEL, LineReader


def _cmd_preprocess(args: argparse.Namespace) -> int:
    meta = preprocess.process_file(args.image, args.output_dir)
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    boxes = segment.segment_page(args.image, mode=args.mode, model=args.model)
    manifest = segment.crop_lines(args.image, boxes, args.output_dir)
    print(json.dumps({"lines": len(manifest), "manifest": manifest}, indent=2))
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    reader = LineReader(model_id=args.model_id, quantize=args.quantize)
    results = reader.read_many(sorted(args.lines))
    print(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    reader = LineReader(model_id=args.model_id, quantize=args.quantize) if args.read else None
    report = run_page(
        args.page,
        args.output_dir,
        reader=reader,
        read=args.read,
        seg_mode=args.mode,
        seg_model=args.seg_model,
    )
    summary = {
        "page": report["page"],
        "deskew_angle_deg": report["clean_meta"]["deskew_angle_deg"],
        "variants": report["variants"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kurrent-reader",
        description="Free local reader for 19th-century German Kurrent scans.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="clean one scan (original kept untouched)")
    p.add_argument("image")
    p.add_argument("output_dir")
    p.set_defaults(func=_cmd_preprocess)

    p = sub.add_parser("segment", help="detect text lines and write crops")
    p.add_argument("image")
    p.add_argument("output_dir")
    p.add_argument("--mode", choices=["boxes", "baseline"], default="boxes")
    p.add_argument("--model", default=None, help="kraken segmentation model (baseline mode)")
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("read", help="read line images with TrOCR Kurrent")
    p.add_argument("lines", nargs="+")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--quantize", choices=["int8"], default="int8")
    p.set_defaults(func=_cmd_read)

    p = sub.add_parser("run", help="full pipeline: clean, segment, read, compare")
    p.add_argument("page")
    p.add_argument("output_dir")
    p.add_argument("--mode", choices=["boxes", "baseline"], default="boxes")
    p.add_argument("--seg-model", default=None)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--quantize", choices=["int8"], default="int8")
    p.add_argument("--no-read", dest="read", action="store_false")
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
