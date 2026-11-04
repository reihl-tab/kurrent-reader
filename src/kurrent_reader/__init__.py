"""kurrent-reader: free local reading of 19th-century German Kurrent scans.

Pipeline: preprocess (denoise / contrast / deskew / binarise) -> segment
(kraken) -> read (TrOCR, dh-unibe/trocr-kurrent). Originals are never
modified; every cleaned artefact is written next to the original and both
variants are read so the two outputs can be compared line by line.
"""

import sys as _sys

__version__ = "0.6.0"

MIN_PYTHON = (3, 11)


def _check_python_version() -> None:
    if _sys.version_info < MIN_PYTHON:
        raise SystemExit(
            "kurrent-reader requires Python "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
            "(you are running "
            + ".".join(str(v) for v in _sys.version_info[:3])
            + "). "
            "Python 3.10 is end-of-life; please use a newer interpreter."
        )


_check_python_version()
