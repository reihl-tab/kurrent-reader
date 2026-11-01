"""kurrent-reader: free local reading of 19th-century German Kurrent scans.

Pipeline: preprocess (denoise / contrast / deskew / binarise) -> segment
(kraken) -> read (TrOCR, dh-unibe/trocr-kurrent). Originals are never
modified; every cleaned artefact is written next to the original and both
variants are read so the two outputs can be compared line by line.
"""

__version__ = "0.4.1"
