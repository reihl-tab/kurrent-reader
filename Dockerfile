# syntax=docker/dockerfile:1
# python:3.11-slim, manifest-list digest captured 2026-11-01 (multi-arch).
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is required by the CPU build of PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Every install reads a hash-locked file under requirements/ (see
# docs/DEPENDENCY-LOCKS.md). The base image's pip 24.0 is used as is: with
# exact pins it no longer hits the safetensors InconsistentCandidate error
# (issue #34) that the unpinned `pip install .` met, so no pip upgrade step.
WORKDIR /app
COPY requirements ./requirements

# Build prerequisites for the --no-build-isolation install below.
RUN pip install --require-hashes -r requirements/build.txt

# Runtime dependencies. CPU-only PyTorch keeps the image small; torch and
# torchvision are locked from the PyTorch CPU index, everything else from PyPI.
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/core.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation ./

COPY tests ./tests
RUN pip install --require-hashes --no-deps -r requirements/dev.txt

# The TrOCR model (~1.3 GB) is downloaded on first use, or bake it in with:
#   RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('dh-unibe/trocr-kurrent')"

ENTRYPOINT ["kurrent-reader"]
CMD ["--help"]
