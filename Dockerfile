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

# pip 24.0 bundled in the base image mis-resolves safetensors (InconsistentCandidate
# on ~=0.7 vs the 0.8.0 candidate) during `pip install .`; upgrade pip first.
# Found by the docker-smoke CI job (issue #34).
RUN pip install --upgrade pip

# CPU-only PyTorch keeps the image small; it is not on PyPI's default index.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9,<=2.14"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY tests ./tests
RUN pip install ".[dev]"

# The TrOCR model (~1.3 GB) is downloaded on first use, or bake it in with:
#   RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('dh-unibe/trocr-kurrent')"

ENTRYPOINT ["kurrent-reader"]
CMD ["--help"]
