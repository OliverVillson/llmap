# Toolchain for Python tickets: pytest/unittest, ruff, mypy, numpy and CPU torch (ML engineering tickets). gcc stays so ctypes/cffi code can build its C side.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv pytest ruff mypy numpy \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN useradd -m -u 1000 coder
USER coder
WORKDIR /work
