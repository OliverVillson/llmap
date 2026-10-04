# Toolchain for C tickets: -Wall -Werror builds, unit tests, ASan/UBSan and valgrind runs.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang cmake gdb valgrind pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 coder
USER coder
WORKDIR /work
