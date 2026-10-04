# Toolchain for JavaScript and TypeScript tickets: tsc --noEmit, lint, unit tests (bun or node).
FROM node:22-bookworm-slim
ARG BUN_VERSION=1.3.14
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl unzip make \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g typescript eslint prettier pnpm \
    && curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash -s "bun-v${BUN_VERSION}"
USER node
WORKDIR /work
