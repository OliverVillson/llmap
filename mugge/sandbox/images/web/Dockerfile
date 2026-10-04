# Toolchain for frontend tickets: the node image plus Playwright with Chromium and a static server,
# for DOM and interaction tests and a screenshot the TUI can show.
ARG PREFIX=mugge
FROM ${PREFIX}-node
USER root
RUN npx -y playwright@1.56.0 install --with-deps chromium && npm install -g serve @playwright/test@1.56.0
USER node
WORKDIR /work
