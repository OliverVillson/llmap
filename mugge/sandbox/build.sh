#!/usr/bin/env bash
# Builds the per-language sandbox images on the project VM: mugge-c, mugge-node, mugge-python, mugge-web.
# Built once and kept on the VM's persistent disk; the architect picks which a project needs.
#   sandbox/build.sh [image ...]      (default: all)    ENGINE=docker to use docker instead of podman
set -euo pipefail
cd "$(dirname "$0")/images"
ENGINE="${ENGINE:-podman}"
PREFIX="${PREFIX:-mugge}"
images=("$@")
[ ${#images[@]} -eq 0 ] && images=(c node python web)
for img in "${images[@]}"; do
  # web builds on top of node
  if [ "$img" = web ] && ! "$ENGINE" image exists "$PREFIX-node" 2>/dev/null && ! "$ENGINE" image inspect "$PREFIX-node" >/dev/null 2>&1; then
    "$ENGINE" build -t "$PREFIX-node" node
  fi
  echo "building $PREFIX-$img"
  "$ENGINE" build --build-arg PREFIX="$PREFIX" -t "$PREFIX-$img" "$img"
done
