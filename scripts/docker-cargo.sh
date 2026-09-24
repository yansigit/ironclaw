#!/bin/bash
# Run Cargo inside the pinned IronClaw 1.4.0 Rust image.
# Build output and the Cargo registry stay in Docker volumes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="rust:1.96-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc"
TARGET_VOLUME="${IRONCLAW_TARGET_VOLUME:-ironclaw-target}"
ENV_ARGS=()
if [ -n "${CURSOR_ACCESS_TOKEN:-}" ]; then
  ENV_ARGS+=(-e CURSOR_ACCESS_TOKEN)
fi
exec docker run --rm \
  -v "$ROOT":/src \
  -v ironclaw-cargo-registry:/usr/local/cargo/registry \
  -v ironclaw-cargo-git:/usr/local/cargo/git \
  -v "$TARGET_VOLUME":/cargo-target \
  -w /src \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e CARGO_TERM_COLOR=always \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  cargo "$@"
