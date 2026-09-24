#!/bin/bash
# Run Cargo inside the pinned IronClaw 1.4.0 Rust image.
# Build output and the Cargo registry stay in Docker volumes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="rust:1.96-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc"
exec docker run --rm \
  -v "$ROOT":/src \
  -v ironclaw-cargo-registry:/usr/local/cargo/registry \
  -v ironclaw-cargo-git:/usr/local/cargo/git \
  -v ironclaw-target:/cargo-target \
  -w /src \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e CARGO_TERM_COLOR=always \
  "$IMAGE" \
  cargo "$@"
