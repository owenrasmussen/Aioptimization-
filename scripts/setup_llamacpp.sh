#!/usr/bin/env bash
# Week 1: clone llama.cpp, pin one commit, build with CUDA.
# Usage: scripts/setup_llamacpp.sh [commit]   (default: current master, then pinned)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/llama.cpp"
PIN_FILE="$ROOT/LLAMACPP_COMMIT"

if [ ! -d "$DEST" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DEST"
fi

COMMIT="${1:-$(cat "$PIN_FILE" 2>/dev/null || true)}"
git -C "$DEST" fetch origin
if [ -n "$COMMIT" ]; then
  git -C "$DEST" checkout --detach "$COMMIT"
else
  git -C "$DEST" checkout --detach origin/master
fi
git -C "$DEST" rev-parse HEAD > "$PIN_FILE"
echo "Pinned llama.cpp at $(cat "$PIN_FILE") (recorded in LLAMACPP_COMMIT; commit that file)"

# 86 = Ampere consumer cards (3080 Ti)
cmake -S "$DEST" -B "$DEST/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build" --config Release -j \
  --target llama-bench llama-perplexity llama-quantize llama-server llama-imatrix

echo "Binaries in $DEST/build/bin"
