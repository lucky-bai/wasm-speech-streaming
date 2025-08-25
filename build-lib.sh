#!/usr/bin/env bash
set -e

# Parse flags: ./build-lib.sh [--no-opt]
NO_OPT=false
while [[ "$1" =~ ^- ]]; do
  case "$1" in
    --no-opt) NO_OPT=true ;;
    *) echo "Unknown option: $1" && exit 1 ;;
  esac
  shift
done

# Download Tailwind from CDN if not present
if [ ! -f css/tailwind-3.4.17.js ]; then
    echo "Downloading tailwind-3.4.17.js..."
    mkdir -p css
    curl -L -o css/tailwind-3.4.17.js https://cdn.tailwindcss.com/3.4.17
fi

# Build wasm
rm -rf build
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/wasm_speech_streaming.wasm \
  --out-dir build --target web

# Optionally run wasm-opt
if ! $NO_OPT && command -v wasm-opt >/dev/null 2>&1; then
    echo "Optimizing wasm with wasm-opt..."
    wasm-opt -O3 --enable-simd --enable-threads \
      -o build/wasm_speech_streaming_bg.opt.wasm \
      build/wasm_speech_streaming_bg.wasm
    mv build/wasm_speech_streaming_bg.opt.wasm build/wasm_speech_streaming_bg.wasm
else
    echo "Skipping wasm-opt"
fi

# Run server
python server.py
