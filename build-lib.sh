set -e

if [ ! -f css/tailwind-3.4.17.js ]; then
    echo "Downloading tailwind-3.4.17.js..."
    mkdir -p css
    curl -L -o css/tailwind-3.4.17.js https://cdn.tailwindcss.com/3.4.17
fi

rm -rf build
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/wasm_speech_streaming.wasm --out-dir build --target web
if command -v wasm-opt >/dev/null 2>&1; then
    echo "Optimizing wasm with wasm-opt..."
    wasm-opt -O3 --enable-simd --enable-threads -o build/wasm_speech_streaming_bg.opt.wasm build/wasm_speech_streaming_bg.wasm
    mv build/wasm_speech_streaming_bg.opt.wasm build/wasm_speech_streaming_bg.wasm
else
    echo "wasm-opt not found, skipping wasm optimization."
fi
python server.py
