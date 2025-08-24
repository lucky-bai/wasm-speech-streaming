set -e

if [ ! -f css/tailwind-3.4.17.js ]; then
    echo "Downloading tailwind-3.4.17.js..."
    mkdir -p css
    curl -L -o css/tailwind-3.4.17.js https://cdn.tailwindcss.com/3.4.17
fi

rm -rf build
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/wasm_speech_streaming.wasm --out-dir build --target web
python server.py
