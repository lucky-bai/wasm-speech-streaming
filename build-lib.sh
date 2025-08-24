set -e
rm -rf build
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/wasm_speech_streaming.wasm --out-dir build --target web --split-linked-modules
python -m http.server
