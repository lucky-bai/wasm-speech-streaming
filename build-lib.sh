set -e
rm -rf build
cargo build --target wasm32-unknown-unknown --release --lib
wasm-bindgen ../../target/wasm32-unknown-unknown/release/candle_wasm_moshi_asr.wasm --out-dir build --target web
python -m http.server
