# WASM Streaming Speech Recognition

Real-time streaming speech-to-text transcription running entirely in your browser using Rust and WebAssembly (WASM). This demo processes audio **entirely offline on your CPU** after downloading a ~950MB speech recognition model.

## Demo

https://github.com/user-attachments/assets/40246f86-9f80-4570-aa2d-bce2d04ed8e4

Try it live: **[https://efficient-nlp-wasm-streaming-speech.static.hf.space/index.html](https://efficient-nlp-wasm-streaming-speech.static.hf.space/index.html)**

## Technologies

- [Kyutai STT Model](https://huggingface.co/kyutai/stt-1b-en_fr) - 1B param streaming speech recognition model for English and French. This demo uses a [4-bit quantized](https://huggingface.co/efficient-nlp/stt-1b-en_fr-quantized) version of the model.
- [Candle](https://github.com/huggingface/candle/) - Hugging Face's ML framework for Rust
- [Rayon](https://github.com/rayon-rs/rayon) - CPU parallelization for Rust
- [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon) - WASM bindings for Rayon

## Related Projects

This is a **research/tech demo**. For more accurate cloud transcription and real-time LLM grammar correction, check out [Voice Writer](https://voicewriter.io).

## Performance

Performance varies by device.

- On Apple Silicon or other recent CPUs, it typically runs in real time.
- On older devices, it may not keep up (real-time factor < 1).
- Mobile devices are not supported.

## Prerequisites

- **Rust, Cargo, wasm32-unknown-unknown**

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup target add wasm32-unknown-unknown
  ```

- **wasm-bindgen-cli**

  ```bash
  cargo install wasm-bindgen-cli
  ```

- **wasm-opt (Binaryen)** – optional but recommended

  - macOS: `brew install binaryen`
  - Ubuntu/Debian: `sudo apt install binaryen`

- **Python 3**
- **curl**

## Running Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/lucky-bai/wasm-speech-streaming
   cd wasm-speech-streaming
   ```

2. **Build the Rust/WASM library:**

   ```bash
   ./build-lib.sh
   ```

3. **Open your browser and go to:**

   ```bash
   http://localhost:8000
   ```

## License

MIT License
