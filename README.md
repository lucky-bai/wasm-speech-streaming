# WASM Streaming Speech Recognition

Real-time streaming speech-to-text transcription running entirely in your browser using Rust and WebAssembly (WASM). This demo processes audio **entirely offline on your CPU** after downloading a ~950MB speech recognition model.

## Demo

<video controls style="max-width: 640px; border-radius: 8px; margin-bottom: 1em;">
  <source src="https://private-user-images.githubusercontent.com/123435/481822889-ff32cbfb-bec6-45c4-8bca-cf483a5a1056.mov?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTYxNTk3MDgsIm5iZiI6MTc1NjE1OTQwOCwicGF0aCI6Ii8xMjM0MzUvNDgxODIyODg5LWZmMzJjYmZiLWJlYzYtNDVjNC04YmNhLWNmNDgzYTVhMTA1Ni5tb3Y_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwODI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDgyNVQyMjAzMjhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZWI2MjkxZTZmOGY1N2U4OGU2NTVlN2M2OGVjZWU1ZjFkNTA4MmM3MGY4MmE4NmZhYjYxY2EyMDI1Zjk5YTliJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.9lGXYfNfvMHDRxs98cALUw1D9ReKBlj6vFENhJ6UTdQ" type="video/mp4">
  Your browser does not support the video tag.
</video>

Try it live: **[https://efficient-nlp-wasm-streaming-speech.static.hf.space/index.html](https://efficient-nlp-wasm-streaming-speech.static.hf.space/index.html)**

## Technologies

- [Kyutai STT Model](https://huggingface.co/kyutai/stt-1b-en_fr) - 1B param streaming speech recognition model for English and French. This demo uses a 4-bit quantized version of the model.
- [Candle](https://github.com/huggingface/candle/) - Hugging Face's ML framework for Rust
- [Rayon](https://github.com/rayon-rs/rayon) - CPU parallelization for Rust
- [wasm-bindgen-rayon](https://github.com/rustwasm/wasm-bindgen-rayon) - WASM bindings for Rayon

## Performance

Performance varies by device.

- On Apple Silicon or other recent CPUs, it typically runs in real time.
- On older devices, it may not keep up (real-time factor < 1).
- Mobile devices are not supported.

## Running Locally

1. **Clone the repository:**

   ```
   git clone https://lucky-bai/wasm-speech-streaming
   cd wasm-speech-streaming
   ```

2. **Build the Rust/WASM library:**

   ```
   ./build-lib.sh
   ```

3. **Open your browser and go to:**
   ```
   http://localhost:8000
   ```

## Limitations

This is a **research/tech demo**. For more accurate cloud transcription and real-time LLM grammar correction, check out [Voice Writer](https://voicewriter.io).
