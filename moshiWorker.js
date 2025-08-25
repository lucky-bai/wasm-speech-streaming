import init, {
  MoshiASRDecoder,
  initThreadPool,
} from "./build/wasm_speech_streaming.js";

async function fetchArrayBuffer(url) {
  const cacheName = "whisper-candle-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}

class MoshiASR {
  static decoder = null;

  // Initialize the model
  static async initialize(params) {
    const { weightsURL, tokenizerURL, mimiURL, configURL } = params;

    if (this.decoder) {
      self.postMessage({ status: "model_ready" });
      return;
    }

    try {
      await init();
      const numThreads = navigator.hardwareConcurrency || 4;
      await initThreadPool(numThreads);

      self.postMessage({
        status: "loading",
        message: `Loading Model with ${numThreads} threads`,
      });

      const [weightsArrayU8, tokenizerArrayU8, mimiArrayU8, configArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(mimiURL),
          fetchArrayBuffer(configURL),
        ]);

      this.decoder = new MoshiASRDecoder(
        weightsArrayU8,
        tokenizerArrayU8,
        mimiArrayU8,
        configArrayU8
      );

      self.postMessage({ status: "model_ready" });
    } catch (error) {
      self.postMessage({ error: error.message });
    }
  }

  static startStream() {
    if (this.decoder) {
      this.decoder.start_streaming();
    }
  }

  static stopStream() {
    if (this.decoder) {
      this.decoder.stop_streaming();
    }
  }

  static processAudio(audioData) {
    if (this.decoder) {
      this.decoder.process_audio_chunk(audioData, (word) => {
        self.postMessage({
          status: "streaming",
          word: word,
        });
      });
    }
  }
}

self.addEventListener("message", async (event) => {
  const { command } = event.data;

  try {
    switch (command) {
      case "initialize":
        const { weightsURL, modelID, tokenizerURL, mimiURL, configURL } =
          event.data;
        await MoshiASR.initialize({
          weightsURL,
          modelID,
          tokenizerURL,
          mimiURL,
          configURL,
        });
        break;

      case "start_stream":
        MoshiASR.startStream();
        break;

      case "stop_stream":
        MoshiASR.stopStream();
        break;

      case "process_audio":
        const { audioData } = event.data;
        MoshiASR.processAudio(audioData);
        break;

      default:
        self.postMessage({ error: "Unknown command: " + command });
    }
  } catch (e) {
    self.postMessage({ error: e.message });
  }
});
