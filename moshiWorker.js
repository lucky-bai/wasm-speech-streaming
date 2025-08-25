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
  static instance = {};
  // Retrieve the model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(params) {
    const { weightsURL, modelID, tokenizerURL, mimiURL, configURL } = params;
    // load individual modelID only once
    if (!this.instance[modelID]) {
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

      this.instance[modelID] = new MoshiASRDecoder(
        weightsArrayU8,
        tokenizerArrayU8,
        mimiArrayU8,
        configArrayU8
      );
    } else {
      self.postMessage({ status: "loading", message: "Model Already Loaded" });
    }

    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { weightsURL, modelID, tokenizerURL, configURL, mimiURL, audioURL } =
    event.data;

  try {
    self.postMessage({ status: "decoding", message: "Starting Decoder" });
    const decoder = await MoshiASR.getInstance({
      weightsURL,
      modelID,
      tokenizerURL,
      mimiURL,
      configURL,
    });

    self.postMessage({ status: "decoding", message: "Loading Audio" });
    const audioArrayU8 = await fetchArrayBuffer(audioURL);

    self.postMessage({ status: "decoding", message: "Running Decoder..." });

    decoder.decode_streaming(audioArrayU8, (word) => {
      // Send each word as it becomes available
      self.postMessage({
        status: "streaming",
        word: word,
      });
    });

    self.postMessage({
      status: "complete",
      message: "complete",
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
