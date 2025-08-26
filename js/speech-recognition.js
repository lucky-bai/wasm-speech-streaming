const WEIGHTS_URL =
  "https://huggingface.co/efficient-nlp/stt-1b-en_fr-quantized/resolve/main/model-q4k.gguf";
const MIMI_URL =
  "https://huggingface.co/efficient-nlp/stt-1b-en_fr-quantized/resolve/main/mimi-pytorch-e351c8d8@125.safetensors";
const TOKENIZER_URL =
  "https://huggingface.co/efficient-nlp/stt-1b-en_fr-quantized/resolve/main/tokenizer_en_fr_audio_8000.json";
const CONFIG_URL =
  "https://huggingface.co/efficient-nlp/stt-1b-en_fr-quantized/resolve/main/config.json";

const moshiWorker = new Worker("./js/moshi-worker.js", { type: "module" });
let isRecording = false;
let audioStream = null;
let audioContext = null;
let processor = null;
let source = null;
let modelInitialized = false;
let pendingStart = false;

// Performance tracking
let audioChunksProcessed = 0;
let sessionStartTime = 0;

function updateStatusDiv(message) {
  document.querySelector("#status-div").textContent = message;
}

function updateDiagnostics() {
  const diagnostics = document.querySelector("#diagnostics");
  if (!diagnostics) return;

  const cpuCount = navigator.hardwareConcurrency || "unknown";

  // Only update metrics when recording, otherwise show final values
  if (isRecording && sessionStartTime) {
    // Calculate real-time factor (audio processed / wall clock time)
    // >1 = faster than real-time, <1 = slower than real-time
    const audioProcessed = audioChunksProcessed * (1024 / 24000);
    const audioSessionDuration = (Date.now() - sessionStartTime) / 1000;
    const realTimeFactor =
      audioSessionDuration > 0 ? audioProcessed / audioSessionDuration : 0;

    // Color code based on performance
    let factorColor = "";
    if (realTimeFactor >= 0.95) {
      factorColor = "text-green-600";
    } else if (realTimeFactor >= 0.8) {
      factorColor = "text-yellow-600";
    } else {
      factorColor = "text-red-600";
    }

    diagnostics.innerHTML = `CPUs: ${cpuCount}, Real-time factor: <span class="${factorColor}">${realTimeFactor.toFixed(
      2
    )}x</span>, Duration: ${audioSessionDuration.toFixed(1)}s`;
  } else if (!sessionStartTime) {
    diagnostics.innerHTML = `CPUs: ${cpuCount}, Real-time factor: <span class="text-gray-600">0.00x</span>, Duration: 0.0s`;
  }
}

window.addEventListener("load", updateDiagnostics);
setInterval(updateDiagnostics, 200);

function initializeModel() {
  if (modelInitialized) return;

  const button = document.querySelector("#speech-button");
  button.disabled = true;
  button.className =
    "bg-gray-400 text-gray-700 font-normal py-2 px-4 rounded cursor-not-allowed";

  moshiWorker.postMessage({
    command: "initialize",
    weightsURL: WEIGHTS_URL,
    mimiURL: MIMI_URL,
    tokenizerURL: TOKENIZER_URL,
    configURL: CONFIG_URL,
  });
}

// Handle messages from worker
moshiWorker.addEventListener("message", async (event) => {
  const data = event.data;
  if (data.status === "model_ready") {
    modelInitialized = true;
    updateStatusDiv("Model loaded - Ready to start");

    const button = document.querySelector("#speech-button");
    button.disabled = false;
    button.className =
      "bg-gray-700 hover:bg-gray-800 text-white font-normal py-2 px-4 rounded";

    if (pendingStart) {
      pendingStart = false;
      await startRecording();
    }
  } else if (data.status === "streaming") {
    // Add new word to transcription in real-time
    const outputDiv = document.querySelector("#output-generation");
    const placeholder = document.querySelector("#output-placeholder");

    if (placeholder) placeholder.hidden = true;

    if (outputDiv.textContent) {
      outputDiv.textContent += " " + data.word;
    } else {
      outputDiv.textContent = data.word;
    }
    outputDiv.hidden = false;
  } else if (data.status === "chunk_processed") {
    audioChunksProcessed++;
  } else if (data.status === "loading") {
    updateStatusDiv(data.message);
  } else if (data.error) {
    updateStatusDiv("Error: " + data.error);
    pendingStart = false;
  }
});

function updateStatus(data) {
  const { status, message, word } = data;
  const outputDiv = document.querySelector("#output-generation");

  if (status === "loading" || status === "decoding") {
    updateStatusDiv(
      message || (status === "loading" ? "Loading..." : "Decoding...")
    );
  } else if (status === "streaming") {
    // Add new word to the transcription in real-time
    if (outputDiv.textContent) {
      outputDiv.textContent += " " + word;
    } else {
      outputDiv.textContent = word;
    }
    outputDiv.hidden = false;
  } else if (status === "complete") {
    updateStatusDiv("Ready");
  }
}

async function startMicrophone() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    updateStatusDiv("Microphone access granted");

    audioContext = new AudioContext({ sampleRate: 24000 });
    source = audioContext.createMediaStreamSource(audioStream);

    processor = audioContext.createScriptProcessor(1024, 1, 1);

    processor.onaudioprocess = function (event) {
      if (!isRecording || !modelInitialized) return;

      const inputBuffer = event.inputBuffer;
      const inputData = inputBuffer.getChannelData(0);

      // Send audio chunk to worker
      const audioChunk = new Float32Array(inputData);
      moshiWorker.postMessage(
        {
          command: "process_audio",
          audioData: audioChunk,
        },
        [audioChunk.buffer]
      );
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
  } catch (error) {
    updateStatusDiv("Microphone access denied: " + error.message);
    throw error;
  }
}

function stopMicrophone() {
  // Disconnect audio nodes
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  // Stop media stream
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }

  updateStatusDiv("Microphone stopped");
}

async function startRecording() {
  const button = document.querySelector("#speech-button");

  try {
    updateStatusDiv("Requesting microphone access...");
    await startMicrophone();

    // Reset performance counters
    audioChunksProcessed = 0;
    sessionStartTime = Date.now();

    // Start streaming session
    moshiWorker.postMessage({ command: "start_stream" });

    isRecording = true;
    button.textContent = "Stop Speech";
    button.className =
      "bg-red-600 hover:bg-red-700 text-white font-normal py-2 px-4 rounded";
    updateStatusDiv("Listening...");

    // Clear previous transcription
    document.querySelector("#output-generation").textContent = "";
    document.querySelector("#output-generation").hidden = true;
    document.querySelector("#output-placeholder").hidden = true;
  } catch (error) {
    console.error("Error starting microphone:", error);
    updateStatusDiv("Error: " + error.message);
    pendingStart = false;
  }
}

document.querySelector("#speech-button").addEventListener("click", async () => {
  const button = document.querySelector("#speech-button");

  if (!isRecording) {
    // Check if model is ready
    if (!modelInitialized) {
      pendingStart = true;
      initializeModel();
      return;
    }

    await startRecording();
  } else {
    stopMicrophone();

    // End streaming session
    moshiWorker.postMessage({ command: "stop_stream" });

    isRecording = false;
    button.textContent = "Start Speech";
    button.className =
      "bg-gray-700 hover:bg-gray-800 text-white font-normal py-2 px-4 rounded";
    updateStatusDiv("Ready to start");
  }
});
