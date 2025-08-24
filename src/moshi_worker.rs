use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::cell::RefCell;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Debug, serde::Deserialize)]
struct SttConfig {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    stt_config: SttConfig,
}

impl Config {
    fn model_config(&self) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }
}

struct MoshiModel {
    state: moshi::asr::State,
    text_tokenizer: Tokenizer,
    timestamps: bool,
    config: Config,
    dev: Device,
}

impl MoshiModel {
    fn load_from_buffers(
        weights: &[u8],
        tokenizer: &[u8],
        mimi: &[u8],
        config_bytes: &[u8],
        dev: &Device,
    ) -> Result<Self> {
        let dtype = DType::F32;

        // Parse config
        let config: Config = serde_json::from_slice(config_bytes)?;
        console_log!("Parsed config successfully");

        // Load text tokenizer
        let text_tokenizer = Tokenizer::from_bytes(tokenizer).unwrap();
        console_log!("Loaded text tokenizer");

        // Load model weights
        let vb_lm =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(weights, dev)?;
        console_log!("Loaded model weights");

        let vb_mimi = candle_nn::VarBuilder::from_slice_safetensors(mimi, dtype, dev)?;
        let cfg = moshi::mimi::Config::v0_1(Some(32));
        let audio_tokenizer = moshi::mimi::Mimi::new(cfg, vb_mimi)?;
        console_log!("Loaded audio tokenizer");

        // Create LM model
        let lm = moshi::lm::LmModel::new(
            &config.model_config(),
            moshi::nn::MaybeQuantizedVarBuilder::Quantized(vb_lm),
        )?;
        console_log!("Created LM model");

        let asr_delay_in_tokens = (config.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0., audio_tokenizer, lm)?;
        console_log!("Created ASR state");

        Ok(MoshiModel {
            state,
            config,
            text_tokenizer,
            timestamps: true,
            dev: dev.clone(),
        })
    }

    fn transcribe(&mut self, pcm: Vec<f32>) -> Result<Vec<String>> {
        // Add the silence prefix to the audio.
        let mut pcm = pcm;
        if self.config.stt_config.audio_silence_prefix_seconds > 0.0 {
            let silence_len =
                (self.config.stt_config.audio_silence_prefix_seconds * 24000.0) as usize;
            pcm.splice(0..0, vec![0.0; silence_len]);
        }
        // Add some silence at the end to ensure all the audio is processed.
        let suffix = (self.config.stt_config.audio_delay_seconds * 24000.0) as usize;
        pcm.resize(pcm.len() + suffix + 24000, 0.0);

        let mut transcription = Vec::new();
        let mut last_word = None;

        for pcm_chunk in pcm.chunks(1920) {
            let pcm_tensor = Tensor::new(pcm_chunk, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self
                .state
                .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { .. } => {}
                    moshi::asr::AsrMsg::EndWord { stop_time, .. } => {
                        if let Some((word, start_time)) = last_word.take() {
                            console_log!("Word: [{}-{}] {}", start_time, stop_time, word);
                            transcription.push(word);
                        }
                    }
                    moshi::asr::AsrMsg::Word {
                        tokens, start_time, ..
                    } => {
                        let word = self
                            .text_tokenizer
                            .decode(&tokens, true)
                            .unwrap_or_else(|_| String::new());

                        if self.timestamps {
                            if let Some((prev_word, prev_start_time)) = last_word.take() {
                                console_log!(
                                    "Word: [{}-{}] {}",
                                    prev_start_time,
                                    start_time,
                                    prev_word
                                );
                                transcription.push(prev_word);
                            }
                            last_word = Some((word, *start_time));
                        } else {
                            transcription.push(word);
                        }
                    }
                }
            }
        }

        if let Some((word, start_time)) = last_word.take() {
            console_log!("Final word: [{}] {}", start_time, word);
            transcription.push(word);
        }

        Ok(transcription)
    }
}

#[wasm_bindgen]
pub struct MoshiASRDecoder {
    inner: Option<RefCell<MoshiModel>>,
}

#[wasm_bindgen]
impl MoshiASRDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8], tokenizer: &[u8], mimi: &[u8], config: &[u8]) -> Self {
        // Load the Moshi model from buffers
        let device = Device::Cpu;
        let model_result = MoshiModel::load_from_buffers(weights, tokenizer, mimi, config, &device);
        match model_result {
            Ok(model) => {
                console_log!("Successfully loaded Moshi model!");
                Self {
                    inner: Some(RefCell::new(model)),
                }
            }
            Err(e) => {
                console_log!("Failed to load Moshi model: {:?}", e);
                Self { inner: None }
            }
        }
    }

    pub fn decode(&self, audio: &[u8]) -> String {
        match &self.inner {
            Some(model_cell) => {
                console_log!("Using loaded Moshi model for transcription");

                // Convert audio bytes to PCM f32 data
                let pcm_result = self.convert_audio_to_pcm(audio);
                match pcm_result {
                    Ok(pcm_data) => {
                        console_log!("Converted audio to PCM, {} samples", pcm_data.len());

                        // Perform transcription
                        let mut model = model_cell.borrow_mut();
                        match model.transcribe(pcm_data) {
                            Ok(words) => {
                                console_log!("Successfully transcribed {} words", words.len());
                                let full_text = words.join(" ");

                                // Format as segments JSON (compatible with existing interface)
                                let json_result = serde_json::json!([{
                                    "start": 0.0,
                                    "duration": 1.0,
                                    "dr": {
                                        "tokens": [],
                                        "text": full_text,
                                        "avg_logprob": -0.5,
                                        "no_speech_prob": 0.1,
                                        "temperature": 0.0,
                                        "compression_ratio": 1.0
                                    }
                                }]);

                                console_log!("Returning transcription: {}", full_text);
                                json_result.to_string()
                            }
                            Err(e) => {
                                console_log!("Failed to transcribe audio: {:?}", e);
                                r#"[{"start": 0.0, "duration": 1.0, "dr": {"tokens": [], "text": "Transcription failed", "avg_logprob": -1.0, "no_speech_prob": 0.0, "temperature": 0.0, "compression_ratio": 1.0}}]"#.to_string()
                            }
                        }
                    }
                    Err(e) => {
                        console_log!("Failed to convert audio format: {:?}", e);
                        r#"[{"start": 0.0, "duration": 1.0, "dr": {"tokens": [], "text": "Audio format error", "avg_logprob": -1.0, "no_speech_prob": 0.0, "temperature": 0.0, "compression_ratio": 1.0}}]"#.to_string()
                    }
                }
            }
            None => {
                console_log!("No model loaded - returning error");
                r#"[{"start": 0.0, "duration": 1.0, "dr": {"tokens": [], "text": "Model not loaded", "avg_logprob": -1.0, "no_speech_prob": 0.0, "temperature": 0.0, "compression_ratio": 1.0}}]"#.to_string()
            }
        }
    }

    fn convert_audio_to_pcm(&self, audio: &[u8]) -> Result<Vec<f32>> {
        // Try to parse as WAV file
        let mut cursor = std::io::Cursor::new(audio);
        let wav_reader = hound::WavReader::new(&mut cursor)?;
        let spec = wav_reader.spec();

        console_log!(
            "Audio format: sample_rate={}, channels={}, bits_per_sample={}",
            spec.sample_rate,
            spec.channels,
            spec.bits_per_sample
        );

        // Moshi expects 24kHz mono
        if spec.sample_rate != 24000 {
            console_log!("Warning: Expected 24kHz audio, got {}Hz", spec.sample_rate);
        }

        // Convert to f32 PCM
        let mut data = wav_reader.into_samples::<i16>().collect::<Vec<_>>();
        data.truncate(data.len() / spec.channels as usize); // Take only first channel if stereo

        let mut pcm_data = Vec::with_capacity(data.len());
        for sample in data.into_iter() {
            let sample = sample?;
            pcm_data.push(sample as f32 / 32768.0); // Convert i16 to f32 [-1, 1]
        }

        console_log!("Converted to {} PCM samples", pcm_data.len());
        Ok(pcm_data)
    }
}
