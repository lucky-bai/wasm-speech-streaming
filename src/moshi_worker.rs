use candle_core::Device;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::console_log;
use crate::model::MoshiModel;
use crate::utils::convert_audio_to_pcm;

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
                let pcm_result = convert_audio_to_pcm(audio);
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
}
