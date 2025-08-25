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

    pub fn decode_streaming(&self, audio: &[u8], callback: &js_sys::Function) {
        if let Some(model_cell) = &self.inner {
            if let Ok(pcm_data) = convert_audio_to_pcm(audio) {
                let mut model = model_cell.borrow_mut();
                let _ = model.transcribe_streaming(pcm_data, |word| {
                    let this = &JsValue::NULL;
                    let word_js = JsValue::from_str(word);
                    let _ = callback.call1(this, &word_js);
                });
            }
        }
    }
}
