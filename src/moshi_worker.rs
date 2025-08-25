use candle_core::Device;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

use crate::console_log;
use crate::model::MoshiModel;

#[wasm_bindgen]
pub struct MoshiASRDecoder {
    inner: Option<RefCell<MoshiModel>>,
    streaming: RefCell<bool>,
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
                    streaming: RefCell::new(false),
                }
            }
            Err(e) => {
                console_log!("Failed to load Moshi model: {:?}", e);
                Self {
                    inner: None,
                    streaming: RefCell::new(false),
                }
            }
        }
    }

    pub fn start_streaming(&self) {
        *self.streaming.borrow_mut() = true;
    }

    pub fn stop_streaming(&self) {
        *self.streaming.borrow_mut() = false;
    }

    pub fn process_audio_chunk(&self, audio_data: &[f32], callback: &js_sys::Function) {
        if !*self.streaming.borrow() {
            return;
        }

        if let Some(model_cell) = &self.inner {
            let mut model = model_cell.borrow_mut();
            let _ = model.process_chunk(audio_data, |word| {
                let this = &JsValue::NULL;
                let word_js = JsValue::from_str(word);
                let _ = callback.call1(this, &word_js);
            });
        }
    }
}
