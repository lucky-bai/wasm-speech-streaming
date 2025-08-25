use crate::config::Config;
use crate::console_log;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

pub struct MoshiModel {
    state: moshi::asr::State,
    text_tokenizer: Tokenizer,
    dev: Device,
}

impl MoshiModel {
    pub fn load_from_buffers(
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

        let state = moshi::asr::State::new(1, 0, 0., audio_tokenizer, lm)?;
        console_log!("Created ASR state");

        Ok(MoshiModel {
            state,
            text_tokenizer,
            dev: dev.clone(),
        })
    }

    pub fn process_chunk<F>(&mut self, pcm_chunk: &[f32], mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        let pcm_tensor = Tensor::new(pcm_chunk, &self.dev)?.reshape((1, 1, ()))?;
        let asr_msgs = self
            .state
            .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        for asr_msg in asr_msgs.iter() {
            match asr_msg {
                moshi::asr::AsrMsg::Step { .. } => {}
                moshi::asr::AsrMsg::EndWord { .. } => {}
                moshi::asr::AsrMsg::Word { tokens, .. } => {
                    let word = self
                        .text_tokenizer
                        .decode(&tokens, true)
                        .unwrap_or_else(|_| String::new());

                    if !word.trim().is_empty() {
                        callback(&word);
                    }
                }
            }
        }

        Ok(())
    }
}
