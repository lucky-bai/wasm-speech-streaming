use crate::config::Config;
use crate::console_log;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

pub struct MoshiModel {
    state: moshi::asr::State,
    text_tokenizer: Tokenizer,
    timestamps: bool,
    config: Config,
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

    pub fn transcribe_streaming<F>(&mut self, pcm: Vec<f32>, mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
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

        let mut last_word: Option<(String, f64)> = None;

        for pcm_chunk in pcm.chunks(1920) {
            let pcm_tensor = Tensor::new(pcm_chunk, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self
                .state
                .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { .. } => {}
                    moshi::asr::AsrMsg::EndWord { .. } => {
                        if let Some((word, _)) = last_word.take() {
                            callback(&word);
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
                            if let Some((prev_word, _)) = last_word.take() {
                                callback(&prev_word);
                            }
                            last_word = Some((word, *start_time));
                        } else {
                            callback(&word);
                        }
                    }
                }
            }
        }

        if let Some((word, _)) = last_word.take() {
            callback(&word);
        }

        Ok(())
    }
}
