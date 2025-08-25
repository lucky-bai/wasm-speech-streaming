use wasm_bindgen::prelude::*;
use anyhow::Result;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (crate::utils::log(&format_args!($($t)*).to_string()))
}

pub fn convert_audio_to_pcm(audio: &[u8]) -> Result<Vec<f32>> {
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