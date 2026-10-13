import { invoke } from "@tauri-apps/api/core";

export type Telemetry = {
  denoise_ms: number;
  vad_ms: number;
  asr_ms: number;
  mt_ms: number;
  tts_first_chunk_ms: number;
  tts_total_ms: number;
  to_first_audio_ms: number;
  total_ms: number;
};

export type CycleResult = {
  wav: number[];
  transcript: string;
  translation: string;
  source_lang: string;
  target_lang: string;
  confidence: number;
  no_speech: boolean;
  low_confidence: boolean;
  telemetry: Telemetry;
};

export const isReady = (): Promise<boolean> => invoke("ready");

/**
 * Runs one cycle on 48 kHz mono WAV bytes. `pcmWav` maps to the Rust
 * `pcm_wav: Vec<u8>` argument (Tauri converts camelCase → snake_case).
 */
export const processUtterance = (
  pcmWav: Uint8Array,
  source: string,
  target: string,
): Promise<CycleResult> =>
  invoke("process_utterance", {
    pcmWav: Array.from(pcmWav),
    source,
    target,
  });
