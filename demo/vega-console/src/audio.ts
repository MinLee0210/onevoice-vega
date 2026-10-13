// Browser mic capture (48 kHz mono, RNNoise's required input rate), WAV
// encoding, and playback — no native audio code, so the demo needs no
// cpal/ALSA port. NOTE: ScriptProcessorNode is deprecated; the production
// path is an AudioWorklet (see docs/decisions/demo-frontend-design-2026-10.md).

export const SAMPLE_RATE = 48_000;

export type Recorder = {
  stop: () => Promise<Float32Array>;
};

/** Starts capturing mic audio at 48 kHz mono. Call `stop()` to finish. */
export async function startRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false },
  });
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);

  const chunks: Float32Array[] = [];
  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(ctx.destination);

  const stop = async (): Promise<Float32Array> => {
    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const out = new Float32Array(total);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    await ctx.close();
    return out;
  };

  return { stop };
}

/** Encodes mono float32 samples as a 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, sampleRate = SAMPLE_RATE): Uint8Array {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const w = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  w(36, "data");
  view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Uint8Array(buf);
}

/** Plays WAV bytes through the speakers. */
export async function playWav(bytes: Uint8Array): Promise<void> {
  const ctx = new AudioContext();
  const data = await ctx.decodeAudioData(bytes.buffer.slice(0) as ArrayBuffer);
  const src = ctx.createBufferSource();
  src.buffer = data;
  src.connect(ctx.destination);
  src.onended = () => void ctx.close();
  src.start();
}
