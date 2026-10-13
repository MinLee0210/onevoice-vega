// Browser mic capture (48 kHz mono, RNNoise's required input rate), WAV
// encoding, and playback — no native audio code, so the demo needs no
// cpal/ALSA port. Capture uses an AudioWorklet; ScriptProcessor is only a
// compatibility fallback for older webviews.

export const SAMPLE_RATE = 48_000;

export type Recorder = {
  stop: () => Promise<Float32Array>;
  /** The actual device rate; browsers may ignore the requested 48 kHz rate. */
  sampleRate: number;
};

/** Starts capturing mic audio at 48 kHz mono. Call `stop()` to finish. */
export async function startRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false },
  });
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = ctx.createMediaStreamSource(stream);
  const chunks: Float32Array[] = [];
  const silent = ctx.createGain();
  silent.gain.value = 0;
  let capture: AudioNode;
  let workletUrl: string | undefined;

  try {
    const code = `class VegaCapture extends AudioWorkletProcessor {
      process(inputs) {
        const channel = inputs[0] && inputs[0][0];
        if (channel) { const copy = new Float32Array(channel); this.port.postMessage(copy, [copy.buffer]); }
        return true;
      }
    }
    registerProcessor("vega-capture", VegaCapture);`;
    workletUrl = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, "vega-capture", { channelCount: 1 });
    node.port.onmessage = (event) => chunks.push(new Float32Array(event.data));
    capture = node;
  } catch {
    const fallback = ctx.createScriptProcessor(4096, 1, 1);
    fallback.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    capture = fallback;
  }

  source.connect(capture);
  capture.connect(silent);
  silent.connect(ctx.destination);

  const stop = async (): Promise<Float32Array> => {
    capture.disconnect();
    source.disconnect();
    silent.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    if (workletUrl) URL.revokeObjectURL(workletUrl);
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

  return { stop, sampleRate: ctx.sampleRate };
}

/** Linear resampling for captured mono PCM before the 48 kHz RNNoise path. */
export function resampleMono(samples: Float32Array, fromRate: number, toRate = SAMPLE_RATE): Float32Array {
  if (fromRate === toRate) return samples;
  const length = Math.round(samples.length * toRate / fromRate);
  const output = new Float32Array(length);
  const ratio = fromRate / toRate;
  for (let i = 0; i < length; i++) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const mix = position - left;
    output[i] = (samples[left] ?? 0) * (1 - mix) + (samples[right] ?? 0) * mix;
  }
  return output;
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

/** Decodes any browser-supported audio file into 48 kHz mono WAV bytes. */
export async function fileToWav48k(file: File): Promise<Uint8Array> {
  const bytes = await file.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(bytes);
  // Always render: this both resamples and downmixes stereo/multichannel files.
  const len = Math.ceil(decoded.duration * SAMPLE_RATE);
  const off = new OfflineAudioContext(1, len, SAMPLE_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const mono = (await off.startRendering()).getChannelData(0);
  await ctx.close();
  return encodeWav(mono, SAMPLE_RATE);
}

/** Plays WAV bytes through the speakers. */
/** Resolves only after playback ends, so broadcast legs cannot overlap. */
export async function playWav(bytes: Uint8Array, onStart?: () => void): Promise<void> {
  const ctx = new AudioContext();
  const data = await ctx.decodeAudioData(bytes.buffer.slice(0) as ArrayBuffer);
  const src = ctx.createBufferSource();
  src.buffer = data;
  src.connect(ctx.destination);
  await new Promise<void>((resolve) => {
    src.onended = () => {
      void ctx.close();
      resolve();
    };
    src.start();
    onStart?.();
  });
}
