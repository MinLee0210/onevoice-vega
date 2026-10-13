import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import { encodeWav, playWav, startRecording, type Recorder } from "./audio";

const LANGS: Record<string, string> = { vi: "VI", en: "EN", zh: "ZH", ko: "KO" };

type Phase = "idle" | "loading" | "listening" | "translating" | "speaking" | "error";

export default function App() {
  const [source, setSource] = useState("vi");
  const [target, setTarget] = useState("en");
  const [phase, setPhase] = useState<Phase>("loading");
  const [ready, setReady] = useState(false);
  const [result, setResult] = useState<api.CycleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<Recorder | null>(null);

  useEffect(() => {
    api
      .isReady()
      .then((ok) => {
        setReady(ok);
        setPhase(ok ? "idle" : "error");
        if (!ok) setError("Models not found — run scripts/setup_models.sh (or set ONEVOICE_ROOT).");
      })
      .catch((e) => {
        setError(String(e));
        setPhase("error");
      });
  }, []);

  const press = async () => {
    if (!ready || phase === "listening" || phase === "translating") return;
    setError(null);
    try {
      recorder.current = await startRecording();
      setPhase("listening");
    } catch (e) {
      setError(`Mic error: ${e}`);
      setPhase("error");
    }
  };

  const release = async () => {
    if (phase !== "listening" || !recorder.current) return;
    const samples = await recorder.current.stop();
    recorder.current = null;
    if (samples.length < 4800) {
      setError("Too short — hold the button while you speak.");
      setPhase("idle");
      return;
    }
    setPhase("translating");
    try {
      const res = await api.processUtterance(encodeWav(samples), source, target);
      setResult(res);
      if (res.no_speech || res.low_confidence) {
        setError(res.no_speech ? "Didn't catch that — no speech." : "Low confidence — say it again.");
        setPhase("idle");
        return;
      }
      setPhase("speaking");
      await playWav(new Uint8Array(res.wav));
      setPhase("idle");
    } catch (e) {
      setError(String(e));
      setPhase("error");
    }
  };

  // Space bar = push to talk.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        void press();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        void release();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  });

  const te = result?.telemetry;
  const pivot = !["en", "vi"].includes(target) || !["en", "vi"].includes(source);

  return (
    <div className="app">
      <header>
        <span className="brand">VEGA</span>
        <span className="badge offline" title="No network calls — the engine runs on-device">
          ● Offline
        </span>
      </header>

      <section className="controls">
        <select value={source} onChange={(e) => setSource(e.target.value)} disabled={phase === "listening"}>
          {Object.entries(LANGS).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        <button className="swap" onClick={() => { setSource(target); setTarget(source); }}>⇄</button>
        <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={phase === "listening"}>
          {Object.entries(LANGS).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        {pivot && <span className="pivot">via EN</span>}
      </section>

      <button
        className={`ptt ${phase === "listening" ? "active" : ""}`}
        onMouseDown={press}
        onMouseUp={release}
        onMouseLeave={() => phase === "listening" && release()}
        disabled={!ready || phase === "translating"}
      >
        {phase === "listening" ? "● LISTENING…" : phase === "translating" ? "translating…" : "HOLD TO TALK · or Space"}
      </button>

      <section className="cards">
        <div className="card">
          <label>transcript ({LANGS[source]})</label>
          <p>{result?.transcript || "—"}</p>
        </div>
        <div className="card">
          <label>translation ({LANGS[target]})</label>
          <p>{result?.translation || "—"}</p>
        </div>
      </section>

      {te && (
        <section className="hud">
          <div className="hud-head">
            <span>⏱ TTFA {(te.to_first_audio_ms / 1000).toFixed(2)}s</span>
            <span>conf {result?.confidence.toFixed(2)}</span>
          </div>
          <div className="bars">
            {[
              ["denoise", te.denoise_ms],
              ["VAD", te.vad_ms],
              ["ASR", te.asr_ms],
              ["MT", te.mt_ms],
              ["TTS", te.tts_total_ms],
            ].map(([name, ms]) => (
              <div className="bar" key={name as string}>
                <span>{name}</span>
                <div className="track">
                  <div style={{ width: `${Math.min(100, ((ms as number) / 500) * 100)}%` }} />
                </div>
                <span>{(ms as number).toFixed(0)}ms</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}
