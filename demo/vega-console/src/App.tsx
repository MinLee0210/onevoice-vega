import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import { encodeWav, fileToWav48k, playWav, startRecording, type Recorder } from "./audio";

const LANGS: Record<string, string> = { vi: "VI", en: "EN", zh: "ZH", ko: "KO" };
const ALL = ["en", "zh", "ko"];

type Phase = "idle" | "loading" | "listening" | "translating" | "speaking" | "error";
type Mode = "hand" | "air";

function Chips({ terms }: { terms: string[] }) {
  if (!terms.length) return null;
  return (
    <div className="chips">
      {terms.map((t) => <span className="chip" key={t}>🔒 {t}</span>)}
    </div>
  );
}

/** Device status is LED-only (no screen on the real handheld). */
function Leds({ phase }: { phase: Phase }) {
  const label =
    phase === "listening" ? "recording" :
    phase === "translating" ? "translating" :
    phase === "speaking" ? "playing" : "ready";
  return (
    <div className="leds">
      <span className={`led ${phase === "listening" ? "rec on" : ""}`} />
      <span className={`led ${phase === "translating" ? "work on" : ""}`} />
      <span className={`led ${phase === "speaking" ? "done on" : ""}`} />
      <span>{label}</span>
    </div>
  );
}

function Hud({ te, confidence }: { te: api.Telemetry; confidence: number }) {
  return (
    <section className="hud">
      <div className="hud-head">
        <span>⏱ TTFA {(te.to_first_audio_ms / 1000).toFixed(2)}s</span>
        <span>conf {confidence.toFixed(2)}</span>
      </div>
      <div className="bars">
        {([["denoise", te.denoise_ms], ["VAD", te.vad_ms], ["ASR", te.asr_ms], ["MT", te.mt_ms], ["TTS", te.tts_total_ms]] as const).map(
          ([name, ms]) => (
            <div className="bar" key={name}>
              <span>{name}</span>
              <div className="track"><div style={{ width: `${Math.min(100, (ms / 500) * 100)}%` }} /></div>
              <span>{ms.toFixed(0)}ms</span>
            </div>
          ),
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>("hand");
  const [source, setSource] = useState("vi");
  const [target, setTarget] = useState("en");
  const [airTargets, setAirTargets] = useState<string[]>(["en", "zh", "ko"]);
  const [phase, setPhase] = useState<Phase>("loading");
  const [ready, setReady] = useState(false);
  const [result, setResult] = useState<api.CycleResult | null>(null);
  const [bc, setBc] = useState<api.BroadcastResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<Recorder | null>(null);
  const busy = phase === "listening" || phase === "translating" || phase === "speaking";

  useEffect(() => {
    api.isReady()
      .then((ok) => {
        setReady(ok);
        setPhase(ok ? "idle" : "error");
        if (!ok) setError("Models not found — run scripts/setup_models.sh (or set ONEVOICE_ROOT).");
      })
      .catch((e) => { setError(String(e)); setPhase("error"); });
  }, []);

  // Shared path for both mic and loaded-audio input.
  const runWav = async (wav: Uint8Array) => {
    setPhase("translating");
    setError(null);
    try {
      if (mode === "hand") {
        const res = await api.processUtterance(wav, source, target);
        setResult(res);
        if (res.no_speech || res.low_confidence) {
          setError(res.no_speech ? "Didn't catch that — no speech." : "Low confidence — try again.");
          setPhase("idle");
          return;
        }
        setPhase("speaking");
        await playWav(new Uint8Array(res.wav));
      } else {
        const res = await api.broadcast(wav, source, airTargets);
        setBc(res);
        if (res.no_speech || res.low_confidence || !res.legs.length) {
          setError(res.no_speech ? "Didn't catch that — no speech." : "Low confidence — try again.");
          setPhase("idle");
          return;
        }
        setPhase("speaking");
        for (const leg of res.legs) await playWav(new Uint8Array(leg.wav));
      }
      setPhase("idle");
    } catch (e) {
      setError(String(e));
      setPhase("error");
    }
  };

  const press = async () => {
    if (!ready || busy) return;
    setError(null);
    try {
      recorder.current = await startRecording();
      setPhase("listening");
    } catch (e) {
      setError(
        `Mic unavailable (${e}). In \`tauri dev\` the webview only exposes the mic in a secure ` +
          `context — use "Load audio" / "Use sample" below instead.`,
      );
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
    await runWav(encodeWav(samples));
  };

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      await runWav(await fileToWav48k(f));
    } catch (err) {
      setError(`Could not read audio: ${err}`);
      setPhase("error");
    }
    e.target.value = "";
  };

  const loadSample = async () => {
    try {
      await runWav(new Uint8Array(await api.sampleWav()));
    } catch (err) {
      setError(String(err));
      setPhase("error");
    }
  };

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) { e.preventDefault(); void press(); }
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") { e.preventDefault(); void release(); } };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  });

  const pivot = mode === "hand" && (["zh", "ko"].includes(target) || ["zh", "ko"].includes(source));

  return (
    <div className="app">
      <header>
        <span className="brand">VEGA</span>
        <div className="modes">
          <button className={mode === "hand" ? "on" : ""} onClick={() => setMode("hand")}>Hand</button>
          <button className={mode === "air" ? "on" : ""} onClick={() => setMode("air")}>Air</button>
        </div>
        <span className="badge offline" title="No network calls — runs on-device">● Offline</span>
      </header>

      <section className="controls">
        <select value={source} onChange={(e) => setSource(e.target.value)} disabled={busy}>
          {Object.entries(LANGS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        {mode === "hand" ? (
          <>
            <button className="swap" onClick={() => { setSource(target); setTarget(source); }} disabled={busy}>⇄</button>
            <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={busy}>
              {Object.entries(LANGS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            {pivot && <span className="pivot">via EN</span>}
          </>
        ) : (
          <div className="air-targets">
            {ALL.map((t) => (
              <label key={t} className={airTargets.includes(t) ? "tg on" : "tg"}>
                <input
                  type="checkbox"
                  checked={airTargets.includes(t)}
                  disabled={busy}
                  onChange={(e) => setAirTargets((prev) => (e.target.checked ? [...prev, t] : prev.filter((x) => x !== t)))}
                />
                {LANGS[t]}
              </label>
            ))}
          </div>
        )}
      </section>

      <button
        className={`ptt ${phase === "listening" ? "active" : ""}`}
        onMouseDown={press}
        onMouseUp={release}
        onMouseLeave={() => phase === "listening" && release()}
        disabled={!ready || busy}
      >
        {phase === "listening"
          ? "LIVE — release to translate"
          : phase === "translating"
            ? "TRANSLATING…"
            : phase === "speaking"
              ? "PLAYING…"
              : "HOLD TO SPEAK"}
      </button>

      <Leds phase={phase} />

      <section className="input-audio">
        <span className="input-label">Input audio</span>
        <label className="filebtn">
          ⬆ Load audio file
          <input type="file" accept="audio/*,.wav" onChange={onFile} disabled={busy} hidden />
        </label>
        <button className="filebtn" onClick={loadSample} disabled={busy}>▶ Use sample (VI)</button>
        <span className="hint">any sample rate — resampled to 48 kHz</span>
      </section>

      {mode === "hand" ? (
        <>
          <section className="cards">
            <div className="card">
              <label>transcript ({LANGS[source]})</label>
              <p>{result?.transcript || "—"}</p>
            </div>
            <div className="card">
              <label>translation ({LANGS[target]})</label>
              <p>{result?.translation || "—"}</p>
              <Chips terms={result?.glossary_terms ?? []} />
            </div>
          </section>
          {result && <Hud te={result.telemetry} confidence={result.confidence} />}
        </>
      ) : (
        <section className="cards">
          <div className="card full">
            <label>broadcast ({LANGS[source]}) → {airTargets.map((t) => LANGS[t]).join(" / ")}</label>
            <p>{bc?.transcript || "—"}</p>
            <div className="legs">
              {bc?.legs.map((leg) => (
                <div className="leg" key={leg.lang}>
                  <span className="leg-lang">{LANGS[leg.lang]}</span>
                  <span className="leg-text">{leg.translation}</span>
                  <Chips terms={leg.glossary_terms} />
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}
