import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import { encodeWav, fileToWav48k, playWav, resampleMono, startRecording, type Recorder } from "./audio";

const LANGS: Record<string, string> = { vi: "VI", en: "EN", zh: "ZH", ko: "KO" };
const LANGUAGE_NAMES: Record<string, string> = { vi: "Vietnamese", en: "English", zh: "Chinese", ko: "Korean" };
const HAND_TARGETS: Record<string, string[]> = { vi: ["en", "zh", "ko"], en: ["vi"], zh: ["vi"], ko: ["vi"] };
const AIR_TARGETS = ["en", "zh", "ko"];
type Phase = "idle" | "loading" | "listening" | "translating" | "speaking" | "error";
type Mode = "hand" | "air";
type InputMode = "voice" | "text";

function Chips({ terms }: { terms: string[] }) {
  if (!terms.length) return null;
  return <div className="chips" aria-label="Enforced glossary terms">{terms.map((term) => <span className="chip" key={term}>Glossary · {term}</span>)}</div>;
}

function Hud({ te, confidence, playbackStartMs }: { te: api.Telemetry; confidence: number; playbackStartMs: number | null }) {
  return <details className="hud"><summary><span>Pipeline details</span><span>{playbackStartMs === null ? "Playback pending" : `${(playbackStartMs / 1000).toFixed(2)}s to playback`} <span aria-hidden="true">⌄</span></span></summary>
    <div className="hud-body"><p>Engine first chunk {(te.to_first_audio_ms / 1000).toFixed(2)}s · full cycle {(te.total_ms / 1000).toFixed(2)}s · recognition confidence {confidence.toFixed(2)}</p>
      <div className="bars">{([["Denoise", te.denoise_ms], ["Voice activity", te.vad_ms], ["Speech recognition", te.asr_ms], ["Translation", te.mt_ms], ["Speech output", te.tts_total_ms]] as const).map(([name, ms]) => <div className="bar" key={name}>
        <span>{name}</span><div className="track"><div style={{ width: `${Math.min(100, (ms / 500) * 100)}%` }} /></div><span>{ms.toFixed(0)} ms</span>
      </div>)}</div>
    </div>
  </details>;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("hand");
  const [inputMode, setInputMode] = useState<InputMode>("voice");
  const [speechText, setSpeechText] = useState("");
  const [speechLang, setSpeechLang] = useState("en");
  const [speechPhase, setSpeechPhase] = useState<"idle" | "synthesizing" | "playing">("idle");
  const [lastSpoken, setLastSpoken] = useState<{ text: string; lang: string; wav: number[] } | null>(null);
  const [source, setSource] = useState("vi");
  const [target, setTarget] = useState("en");
  const [airTargets, setAirTargets] = useState<string[]>(AIR_TARGETS);
  const [phase, setPhase] = useState<Phase>("loading");
  const [ready, setReady] = useState(false);
  const [result, setResult] = useState<api.CycleResult | null>(null);
  const [bc, setBc] = useState<api.BroadcastResult | null>(null);
  const [activeAirLanguage, setActiveAirLanguage] = useState<string | null>(null);
  const [airSequencePlaying, setAirSequencePlaying] = useState(false);
  const [playbackStartMs, setPlaybackStartMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<Recorder | null>(null);
  const held = useRef(false);
  const keyboardHeld = useRef(false);
  const starting = useRef(false);
  const airReplayInFlight = useRef(false);
  const busy = phase === "listening" || phase === "translating" || phase === "speaking" || speechPhase !== "idle";
  const handTargets = HAND_TARGETS[source];

  useEffect(() => {
    api.isReady().then((ok) => {
      setReady(ok); setPhase(ok ? "idle" : "error");
      if (!ok) setError("Model bundle not found — run scripts/setup_models.sh or set ONEVOICE_ROOT.");
    }).catch((err) => { setError(String(err)); setPhase("error"); });
  }, []);

  const clearOutput = () => { setResult(null); setBc(null); setPlaybackStartMs(null); setActiveAirLanguage(null); setAirSequencePlaying(false); };
  const selectMode = (next: Mode) => { if (!busy && next !== mode) { setMode(next); setError(null); clearOutput(); if (next === "air") setSource("vi"); } };
  const selectInputMode = (next: InputMode) => {
    if (busy) return;
    setInputMode(next);
    if (ready) { setError(null); setPhase("idle"); }
    if (next === "text") setSpeechLang(mode === "hand" ? target : airTargets[0]);
  };
  const selectSource = (next: string) => { setSource(next); clearOutput(); if (!HAND_TARGETS[next].includes(target)) setTarget(HAND_TARGETS[next][0]); };

  const runWav = async (wav: Uint8Array) => {
    const startedAt = performance.now();
    const selectedMode = mode, selectedSource = source, selectedTarget = target, selectedAirTargets = [...airTargets];
    setPhase("translating"); setError(null); setPlaybackStartMs(null); setActiveAirLanguage(null); setAirSequencePlaying(false);
    if (selectedMode === "hand") setResult(null); else setBc(null);
    try {
      if (selectedMode === "hand") {
        const res = await api.processUtterance(wav, selectedSource, selectedTarget);
        setResult(res);
        if (res.no_speech || res.low_confidence) {
          setError(res.no_speech ? "No speech detected. Hold PTT and try again." : "Low recognition confidence. Please repeat the instruction."); setPhase("idle"); return;
        }
        setPhase("speaking");
        await playWav(new Uint8Array(res.wav), () => setPlaybackStartMs(performance.now() - startedAt));
      } else {
        if (!selectedAirTargets.length) throw new Error("Choose at least one broadcast language.");
        const res = await api.broadcast(wav, selectedSource, selectedAirTargets);
        setBc(res);
        if (res.no_speech || res.low_confidence) {
          setError(res.no_speech ? "No speech detected. Hold PTT and try again." : "Low recognition confidence. Please repeat the instruction."); setPhase("idle"); return;
        }
        if (!res.legs.length) throw new Error("No broadcast output was created.");
        setPhase("speaking");
        setAirSequencePlaying(true);
        for (const leg of res.legs) { setActiveAirLanguage(leg.lang); await playWav(new Uint8Array(leg.wav)); }
        setActiveAirLanguage(null);
        setAirSequencePlaying(false);
      }
      setPhase("idle");
    } catch (err) { setActiveAirLanguage(null); setAirSequencePlaying(false); setError(String(err)); setPhase("error"); }
  };

  const press = async () => {
    if (!ready || busy || starting.current || inputMode !== "voice") return;
    held.current = true;
    starting.current = true;
    setError(null);
    try {
      recorder.current = await startRecording();
      setPhase("listening");
      if (!held.current) void release();
    } catch (err) { held.current = false; setError(`Microphone unavailable (${err}). Use Load audio or the VI sample instead.`); setPhase("error"); }
    finally { starting.current = false; }
  };
  const release = async () => {
    held.current = false;
    if (!recorder.current) return;
    const activeRecorder = recorder.current; recorder.current = null;
    const normalized = resampleMono(await activeRecorder.stop(), activeRecorder.sampleRate);
    if (normalized.length < 4800) { setError("Too short — hold the button while you speak."); setPhase("idle"); return; }
    await runWav(encodeWav(normalized));
  };
  const onFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; if (!file) return;
    try { await runWav(await fileToWav48k(file)); } catch (err) { setError(`Could not read audio: ${err}`); setPhase("error"); }
    event.target.value = "";
  };
  const loadSample = async () => { try { await runWav(new Uint8Array(await api.sampleWav())); } catch (err) { setError(String(err)); setPhase("error"); } };
  const speakTyped = async () => {
    const text = speechText.trim();
    if (!ready || busy || !text) return;
    setError(null); setSpeechPhase("synthesizing");
    try {
      const wav = await api.speakText(text, speechLang);
      setLastSpoken({ text, lang: speechLang, wav });
      setSpeechPhase("playing");
      await playWav(new Uint8Array(wav));
    } catch (err) { setError(`Could not speak text: ${String(err)}`); }
    finally { setSpeechPhase("idle"); }
  };
  const replayTyped = async () => {
    if (busy || !lastSpoken) return;
    setSpeechPhase("playing");
    try { await playWav(new Uint8Array(lastSpoken.wav)); }
    catch (err) { setError(`Could not replay speech: ${String(err)}`); }
    finally { setSpeechPhase("idle"); }
  };
  const replay = async () => {
    if (busy || !result?.wav.length) return;
    setPhase("speaking");
    try { await playWav(new Uint8Array(result.wav)); setPhase("idle"); } catch (err) { setError(String(err)); setPhase("error"); }
  };
  const replayAirLeg = async (leg: api.BroadcastLeg) => {
    if (busy || airReplayInFlight.current || !leg.wav.length) return;
    airReplayInFlight.current = true;
    setError(null);
    setAirSequencePlaying(false);
    setActiveAirLanguage(leg.lang);
    setPhase("speaking");
    try { await playWav(new Uint8Array(leg.wav)); setPhase("idle"); }
    catch (err) { setError(`Could not play ${LANGUAGE_NAMES[leg.lang]} audio: ${String(err)}`); setPhase("error"); }
    finally { setActiveAirLanguage(null); airReplayInFlight.current = false; }
  };

  useEffect(() => {
    const interactive = (element: EventTarget | null) => element instanceof HTMLElement && Boolean(element.closest("button, input, textarea, select, a, [contenteditable='true']"));
    const down = (event: KeyboardEvent) => { if (inputMode === "voice" && event.code === "Space" && !event.repeat && !interactive(event.target)) { event.preventDefault(); keyboardHeld.current = true; void press(); } };
    const up = (event: KeyboardEvent) => { if (event.code === "Space" && keyboardHeld.current) { event.preventDefault(); keyboardHeld.current = false; void release(); } };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  });

  const pivot = mode === "hand" && ["zh", "ko"].includes(target);
  const status = speechPhase === "synthesizing" ? "Creating speech" : speechPhase === "playing" ? "Speaking text" : phase === "loading" ? "Checking models" : phase === "listening" ? "Listening" : phase === "translating" ? "Translating" : phase === "speaking" ? "Playing audio" : phase === "error" ? "Needs attention" : inputMode === "text" ? "Ready to speak text" : "Ready to record";
  return <div className="app-shell">
    <aside className="sidebar" aria-label="Console navigation">
      <div className="identity"><span className="identity-mark" aria-hidden="true">V</span><div><strong>VEGA</strong><span>by QUAVIO</span></div></div>
      <div className="nav-group"><span className="eyebrow">DEPLOYMENT</span><div className="mode-list" aria-label="Carrier mode">
        <button className={mode === "hand" ? "mode-option selected" : "mode-option"} onClick={() => selectMode("hand")} disabled={busy} aria-pressed={mode === "hand"}><span className="mode-icon" aria-hidden="true">◫</span><span><strong>Vega Hand</strong><small>One-to-one exchange</small></span><span className="mode-arrow" aria-hidden="true">↗</span></button>
        <button className={mode === "air" ? "mode-option selected" : "mode-option"} onClick={() => selectMode("air")} disabled={busy} aria-pressed={mode === "air"}><span className="mode-icon" aria-hidden="true">⌁</span><span><strong>Vega Air</strong><small>Multilingual broadcast</small></span><span className="mode-arrow" aria-hidden="true">↗</span></button>
      </div></div>
      <div className="sidebar-foot"><span className="eyebrow">ENGINE</span><div className="local-row"><span className="local-dot" aria-hidden="true"/><strong>Local processing</strong></div><p>Speech, translation and playback run on this device.</p><span className="version">ONEVOICE / CONSOLE 01</span></div>
    </aside>
    <main className="workspace">
      <header className="workspace-header"><div><span className="eyebrow">OPERATIONS CONSOLE / {mode === "hand" ? "01" : "02"}</span><h1>{inputMode === "text" ? "Speak a typed message" : mode === "hand" ? "Face-to-face translation" : "Multilingual broadcast"}</h1><p>{inputMode === "text" ? "Choose a local voice and play a reply through this console." : mode === "hand" ? "A direct voice link for conversations on the floor." : "One operator instruction, played in each selected language."}</p></div><div className={`status ${speechPhase !== "idle" ? speechPhase : phase}`} role="status" aria-live="polite"><span className="status-dot" aria-hidden="true"/>{status}</div></header>
      <div className="input-switch" aria-label="Input method"><span className="eyebrow">CHOOSE AN ACTION</span><div className="input-tabs"><button className={inputMode === "voice" ? "selected" : ""} aria-pressed={inputMode === "voice"} disabled={busy} onClick={() => selectInputMode("voice")}>Translate speech</button><button className={inputMode === "text" ? "selected" : ""} aria-pressed={inputMode === "text"} disabled={busy} onClick={() => selectInputMode("text")}>Text to speech</button></div></div>
      {inputMode === "voice" && <section className="route-panel" aria-label="Translation route"><div className="panel-title"><span className="eyebrow">01 / LANGUAGE ROUTE</span>{pivot && <span className="route-note">Via English pivot</span>}</div><div className="route-content">
        <div className="route-field"><label htmlFor="source-language">SOURCE</label>{mode === "air" ? <div className="fixed-source"><strong>VI</strong><span>Vietnamese · operator</span></div> : <select id="source-language" value={source} onChange={(event) => selectSource(event.target.value)} disabled={busy}>{Object.keys(LANGS).map((lang) => <option key={lang} value={lang}>{LANGUAGE_NAMES[lang]}</option>)}</select>}</div>
        <div className="route-divider" aria-hidden="true">→</div>
        {mode === "hand" ? <div className="route-field"><label htmlFor="target-language">OUTPUT</label><div className="target-field"><select id="target-language" value={target} onChange={(event) => { setTarget(event.target.value); clearOutput(); }} disabled={busy}>{handTargets.map((lang) => <option key={lang} value={lang}>{LANGUAGE_NAMES[lang]}</option>)}</select><button className="swap" aria-label="Swap source and target languages" title="Swap languages" onClick={() => { setSource(target); setTarget(source); clearOutput(); }} disabled={busy}>⇄</button></div></div> : <div className="route-field"><span className="field-label">BROADCAST TO</span><div className="air-targets" aria-label="Broadcast languages">{AIR_TARGETS.map((lang) => { const selected = airTargets.includes(lang); return <label key={lang} className={selected ? "tg on" : "tg"}><input type="checkbox" checked={selected} disabled={busy || (selected && airTargets.length === 1)} onChange={(event) => { setAirTargets((previous) => event.target.checked ? [...previous, lang] : previous.filter((item) => item !== lang)); clearOutput(); }}/><span>{LANGUAGE_NAMES[lang]}</span><span className="check" aria-hidden="true">✓</span></label>; })}</div></div>}
      </div></section>}
      {inputMode === "voice" ? <><section className={`capture-panel ${phase}`} aria-label="Voice input"><div className="capture-copy"><span className="eyebrow">CAPTURE</span><h2>{phase === "listening" ? "Listening to you" : phase === "translating" ? "Processing instruction" : phase === "speaking" ? "Output in progress" : "Speak an instruction"}</h2><p>{phase === "listening" ? "Release the button to translate." : mode === "air" ? "Hold to record your Vietnamese instruction, then release to broadcast." : "Hold to record, then release to translate and play the response."}</p></div><button className={`ptt ${phase === "listening" ? "active" : ""}`} disabled={!ready || (busy && phase !== "listening")} aria-pressed={phase === "listening"}
          onPointerDown={(event) => { if (event.button === 0) { event.currentTarget.setPointerCapture(event.pointerId); void press(); } }} onPointerUp={(event) => { if (event.button === 0) void release(); }} onPointerCancel={() => void release()}
          onKeyDown={(event) => { if (event.code === "Space" && !event.repeat) { event.preventDefault(); keyboardHeld.current = true; void press(); } }} onKeyUp={(event) => { if (event.code === "Space" && keyboardHeld.current) { event.preventDefault(); keyboardHeld.current = false; void release(); } }}><span className="ptt-symbol" aria-hidden="true">{phase === "listening" ? "◼" : "●"}</span><span>{phase === "listening" ? "Release to send" : phase === "translating" ? "Translating…" : phase === "speaking" ? "Playing…" : "Hold to speak"}</span></button><div className="capture-hint">Press and hold <kbd>SPACE</kbd> or use the button</div></section>
      <div className="demo-strip"><span>TRY WITH AUDIO</span><label className="utility-action">Upload a recording<input type="file" accept="audio/*,.wav" onChange={onFile} disabled={!ready || busy} hidden/></label><button className="utility-action" onClick={loadSample} disabled={!ready || busy || source !== "vi"} title={source === "vi" ? "Uses the bundled Vietnamese fixture" : "The bundled sample is Vietnamese"}>Run Vietnamese sample <span aria-hidden="true">↗</span></button><small>48 kHz mono input</small></div></> : <section className="text-panel" aria-label="Text to speech"><div className="text-panel-head"><div><span className="eyebrow">LOCAL VOICE OUTPUT</span><h2>Type a reply to speak</h2><p>Vega speaks this text exactly as written. No translation is applied.</p></div><span className="text-limit">{speechText.length}/280</span></div><label htmlFor="speech-text">MESSAGE TO SPEAK</label><textarea id="speech-text" value={speechText} onChange={(event) => setSpeechText(event.target.value)} maxLength={280} disabled={busy} placeholder="Type a short instruction or reply…" rows={3}/><div className="text-panel-actions"><div><label htmlFor="speech-language">VOICE LANGUAGE</label><select id="speech-language" value={speechLang} onChange={(event) => setSpeechLang(event.target.value)} disabled={busy}>{Object.keys(LANGS).map((lang) => <option key={lang} value={lang}>{LANGUAGE_NAMES[lang]}</option>)}</select></div><button className="speak-primary" onClick={() => void speakTyped()} disabled={!ready || busy || !speechText.trim()}>{speechPhase === "synthesizing" ? "Creating speech…" : speechPhase === "playing" ? "Speaking…" : "▶ Speak this text aloud"}</button></div></section>}
      {error && <div className="alert-error" role="alert"><strong>Attention required</strong><span>{error}</span></div>}
      <section className="results" aria-label={inputMode === "text" ? "Spoken text result" : mode === "hand" ? "Translation result" : "Broadcast result"}><div className="results-title"><div><span className="eyebrow">{inputMode === "text" ? "02 / OUTPUT" : "03 / OUTPUT"}</span><h2>{inputMode === "text" ? "Spoken reply" : mode === "hand" ? "Conversation" : "Broadcast sequence"}</h2></div>{inputMode === "text" && lastSpoken && <button className="replay" onClick={() => void replayTyped()} disabled={busy}>▶ <span>Speak again</span></button>}</div>
        {inputMode === "text" ? <article className="spoken-output"><div className="utterance-label"><span>SPOKEN TEXT</span><span>{lastSpoken ? LANGUAGE_NAMES[lastSpoken.lang] : LANGUAGE_NAMES[speechLang]}</span></div><p className={lastSpoken ? "" : "placeholder"}>{lastSpoken?.text || "Your spoken reply will appear here."}</p></article> : mode === "hand" ? <div className="exchange"><article className="utterance"><div className="utterance-label"><span>HEARD</span><span>{LANGUAGE_NAMES[source]}</span></div><p className={result?.transcript ? "" : "placeholder"}>{result?.transcript || "Your spoken instruction will appear here."}</p></article><div className="exchange-arrow" aria-hidden="true">↓</div><article className="utterance translation"><div className="utterance-label"><span>TRANSLATION</span><span>{LANGUAGE_NAMES[target]}</span></div><p className={result?.translation ? "" : "placeholder"}>{result?.translation || "The translated response will appear here."}</p><Chips terms={result?.glossary_terms ?? []}/>{result && !result.no_speech && !result.low_confidence && <button className="speak-translation" onClick={() => void replay()} disabled={busy}>▶ Speak this translation again</button>}</article></div> : <><div className="broadcast-source"><span className="utterance-label"><span>HEARD · VIETNAMESE</span></span><p className={bc?.transcript ? "" : "placeholder"}>{bc?.transcript || "Your operator instruction will appear here."}</p></div><div className="legs">{(bc?.legs ?? airTargets.map((lang) => ({lang, translation: "Awaiting broadcast", glossary_terms: [], wav: []}))).map((leg, index) => <div className={`leg ${activeAirLanguage === leg.lang ? "active" : ""}`} key={leg.lang}><span className="leg-order">{String(index + 1).padStart(2, "0")}</span><span className="leg-lang">{LANGUAGE_NAMES[leg.lang]}</span><div className="leg-content"><span className={bc ? "leg-text" : "leg-text placeholder"}>{leg.translation}</span><Chips terms={leg.glossary_terms}/></div><span className="leg-state">{activeAirLanguage === leg.lang ? "PLAYING" : airSequencePlaying ? "QUEUED" : bc ? "READY" : "PENDING"}</span><button className="leg-play" type="button" onClick={() => void replayAirLeg(leg)} disabled={busy || !leg.wav.length} aria-label={`Speak ${LANGUAGE_NAMES[leg.lang]} translation`} title={`Play ${LANGUAGE_NAMES[leg.lang]} audio`}>{activeAirLanguage === leg.lang ? "Playing…" : "▶ Speak"}</button></div>)}</div></>}
      </section>
      {inputMode === "voice" && mode === "hand" && result && <Hud te={result.telemetry} confidence={result.confidence} playbackStartMs={playbackStartMs}/>}<footer className="workspace-footer">VEGA / OFFLINE SPEECH SYSTEM <span>HAND + AIR · ONE ENGINE</span></footer>
    </main>
  </div>;
}
