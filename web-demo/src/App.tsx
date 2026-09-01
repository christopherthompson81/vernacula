import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LANGUAGES, MODELS, NUM_STEPS } from "./inference/config.ts";
import { phonemize } from "./inference/phonemizer.ts";
import { OmniVoice, type Voice } from "./inference/omnivoice.ts";
import { encodeWav } from "./inference/audioPost.ts";
import { fetchModel } from "./inference/modelCache.ts";
import type { Progress, Token } from "./types.ts";

/** Pair each orthographic word with its IPA, positionally.
 *
 * ⚠ Positional, and therefore approximate. The phonemizer is not word-for-word: normalization
 * expands numbers and abbreviations ("20°C" becomes several words), so the two sequences can
 * differ in length. Showing the pairing only when the counts agree is honest; a forced alignment
 * is what karaoke highlighting will need anyway. */
function pairTokens(text: string, ipa: string): Token[] | null {
  const words = text.trim().split(/\s+/u).filter(Boolean);
  const ipaWords = ipa.trim().split(/\s+/u).filter((w) => !/^[.,!?;:]$/u.test(w));
  if (words.length !== ipaWords.length || words.length === 0) return null;
  return words.map((w, i) => ({ text: w, ipa: ipaWords[i] }));
}

export default function App() {
  const [lang, setLang] = useState("en");
  const [text, setText] = useState(LANGUAGES.find((l) => l.code === "en")!.sample);
  const [progress, setProgress] = useState<Progress>({ stage: "idle" });
  const [ipa, setIpa] = useState("");
  const [tokens, setTokens] = useState<Token[] | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<string>("");
  const [ep, setEp] = useState<string>("");
  const engine = useRef<OmniVoice | null>(null);
  const voice = useRef<Voice | null>(null);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  const langOpt = useMemo(() => LANGUAGES.find((l) => l.code === lang)!, [lang]);
  const trained = langOpt.trained !== false;

  const ensureEngine = useCallback(async () => {
    if (engine.current) return engine.current;
    setProgress({ stage: "loading-models", detail: "starting" });
    const ov = await OmniVoice.load({
      ...MODELS,
      fetchBytes: (url, label) =>
        fetchModel(url, (p) =>
          setProgress({
            stage: "loading-models",
            fraction: p.total ? p.loaded / p.total : undefined,
            detail: `${label} ${(p.loaded / 1e6).toFixed(0)}/${(p.total / 1e6).toFixed(0)} MB${p.cached ? " (cached)" : ""}`,
          })),
      onProgress: (detail) => setProgress({ stage: "loading-models", detail }),
    });
    engine.current = ov;
    voice.current = ov.voices[0];
    setEp(ov.backend.ep);
    return ov;
  }, []);

  const generate = useCallback(async () => {
    if (!text.trim()) return;
    try {
      setProgress({ stage: "phonemizing" });
      const out = (await phonemize(text, lang, (d) => setProgress({ stage: "phonemizing", detail: d }))).trim();
      setIpa(out);
      setTokens(pairTokens(text, out));

      const ov = await ensureEngine();
      setProgress({ stage: "generating", fraction: 0 });
      const r = await ov.synthesize(out, voice.current!, { numStep: NUM_STEPS },
        (step, total) => setProgress({ stage: "generating", fraction: step / total, detail: `step ${step}/${total}` }));

      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(encodeWav(r.audio, r.sampleRate)));
      setStats(`${(r.audio.length / r.sampleRate).toFixed(1)}s audio in ${(r.generateMs / 1000).toFixed(1)}s`
        + ` · ${r.targetTokens} tokens · ${NUM_STEPS} steps`);
      setProgress({ stage: "ready" });
    } catch (e) {
      setProgress({ stage: "error", detail: e instanceof Error ? e.message : String(e) });
    }
  }, [text, lang, ensureEngine, audioUrl]);

  const busy = progress.stage === "loading-models" || progress.stage === "phonemizing" || progress.stage === "generating";

  return (
    <main>
      <header>
        <h1>vernacula-tts</h1>
        <p className="sub">
          Text → canonical IPA → speech, entirely in your browser. Nothing is uploaded.
        </p>
      </header>

      <section className="controls">
        <label>
          Language
          <select value={lang} disabled={busy}
                  onChange={(e) => {
                    const c = e.target.value;
                    setLang(c);
                    setText(LANGUAGES.find((l) => l.code === c)!.sample);
                    setIpa(""); setTokens(null); setStats("");
                  }}>
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.name}{l.trained === false ? " (untrained)" : ""}</option>
            ))}
          </select>
        </label>
        <button onClick={generate} disabled={busy || !text.trim()}>
          {busy ? "Working…" : "Generate"}
        </button>
      </section>

      <textarea value={text} disabled={busy} rows={3}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type something to say…" />

      {!trained && (
        <p className="note">
          <strong>{langOpt.name}</strong> was not in the fine-tune corpus. It still renders — the model
          reads IPA and draws on phones it already holds — but the result is extrapolated, prosody most of all.
        </p>
      )}

      {busy && (
        <div className="progress">
          <div className="bar"><div style={{ width: `${(progress.fraction ?? 0) * 100}%` }} /></div>
          <span>{progress.stage}{progress.detail ? ` — ${progress.detail}` : ""}</span>
        </div>
      )}

      {progress.stage === "error" && <p className="error">{progress.detail}</p>}

      {ipa && (
        <section className="result">
          <h2>IPA</h2>
          {tokens
            ? <p className="pairs">{tokens.map((t, i) => (
                <span className="pair" key={i}><span className="ortho">{t.text}</span><span className="ipa">{t.ipa}</span></span>
              ))}</p>
            : <p className="ipa-flat">{ipa}</p>}
          {!tokens && ipa && (
            <p className="note small">
              Word pairing is hidden here: normalization changed the word count (numbers and
              abbreviations expand), so a positional pairing would line the wrong words up.
            </p>
          )}
        </section>
      )}

      {audioUrl && (
        <section className="result">
          <h2>Audio</h2>
          <audio controls src={audioUrl} />
          <p className="stats">{stats}{ep ? ` · ${ep}` : ""}</p>
        </section>
      )}

      <footer>
        {ep === "wasm" && (
          <p className="note small">
            Running on WASM. WebGPU is roughly 7× faster where available — on this machine 177 ms
            per forward pass against 1295 ms.
          </p>
        )}
        <p className="small">
          Phonemes from <a href="https://github.com/christopherthompson81/vernacula-phonemizer">vernacula-phonemizer</a>;
          speech from an IPA fine-tune of <a href="https://huggingface.co/k2-fsa/OmniVoice">k2-fsa/OmniVoice</a>,
          quantized to <a href="https://huggingface.co/christopherthompson81/omnivoice-ipa-onnx">472 MB</a> and
          cached in your browser after the first load.
        </p>
      </footer>
    </main>
  );
}
