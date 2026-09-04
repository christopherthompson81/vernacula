import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MODELS, NUM_STEPS } from "./inference/config.ts";
import { DONOR_NAMES, LANGUAGE_BY_CODE, voiceLangOf, type LanguageOption } from "./inference/languages.ts";
import { LanguagePicker } from "./LanguagePicker.tsx";
import { phonemize } from "./inference/phonemizer.ts";
import { OmniVoice, parseJsonc, voiceFor, voicesFor, type Voice } from "./inference/omnivoice.ts";
import type { WordTiming } from "./inference/alignment.ts";
import { encodeWav } from "./inference/audioPost.ts";
import { cacheUnavailableReason, fetchModel } from "./inference/modelCache.ts";
import type { Progress, Token } from "./types.ts";

/** Pair each orthographic word with its timed IPA word, positionally.
 *
 * ⚠ Positional, and therefore approximate. The phonemizer is not word-for-word: normalization
 * expands numbers and abbreviations ("20°C" becomes several words), so the two sequences can
 * differ in length. Showing the pairing only when the counts agree is honest. Punctuation tokens
 * in the IPA stream are timed too (they are pauses) but have no orthographic partner. */
const PUNCT_TOKEN = /^[.,!?;:]$/u;

function pairTokens(text: string, words: WordTiming[]): Token[] | null {
  const ortho = text.trim().split(/\s+/u).filter(Boolean);
  const ipaWords = words.filter((w) => !PUNCT_TOKEN.test(w.ipa));
  if (ortho.length !== ipaWords.length || ortho.length === 0) return null;
  return ortho.map((t, i) => ({ text: t, ipa: ipaWords[i].ipa, start: ipaWords[i].start, end: ipaWords[i].end }));
}

/** Index of the word under the playhead, or -1 in a pause: nothing is being said, so nothing is lit. */
function activeIndex(tokens: { start?: number; end?: number }[], t: number): number {
  for (let k = 0; k < tokens.length; k++) {
    const { start, end } = tokens[k];
    if (start !== undefined && end !== undefined && start <= t && t < end) return k;
  }
  return -1;
}

export default function App() {
  const [lang, setLang] = useState("en");
  const [text, setText] = useState(LANGUAGE_BY_CODE.get("en")!.sample ?? "");
  const [progress, setProgress] = useState<Progress>({ stage: "idle" });
  const [ipa, setIpa] = useState("");
  const [tokens, setTokens] = useState<Token[] | null>(null);
  /** Every timed IPA token, punctuation included — the fallback highlight when pairing fails. */
  const [words, setWords] = useState<WordTiming[] | null>(null);
  const [active, setActive] = useState(-1);
  const audioEl = useRef<HTMLAudioElement | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [stats, setStats] = useState<string>("");
  const [ep, setEp] = useState<string>("");
  const [voiceId, setVoiceId] = useState<string | undefined>();
  /**
   * The whole voice catalogue, fetched on mount.
   *
   * ⚠ NOT taken from the loaded engine, which is what it used to be. The engine only exists after
   * the 472 MB model has downloaded, so the voice picker stayed hidden through the entire first
   * session — you could not see which voices a language had until after you had already generated
   * with one. Choosing a voice needs the metadata file (labels and languages), not the model.
   */
  const [allVoices, setAllVoices] = useState<Voice[]>([]);
  const engine = useRef<OmniVoice | null>(null);
  const voice = useRef<Voice | null>(null);

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }, [audioUrl]);

  useEffect(() => {
    let live = true;
    fetch(MODELS.voicesUrl)
      .then((r) => r.text())
      .then((t) => { if (live) setAllVoices(parseJsonc<Voice[]>(t)); })
      .catch(() => { /* the picker just stays hidden; generation still works */ });
    return () => { live = false; };
  }, []);

  /** Voices offered for the selected language — its own, or its donor's. */
  const langVoices = useMemo(() => voicesFor(allVoices, voiceLangOf(lang)), [allVoices, lang]);

  const langOpt = useMemo(() => LANGUAGE_BY_CODE.get(lang) as LanguageOption, [lang]);
  const trained = langOpt.trained === true;

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
    setEp(ov.backend.ep);
    return ov;
  }, [lang]);

  const generate = useCallback(async () => {
    if (!text.trim()) return;
    try {
      setProgress({ stage: "phonemizing" });
      const out = (await phonemize(text, lang, (d) => setProgress({ stage: "phonemizing", detail: d }))).trim();
      setIpa(out);
      setTokens(null); setWords(null); setActive(-1);

      const ov = await ensureEngine();
      setProgress({ stage: "generating", fraction: 0 });
      // ⚠ Pick the voice for THIS language every time, not once at load: the reference carries the
      // speaker's accent, so reusing an English voice for German is audibly wrong.
      const v = voiceFor(ov.voices, voiceLangOf(lang), voiceId);
      voice.current = v;
      const r = await ov.synthesize(out, v, { numStep: NUM_STEPS },
        (step, total) => setProgress({ stage: "generating", fraction: step / total, detail: `step ${step}/${total}` }));

      setWords(r.words);
      setTokens(pairTokens(text, r.words));
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(encodeWav(r.audio, r.sampleRate)));
      setStats(`${(r.audio.length / r.sampleRate).toFixed(1)}s audio in ${(r.generateMs / 1000).toFixed(1)}s`
        + ` · ${r.targetTokens} tokens · ${NUM_STEPS} steps · voice ${v.label}`);
      setProgress({ stage: "ready" });
    } catch (e) {
      setProgress({ stage: "error", detail: e instanceof Error ? e.message : String(e) });
    }
  }, [text, lang, voiceId, ensureEngine, audioUrl]);

  // Karaoke: while the audio plays, follow the playhead with requestAnimationFrame rather than
  // `timeupdate`, which fires only ~4x/s and visibly lags short words.
  useEffect(() => {
    const el = audioEl.current;
    if (!el || !words) return;
    const timed = tokens ?? words;
    let raf = 0;
    const tick = () => { setActive(activeIndex(timed, el.currentTime)); raf = requestAnimationFrame(tick); };
    const start = () => { cancelAnimationFrame(raf); tick(); };
    const stop = () => { cancelAnimationFrame(raf); setActive(el.ended ? -1 : activeIndex(timed, el.currentTime)); };
    // `timeupdate` as well: rAF is suspended in an occluded tab, and the coarse ticks keep the
    // highlight alive there (and made the headless-ish REPL verification possible at all).
    const coarse = () => setActive(activeIndex(timed, el.currentTime));
    el.addEventListener("play", start); el.addEventListener("pause", stop); el.addEventListener("ended", stop);
    el.addEventListener("seeked", stop); el.addEventListener("timeupdate", coarse);
    return () => { cancelAnimationFrame(raf); el.removeEventListener("play", start); el.removeEventListener("pause", stop);
                   el.removeEventListener("ended", stop); el.removeEventListener("seeked", stop);
                   el.removeEventListener("timeupdate", coarse); };
  }, [words, tokens, audioUrl]);

  const busy = progress.stage === "loading-models" || progress.stage === "phonemizing" || progress.stage === "generating";

  return (
    <main>
      <header>
        {/* Decorative, so alt="" — the h1 beside it already names the page, and a screen reader
            announcing "bat with waveform wings" before the title is noise, not information. */}
        <img className="mark" src="/vern-waveform.png" alt="" width={340} height={133} />
        <h1>vernacula-tts</h1>
        <p className="sub">
          Text → canonical IPA → speech, entirely in your browser. Nothing is uploaded.
        </p>
      </header>

      <section className="controls">
        <label className="lang">
          Language
          <LanguagePicker value={lang} disabled={busy} onChange={(c) => {
            setLang(c);
            setText(LANGUAGE_BY_CODE.get(c)?.sample ?? "");
            setIpa(""); setTokens(null); setWords(null); setActive(-1); setStats("");
            // A voice belongs to a language; carrying one across would reintroduce the accent bleed
            // that per-language references exist to remove.
            setVoiceId(undefined);
          }} />
        </label>
        {langVoices.length > 1 && (
          <label>
            Voice
            <select value={voiceId ?? langVoices[0].id} disabled={busy}
                    onChange={(e) => setVoiceId(e.target.value)}>
              {langVoices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.sex ? `${v.sex === "F" ? "♀" : "♂"} ` : ""}{v.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <button onClick={generate} disabled={busy || !text.trim()}>
          {busy ? "Working…" : "Generate"}
        </button>
      </section>

      <textarea value={text} disabled={busy} rows={3}
                onChange={(e) => setText(e.target.value)}
                placeholder={langOpt.sample ? "Type something to say…"
                             : `Type something in ${langOpt.name} — no example sentence is bundled for it yet`} />

      {(!trained || langOpt.voice) && (
        <p className="note">
          {!trained && (
            <>
              <strong>{langOpt.name}</strong> was not in the fine-tune corpus. It still renders — the
              model reads IPA and draws on phones it already holds — but the result is extrapolated,
              prosody most of all.{" "}
            </>
          )}
          {langOpt.voice && (
            <>
              There is no native reference voice for it yet, so it is read by a{" "}
              <strong>{LANGUAGE_BY_CODE.get(langOpt.voice)?.name ?? DONOR_NAMES[langOpt.voice] ?? langOpt.voice}</strong> speaker:
              voice cloning copies accent along with timbre, so expect one.
            </>
          )}
        </p>
      )}

      {busy && (
        <div className="progress">
          <div className="bar"><div style={{ width: `${(progress.fraction ?? 0) * 100}%` }} /></div>
          <span>{progress.stage}{progress.detail ? ` — ${progress.detail}` : ""}</span>
        </div>
      )}

      {progress.stage === "error" && <p className="error">{progress.detail}</p>}

      {/* Shown once, not on error: the page works here, it just cannot KEEP the model. Saying so up
          front beats letting someone re-download 472 MB twice before wondering why. */}
      {cacheUnavailableReason && <p className="notice">{cacheUnavailableReason}</p>}

      {ipa && (
        <section className="result">
          <h2>IPA</h2>
          {tokens
            ? <p className="pairs">{tokens.map((t, i) => (
                <span className={"pair" + (i === active ? " active" : "")} key={i}
                      onClick={() => { if (audioEl.current && t.start !== undefined) audioEl.current.currentTime = t.start; }}>
                  <span className="ortho">{t.text}</span><span className="ipa">{t.ipa}</span>
                </span>
              ))}</p>
            : words
            ? <p className="ipa-flat">{words.map((w, i) => (
                <span className={"word" + (i === active ? " active" : "")} key={i}
                      onClick={() => { if (audioEl.current) audioEl.current.currentTime = w.start; }}>{w.ipa}{" "}</span>
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
          <audio controls src={audioUrl} ref={audioEl} />
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
          {/* ⚠ TWO UPSTREAMS, TWO LICENCES, and saying only the strict one would be wrong. The
              TRANSFORMER weights are CC-BY-NC ("due to constraints from its training data").
              The CODEC is not: OmniVoice ships Boson's tokenizer byte-identical (same sha256), so
              it carries Boson's own community licence, which permits commercial use below 100k
              annual active users. Neither says anything about the phonemizer or this demo's code. */}
          {" "}The transformer weights are <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC-BY-NC</a>;
          the <a href="https://huggingface.co/bosonai/higgs-audio-v2-tokenizer">Higgs codec</a> is
          Boson's, under its own licence.
        </p>
      </footer>
    </main>
  );
}
