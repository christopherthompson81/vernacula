"""Stream-ingest FLEURS -> Higgs codec tokens + IPA, discarding waveforms.

Per language:
  1. download train.tar.gz (transient, deleted after)
  2. iterate wav members in memory, resample 16k->24k, run higgs_encoder.onnx
  3. store int16 audio_codes (npz, keyed by utterance id) + manifest.jsonl
     (id, lang, ipa, gender, dur_s, n_frames)
  4. delete the tar

IPA comes PER WAV from the alignment DB (work/asr_align/align.sqlite), falling back to the
per-SENTENCE phonemizer output for rows the DB does not have.

⚠ PER WAV, NOT PER SENTENCE, AND THAT IS THE WHOLE REASON THIS CHANGED. `read_text` records what a
reader ACTUALLY said on ONE recording — an English numeral in a Hausa sentence, a Portuguese one in
Umbundu, the Bengali year form — and its IPA is re-derived from that. A per-sentence lookup gives every
take of a sentence the same string and cannot represent it, so the corrections were invisible to
training. 165 rows carry a hand `read_text` today; the mechanism only pays off if the corpus reads it.

⚠ AND THE DB IS THE VERNACULA ENGINE'S OUTPUT, not `work/phonemized` (espeak), which this script used
to read. Two IPA sources for one corpus is the drift this pipeline has been bitten by before.

Usage:
  ingest_fleurs.py [--provider cpu|cuda] [--validate N] fleurs_code [fleurs_code ...]
"""
import argparse, io, json, os, sys, tarfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filter import EXCLUDE_STATUSES, EXCLUDE_UNLESS_HAND_READ_TEXT  # noqa: E402
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
from huggingface_hub import hf_hub_download

ROOT = "/mnt/data/omnivoice_ipa"
ONNX = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx"
TSV = f"{ROOT}/corpus/fleurs_transcripts/data"
PHON = f"{ROOT}/work/phonemized_vernacula"
ALIGN_DB = f"{ROOT}/work/asr_align/align.sqlite"
OUT = f"{ROOT}/corpus/tokens"
TRANSIENT = f"{ROOT}/corpus/_transient"
# Persistent audio cache (on /mnt/data, ~483 GB free; all 24 FLEURS train tars ~50 GB).
# Kept by default so a re-ingest (e.g. the full multi-speaker rebuild) downloads once.
AUDIO_CACHE = f"{ROOT}/corpus/audio_cache"
SR_IN, SR_OUT = 16000, 24000
# Duration filter, matching the official omnivoice/scripts/extract_audio_tokens.py:
# it does NOT chunk long audio — it filters min_len<=dur<=max_len and drops the rest
# (its help examples are min 2.0s / max 15.0s). We skip the same way; the upper bound
# also guards the semantic encoder's quadratic attention from OOM on outliers.
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
os.makedirs(OUT, exist_ok=True)
os.makedirs(TRANSIENT, exist_ok=True)


def session(provider):
    # CUDA MUST disable TF32: the VQ does a hard nearest-codebook argmax, so TF32
    # matmul drift flips code indices (codes diverge entirely from fp32/CPU). With
    # use_tf32=0, CUDA codes are bitwise-identical to CPU at ~60x the speed.
    # cudnn_conv_algo_search=DEFAULT: with dynamic input lengths, EXHAUSTIVE re-tunes
    # per shape (slow + "Conv running in Fallback mode" spam); DEFAULT is fastest here.
    # arena_extend_strategy=kSameAsRequested: EVERY utterance is a different length, so the default
    # kNextPowerOfTwo arena grows in doubling blocks and FRAGMENTS. sw_ke and yo_ng died partway
    # through with "Failed to allocate ... size 62668800" while 22.6 GB of 24.5 GB was free — the
    # message says OOM and means fragmentation. It is not the audio: 884 utterances succeeded and
    # then a 10.2s clip failed, and ne_np has MORE utterances (3332) and a 256s maximum and survived,
    # purely on the luck of its shape ordering. With kSameAsRequested, sw_ke ran 2998/2998 clean.
    provs = {"cpu": ["CPUExecutionProvider"],
             "cuda": [("CUDAExecutionProvider",
                       {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
                        "arena_extend_strategy": "kSameAsRequested"}),
                      "CPUExecutionProvider"]}[provider]
    return ort.InferenceSession(f"{ONNX}/higgs_encoder.onnx", providers=provs)


def id_to_ipa(lang):
    """{sentence_id: ipa} from the id-keyed phonemizer output (byid/<lang>.tsv). The FALLBACK."""
    m = {}
    path = f"{PHON}/byid/{lang}.tsv"
    if not os.path.exists(path):
        return m
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                m[parts[0]] = parts[1]
    return m


def wav_to_ipa(lang):
    """({wav_basename: (ipa, read_text_src, status)}, {every wav the DB knows}) — the PREFERRED source.

    ⚠ THIS NO LONGER DROPS UNUSABLE ROWS, AND THAT IS THE POINT. It used to omit `defective_audio` and
    unrepaired `reader_divergence` so a caller "could not forget" — which fused a REVISABLE JUDGEMENT to
    an artifact that is otherwise a pure function of the audio. Every revision then cost a GPU
    re-encode, and worse, an old judgement stayed frozen in whichever languages were encoded before it
    changed: cy_gb and es_419 were ingested 2026-07-01 and carried 970 `defective_audio` rows in their
    manifests for two months while the other 96 languages were pruned. Nothing in any log said so.

    Now every row is encoded and carries its `status`; `corpus_filter.load_manifest` applies the policy
    at load. Changing our mind is a re-label (`build_manifest.py`, seconds) instead of a re-ingest.
    """
    if not os.path.exists(ALIGN_DB):
        return {}, set()
    import sqlite3
    db = sqlite3.connect(f"file:{ALIGN_DB}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT wav, ipa, COALESCE(read_text_src,''), COALESCE(status,'') FROM utt "
        "WHERE lang=? AND ipa IS NOT NULL AND TRIM(ipa) <> ''",
        (lang,),
    ).fetchall()
    # ⚠ EVERY wav THE DB KNOWS, usable or not — WITHOUT THIS, "absent" READS AS "EXCLUDED". The align
    # DB does not necessarily cover a whole language: as_in has 1,120 rows against 2,812 in train.tsv,
    # byid and the audio tar, and NONE of the missing ones is excluded by status — the alignment pass
    # simply never reached them. Treating absence as exclusion silently dropped 60% of Assamese from
    # training and logged it as "1692 excluded", which reads as deliberate.
    known = {w for (w,) in db.execute("SELECT wav FROM utt WHERE lang=?", (lang,))}
    db.close()
    return {w: (i, src, st) for w, i, src, st in rows}, known


def fname_to_meta(lang):
    """{wav_basename: (id, gender)} from train.tsv (col1=file, col0=id, col6=gender).
    Raw tab-split (FLEURS tsv is not quoted CSV; csv.reader mis-merges quote chars)."""
    m = {}
    with open(f"{TSV}/{lang}/train.tsv", encoding="utf-8") as f:
        for line in f:
            r = line.rstrip("\n").split("\t")
            if len(r) >= 7:
                m[os.path.basename(r[1])] = (r[0], r[6])
    return m


def ingest(lang, sess, validate=0, keep_audio=True):
    t0 = time.time()
    ipa_map = id_to_ipa(lang)          # per SENTENCE, fallback
    by_wav, known = wav_to_ipa(lang)   # per WAV, preferred; already excludes unusable rows
    n_db = n_fallback = n_flagged = 0
    meta = fname_to_meta(lang)
    # keep_audio: download into the persistent cache (re-runs skip the download since
    # hf_hub_download sees the file already present). Else use the transient dir + delete.
    dl_dir = AUDIO_CACHE if keep_audio else TRANSIENT
    os.makedirs(dl_dir, exist_ok=True)
    tar_path = hf_hub_download("google/fleurs", f"data/{lang}/audio/train.tar.gz",
                               repo_type="dataset", local_dir=dl_dir)
    codes_out, manifest = {}, []
    in_name = sess.get_inputs()[0].name
    n_ok = n_skip = 0
    val_pairs = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue
            base = os.path.basename(member.name)
            if base not in meta:
                n_skip += 1
                continue
            sentence_id, gender = meta[base]
            hit = by_wav.get(base)
            if hit is not None:
                ipa, rt_src, status = hit
                n_db += 1
                if status in EXCLUDE_STATUSES or (
                        status in EXCLUDE_UNLESS_HAND_READ_TEXT and rt_src != "hand"):
                    # ⚠ FLAGGED, NOT SKIPPED. Encoded and labelled; the loader drops it. See wav_to_ipa.
                    n_flagged += 1
            else:
                ipa, rt_src, status = ipa_map.get(sentence_id), "none", ""
                n_fallback += 1
            if not ipa:
                n_skip += 1
                continue
            # FLEURS 'id' (col0) is a per-SENTENCE id SHARED across speaker recordings —
            # keying codes by it makes later speakers overwrite earlier ones. Key by the
            # unique wav basename instead so every recording is retained; keep sentence_id
            # for reference (IPA/text is per-sentence, so it's looked up by sentence_id).
            uid = base[:-len(".wav")]
            f = tar.extractfile(member)
            wav, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            # duration filter (see MIN/MAX_SECONDS): drop out-of-range like the
            # official extractor; upper bound also guards quadratic-attention OOM.
            if not (MIN_SECONDS <= len(wav) / sr <= MAX_SECONDS):
                n_skip += 1
                continue
            if sr != SR_OUT:
                # soxr_hq: high-quality and ~10x faster than librosa's default kaiser_best
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR_OUT, res_type="soxr_hq")
            # encoder requires input length a multiple of 960 (its two internal
            # encoder paths otherwise disagree by one frame at the Concat).
            pad = (-len(wav)) % 960
            x = np.pad(wav, (0, pad)).reshape(1, 1, -1).astype(np.float32)
            codes = sess.run(["audio_codes"], {in_name: x})[0][0]  # [8, T]
            codes_out[uid] = codes.astype(np.int16)
            manifest.append(dict(id=uid, sentence_id=sentence_id, lang=lang, ipa=ipa,
                                 gender=gender, dur_s=round(len(wav) / SR_OUT, 2),
                                 n_frames=int(codes.shape[-1]),
                                 # ⚠ so a consumer can tell a reader-corrected pair from an ordinary
                                 # one; "hand" means the transcript is what was SAID, not the script.
                                 ipa_src=rt_src,
                                 # ⚠ THE LABEL TRAVELS WITH THE ROW so policy can change without a
                                 # re-encode. Empty means the DB has no verdict, NOT that it is clean.
                                 status=status))
            n_ok += 1
            if validate and len(val_pairs) < validate:
                val_pairs.append((uid, wav, codes))
    np.savez_compressed(f"{OUT}/codes_{lang}.npz", **{k: v for k, v in codes_out.items()})
    with open(f"{OUT}/manifest_{lang}.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not keep_audio:
        os.remove(tar_path)
    sz = os.path.getsize(f"{OUT}/codes_{lang}.npz") / 1e6
    dt = time.time() - t0
    # ⚠ Name the IPA source in the log. A silent fallback to the per-sentence file is exactly how the
    # read_text corrections would go missing again, and it looks identical to success in the counts.
    src = (f"ipa: {n_db} from align-db"
           + (f", {n_fallback} fallback-by-sentence" if n_fallback else "")
           + (f", {n_flagged} flagged (encoded; filtered at load)" if n_flagged else ""))
    print(f"{lang}: {n_ok} utts, {n_skip} skipped, codes {sz:.1f} MB, {dt:.0f}s "
          f"({n_ok/dt:.1f} utt/s)  [{src}]")
    return val_pairs


def validate_roundtrip(val_pairs, lang, provider):
    """Decode codes back to audio, log-mel spectral distance vs input (NOT SNR:
    GAN codec changes phase/fine detail; spectral envelope is the right check)."""
    if not val_pairs:
        return
    dec = ort.InferenceSession(f"{ONNX}/higgs_decoder.onnx",
                               providers=["CPUExecutionProvider"])
    vdir = f"{ROOT}/work/codec_validation"
    os.makedirs(vdir, exist_ok=True)
    print(f"  -- codec round-trip ({lang}) --")
    for uid, wav, codes in val_pairs:
        rec = dec.run(["audio_values"], {"audio_codes": codes[None].astype(np.int64)})[0]
        rec = np.asarray(rec).reshape(-1)
        n = min(len(wav), len(rec))
        S1 = librosa.power_to_db(librosa.feature.melspectrogram(
            y=wav[:n], sr=SR_OUT, n_mels=80))
        S2 = librosa.power_to_db(librosa.feature.melspectrogram(
            y=rec[:n], sr=SR_OUT, n_mels=80))
        lsd = float(np.sqrt(np.mean((S1 - S2) ** 2)))
        sf.write(f"{vdir}/{lang}_{uid}_orig.wav", wav[:n], SR_OUT)
        sf.write(f"{vdir}/{lang}_{uid}_recon.wav", rec[:n], SR_OUT)
        print(f"     {uid}: log-mel-SD = {lsd:.2f} dB  (orig/recon wav -> {vdir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--validate", type=int, default=0,
                    help="decode N utts back and report log-mel spectral distance")
    ap.add_argument("--no-keep-audio", action="store_true",
                    help="delete each tar after processing (default: keep in audio_cache/ "
                         "on /mnt/data so a re-ingest doesn't re-download)")
    ap.add_argument("langs", nargs="+")
    a = ap.parse_args()
    sess = session(a.provider)
    print(f"provider={a.provider} ({sess.get_providers()[0]})")
    for lang in a.langs:
        vp = ingest(lang, sess, a.validate, keep_audio=not a.no_keep_audio)
        if a.validate:
            validate_roundtrip(vp, lang, a.provider)


if __name__ == "__main__":
    main()
