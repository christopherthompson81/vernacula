#!/usr/bin/env python3
"""
Formal content-preservation metrics for ASR output against reference
transcripts, without using an LLM. Intended as the validation harness for
the per-domain KenLM A/Bs we run on medical/legal/etc. audio.

Metrics (all deterministic, reproducible, CPU-only):

  1. WER / CER                  jiwer. Verbatim-strict metrics kept for context.
  2. Content-word WER           WER after stripping a small stopword list.
  3. Entity recall / precision  Via scispaCy's en_ner_bc5cdr_md (chemicals +
                                diseases). For each sample, tag entities in
                                ref and hyp, compute set precision/recall/F1.
                                This is the standard formal metric for
                                medical-ASR evaluation and is what we care
                                about ("did the transcript capture the
                                medical content?").
  4. Drug-focused recall        Same idea, filtered to CHEMICAL labels.
  5. ROUGE-L F1                 Longest-common-subsequence F1; tolerates
                                reordering and minor surface edits in a way
                                WER punishes.

Usage:

  python evaluate.py \
      --ref-dir /path/to/references \
      --hyp-dir /path/to/hypothesis-outputs \
      --modes greedy,general,medical
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import jiwer
    import spacy
except ImportError as e:
    print(f"install: pip install jiwer scispacy en-ner-bc5cdr-md ({e})", file=sys.stderr)
    sys.exit(1)


STOPWORDS = {
    "a","the","to","of","in","on","at","is","are","was","were","be","been","being",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
    "his","our","their","its","this","that","these","those","for","by","with","as",
    "but","and","or","so","if","because","do","does","did","have","has","had",
    "will","would","should","could","can","may","might","ok","okay","yeah","yes",
    "no","not","very","just","about","um","uh","mm","mmm","ohh","oh","ah","eh","hmm",
    "well","really","actually","like","kind","sort","thing","stuff","some","any",
    "all","there","here","go","going","get","got","im","ive","youve","theyre",
    "didnt","dont",
}

WORD_TRANSFORMS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

CHAR_TRANSFORMS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def extract_md_text(md_path: Path) -> str:
    """Pull clean text body out of the Vernacula transcription markdown,
    dropping BOM, speaker headers, and empty lines."""
    text = md_path.read_text().lstrip("\ufeff")
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return " ".join(out)


def normalize_ref(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufeff", "")).strip()


def content_word_text(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return " ".join(w for w in words if w not in STOPWORDS)


def lcs_length(a: list[str], b: list[str]) -> int:
    """Classic O(mn) LCS length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l_f1(ref_words: list[str], hyp_words: list[str]) -> float:
    if not ref_words or not hyp_words:
        return 0.0
    lcs = lcs_length(ref_words, hyp_words)
    p = lcs / len(hyp_words)
    r = lcs / len(ref_words)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def entity_set(nlp, text: str, labels: set[str] | None = None) -> set[str]:
    """Return the lowercased set of entity surface forms matching the labels."""
    doc = nlp(text)
    out: set[str] = set()
    for ent in doc.ents:
        if labels is None or ent.label_ in labels:
            surface = ent.text.lower().strip()
            if surface:
                out.add(surface)
    return out


def prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def evaluate_one(nlp, ref_text: str, hyp_text: str) -> dict:
    # WER / CER use jiwer's transformed views so tokenisation + case are uniform.
    out = {}
    wer_out = jiwer.process_words(ref_text, hyp_text,
                                  reference_transform=WORD_TRANSFORMS,
                                  hypothesis_transform=WORD_TRANSFORMS)
    out["wer"] = wer_out.wer
    out["subs"] = wer_out.substitutions
    out["dels"] = wer_out.deletions
    out["ins"]  = wer_out.insertions
    out["cer"] = jiwer.cer(ref_text, hyp_text,
                           reference_transform=CHAR_TRANSFORMS,
                           hypothesis_transform=CHAR_TRANSFORMS)
    # Content-word WER
    cw_ref = content_word_text(ref_text)
    cw_hyp = content_word_text(hyp_text)
    out["cw_wer"] = jiwer.wer(cw_ref, cw_hyp)
    # ROUGE-L F1 over content words (gives a score of similarity unaffected
    # by disfluency density)
    out["rouge_l"] = rouge_l_f1(cw_ref.split(), cw_hyp.split())
    # Entity sets
    ref_ents_all  = entity_set(nlp, ref_text)
    hyp_ents_all  = entity_set(nlp, hyp_text)
    ref_chems     = entity_set(nlp, ref_text, {"CHEMICAL"})
    hyp_chems     = entity_set(nlp, hyp_text, {"CHEMICAL"})
    ref_diseases  = entity_set(nlp, ref_text, {"DISEASE"})
    hyp_diseases  = entity_set(nlp, hyp_text, {"DISEASE"})
    out["ents_all"] = prf(ref_ents_all, hyp_ents_all)
    out["chems"]    = prf(ref_chems,    hyp_chems)
    out["diseases"] = prf(ref_diseases, hyp_diseases)
    out["ent_counts"] = (len(ref_ents_all), len(ref_chems), len(ref_diseases))
    return out


def fmt_prf(triple: tuple[float, float, float]) -> str:
    p, r, f = triple
    return f"P{p*100:5.1f}/R{r*100:5.1f}/F{f*100:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir",   type=Path, required=True,
                    help="Directory of reference transcripts (.txt files).")
    ap.add_argument("--hyp-dir",   type=Path, required=True,
                    help="Directory of hypothesis markdown files. Looks for "
                         "<ref-stem>-<mode>.md.")
    ap.add_argument("--modes",     default="greedy,general,medical",
                    help="Comma-separated list of mode suffixes to compare.")
    ap.add_argument("--spacy-model", default="en_ner_bc5cdr_md",
                    help="spaCy NER model id (default en_ner_bc5cdr_md).")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"loading {args.spacy_model}...", file=sys.stderr)
    nlp = spacy.load(args.spacy_model)

    ref_files = sorted(args.ref_dir.glob("*.txt"))
    if not ref_files:
        print(f"no .txt references in {args.ref_dir}", file=sys.stderr)
        return 1

    print(f"\n{'file':<24}{'mode':<10}  WER   CER  cw-WER  ROUGE-L   ALL-ents           CHEMs              DISEASEs")
    print("-" * 128)

    # Also collect per-mode aggregate (micro-avg across all ref files)
    agg = defaultdict(lambda: {"wer_num":0, "wer_den":0, "cer_num":0.0, "cer_den":0,
                               "rouge_sum":0.0, "n":0,
                               "tp_all":0, "p_den_all":0, "r_den_all":0,
                               "tp_chem":0, "p_den_chem":0, "r_den_chem":0,
                               "tp_dis":0, "p_den_dis":0, "r_den_dis":0})

    for ref_path in ref_files:
        ref_text = normalize_ref(ref_path.read_text())
        for m in modes:
            hp = args.hyp_dir / f"{ref_path.stem}-{m}.md"
            if not hp.exists():
                hp = args.hyp_dir / f"cons{ref_path.stem[-2:]}-{m}.md"  # fallback for older naming
                if not hp.exists(): continue
            hyp_text = extract_md_text(hp)
            r = evaluate_one(nlp, ref_text, hyp_text)

            print(f"{ref_path.stem[-16:]:<24}{m:<10}  "
                  f"{r['wer']*100:5.2f}% {r['cer']*100:5.2f}% {r['cw_wer']*100:6.2f}% "
                  f"  {r['rouge_l']*100:6.2f}%   "
                  f"{fmt_prf(r['ents_all']):<18} "
                  f"{fmt_prf(r['chems']):<18} "
                  f"{fmt_prf(r['diseases']):<18}")

            # Aggregate for micro-average
            a = agg[m]
            a["n"] += 1
            ref_words = sum(len(w) for w in WORD_TRANSFORMS(ref_text))
            a["wer_num"] += r["subs"] + r["dels"] + r["ins"]
            a["wer_den"] += ref_words
            a["rouge_sum"] += r["rouge_l"]
            gold_all, pred_all = entity_set(nlp, ref_text), entity_set(nlp, hyp_text)
            a["tp_all"]    += len(gold_all & pred_all)
            a["p_den_all"] += len(pred_all)
            a["r_den_all"] += len(gold_all)
            gold_c, pred_c = entity_set(nlp, ref_text, {"CHEMICAL"}), entity_set(nlp, hyp_text, {"CHEMICAL"})
            a["tp_chem"]    += len(gold_c & pred_c)
            a["p_den_chem"] += len(pred_c)
            a["r_den_chem"] += len(gold_c)
            gold_d, pred_d = entity_set(nlp, ref_text, {"DISEASE"}), entity_set(nlp, hyp_text, {"DISEASE"})
            a["tp_dis"]    += len(gold_d & pred_d)
            a["p_den_dis"] += len(pred_d)
            a["r_den_dis"] += len(gold_d)

    print("-" * 128)
    print("micro-averages (one row per mode):")
    for m in modes:
        a = agg[m]
        if a["n"] == 0: continue
        wer = a["wer_num"] / max(a["wer_den"], 1)
        rouge = a["rouge_sum"] / a["n"]
        def micro_prf(tp, p_den, r_den):
            p = tp / max(p_den, 1); r = tp / max(r_den, 1)
            return p, r, 0.0 if p+r==0 else 2*p*r/(p+r)
        print(f"  {m:<10}  WER {wer*100:5.2f}%  ROUGE-L {rouge*100:5.2f}%  "
              f"ALL-ents {fmt_prf(micro_prf(a['tp_all'], a['p_den_all'], a['r_den_all']))}  "
              f"CHEMs {fmt_prf(micro_prf(a['tp_chem'], a['p_den_chem'], a['r_den_chem']))}  "
              f"DISEASEs {fmt_prf(micro_prf(a['tp_dis'], a['p_den_dis'], a['r_den_dis']))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
