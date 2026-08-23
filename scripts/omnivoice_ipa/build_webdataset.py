"""Repackage the collected corpus into OmniVoice's training data format.

OmniVoice's Dataset (omnivoice/data/dataset.py) reads WebDataset tar shards keyed by
__key__, with pre-extracted int16 audio codes as a per-sample `<id>.npy` [8,T] array,
plus a SEPARATE per-shard label JSONL ({"id":..., "text":...}) that the processor reads
for training fields. A `data.lst` manifest ("tar label_jsonl num_items num_seconds")
lists shards per language, referenced by a top-level data-config JSON.

Scope: exclusively-IPA fine-tune (no orthographic text, no use_pinyin_ratio-style
substitution needed) — the phonemized IPA string goes straight into the "text" field,
since OmniVoiceSampleProcessor does no validation on what "text" contains.

Per-language `repeat` in the data-config implements Task #3's density-flattening
sampling weights (work/sampling_weights.csv) directly — WebDataset's language sampler
reads a language `repeat`x per epoch, exactly the oversampling factor we computed.

Outputs (/mnt/data/omnivoice_ipa/train/):
  shards/<lang>/000000.tar        (all utts of a language in one shard; corpus is small)
  shards/<lang>/000000.jsonl      ({"id","text"} per line, aligned to the tar)
  shards/<lang>/data.lst          ("tar jsonl num_items num_seconds")
  data_config.json                {"train": [{"language_id","manifest_path","repeat"}, ...]}
"""
import io
import json
import math
import os
import sys
import tarfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filter import load_exclusions, load_manifest  # noqa: E402

ROOT = "/mnt/data/omnivoice_ipa"
TOKENS = f"{ROOT}/corpus/tokens"
OUT = f"{ROOT}/train/shards"
DATA_CONFIG = f"{ROOT}/train/data_config.json"


DEV_HOLDOUT_FRAC = 0.03   # per-language dev split for loss monitoring, not a rigorous eval
DEV_HOLDOUT_MIN = 20
DEV_HOLDOUT_MAX = 80


def _write_shard(lang_dir, split, rows, codes):
    tar_path = f"{lang_dir}/{split}.tar"
    jsonl_path = f"{lang_dir}/{split}.jsonl"
    n_items = 0
    total_seconds = 0.0
    with tarfile.open(tar_path, "w") as tar, open(jsonl_path, "w", encoding="utf-8") as jf:
        for row in rows:
            uid = row["id"]
            if uid not in codes:
                continue
            arr = codes[uid].astype(np.int16)
            buf = io.BytesIO()
            np.save(buf, arr)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"{uid}.npy")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            jf.write(json.dumps({"id": uid, "text": row["ipa"]}, ensure_ascii=False) + "\n")
            n_items += 1
            total_seconds += row["dur_s"]
    with open(f"{lang_dir}/data_{split}.lst", "w", encoding="utf-8") as f:
        f.write(f"{tar_path} {jsonl_path} {n_items} {total_seconds:.1f}\n")
    return n_items, total_seconds


def _repeat_shard(lang_dir, split, repeat):
    """Write `repeat` distinct hardlinked copies of a shard's data.lst lines.

    webdataset's group_by_keys reopens the SAME tar URL back-to-back as one
    continuous member stream when a manifest lists it multiple times (the
    library's own `* repeat` trick) — it then sees each basename twice and
    raises "duplicate file name in tar file". Hardlinked copies under distinct
    names are indistinguishable in content but distinct URLs, sidestepping it.
    """
    base_tar, base_jsonl = f"{lang_dir}/{split}.tar", f"{lang_dir}/{split}.jsonl"
    n_items = total_seconds = None
    with open(f"{lang_dir}/data_{split}.lst", "w", encoding="utf-8") as out:
        orig_line = open(f"{lang_dir}/_data_{split}_single.lst", encoding="utf-8").read().strip()
        _, _, n_items, total_seconds = orig_line.split(" ", 3)
        for i in range(repeat):
            tar_i, jsonl_i = f"{lang_dir}/{split}_r{i}.tar", f"{lang_dir}/{split}_r{i}.jsonl"
            if not os.path.exists(tar_i):
                os.link(base_tar, tar_i)
            if not os.path.exists(jsonl_i):
                os.link(base_jsonl, jsonl_i)
            out.write(f"{tar_i} {jsonl_i} {n_items} {total_seconds}\n")


def build_lang(lang, repeat, exclusions):
    codes = np.load(f"{TOKENS}/codes_{lang}.npz")
    # Audio-gate exclusion (work/exclusions.tsv) is applied HERE and, more importantly, in
    # sampling_budget.py — see corpus_filter.py for why the order matters.
    manifest, n_dropped = load_manifest(lang, exclusions)
    lang_dir = f"{OUT}/{lang}"
    os.makedirs(lang_dir, exist_ok=True)

    # ⚠ THE SPLIT IS BY SENTENCE, NOT BY ROW, AND THE DIFFERENCE IS TRAIN/DEV LEAKAGE.
    #
    # The previous version deduped on `id` before splitting, with a comment saying `id` was a
    # per-SENTENCE key shared across speakers. It is not — `id` is the WAV STEM, unique per
    # recording, so the dedup was a no-op and the split was a plain row slice. FLEURS records the
    # same sentence with ~2.2 speakers on average (cy_gb: 3,263 recordings over 1,502 sentences),
    # so slicing rows put the SAME SENTENCE in both splits, read by a different voice: measured at
    # 73-99% of every dev set (xh_za 99%, cy_gb 95%, en_us 87%). Dev loss was scoring recall of a
    # sentence already trained on, not generalization.
    #
    # Grouping by `sentence_id` and assigning whole groups makes the splits text-disjoint. Dev is
    # then sized in SENTENCES and lands slightly over the row target when a group has several
    # speakers, which is the correct direction: a clean small dev beats a leaky larger one.
    by_sentence = {}
    for r in manifest:
        by_sentence.setdefault(r["sentence_id"], []).append(r)
    groups = list(by_sentence.values())
    n_dev_rows = min(max(round(len(manifest) * DEV_HOLDOUT_FRAC), DEV_HOLDOUT_MIN), DEV_HOLDOUT_MAX)
    n_dev_rows = min(n_dev_rows, len(manifest) // 10)  # never hold out more than 10%
    dev_rows, train_rows, taken = [], [], 0
    for g in groups:
        if taken < n_dev_rows:
            dev_rows.extend(g)
            taken += len(g)
        else:
            train_rows.extend(g)
    assert not ({r["sentence_id"] for r in dev_rows} & {r["sentence_id"] for r in train_rows}), \
        f"{lang}: dev/train share a sentence_id"
    train_n, train_secs = _write_shard(lang_dir, "train", train_rows, codes)
    dev_n, dev_secs = _write_shard(lang_dir, "dev", dev_rows, codes)
    # _write_shard already wrote data_train.lst / data_dev.lst as single-entry
    # manifests; stash them, then rebuild data_train.lst with `repeat` distinct
    # hardlinked copies (dev is never repeated).
    os.replace(f"{lang_dir}/data_train.lst", f"{lang_dir}/_data_train_single.lst")
    _repeat_shard(lang_dir, "train", repeat)
    return train_n, train_secs, dev_n, dev_secs, n_dropped


def main():
    weights = pd.read_csv(f"{ROOT}/work/sampling_weights.csv").set_index("lang")["weight"].to_dict()
    exclusions = load_exclusions()
    total_dropped = 0
    # ⚠ THE LANGUAGE SET IS THE COVERAGE ARGUMENT, NOT WHATEVER HAPPENS TO BE INGESTED. This globbed
    #   the tokens dir, which silently became a 102-language build the moment the corpus completed —
    #   discarding the census-derived greedy cover that `sampling_budget.POP_ORDER` encodes. Each of
    #   those 28 is present as the OWNER of specific IPA primitives (English the 53 generalist base
    #   letters, Zulu clicks and breathy voice, Hausa ejectives, Fula prenasals), and the MAX_WEIGHT=3
    #   reasoning was tuned against that mix. Training on all 102 is a legitimate but DIFFERENT
    #   experiment; it must be chosen, not inherited from an ls.
    #   `--all` opts into it explicitly.
    if "--all" in sys.argv:
        langs = sorted(w[len("codes_"):-len(".npz")] for w in os.listdir(TOKENS) if w.startswith("codes_"))
    else:
        from sampling_budget import POP_ORDER
        have = {w[len("codes_"):-len(".npz")] for w in os.listdir(TOKENS) if w.startswith("codes_")}
        langs = [l for l in POP_ORDER if l in have]
        missing = [l for l in POP_ORDER if l not in have]
        if missing:
            print(f"⚠ coverage-set languages with no codes: {' '.join(missing)}")
        print(f"coverage set: {len(langs)} languages (pass --all for every ingested language)")
    train_entries, dev_entries = [], []
    for lang in langs:
        # CEIL, not round: a weight of W means the scarcest owned primitive sits at
        # N_TOKENS/W exposures and needs W× to clear the redundancy target. round() drops
        # sub-0.5 boosts to 1× (the first-attempt bug: ha_ng 1.31→1, ga_ie 1.46→1 got NO
        # oversampling, leaving those thin phones under target). ceil guarantees every
        # language's scarcest primitive reaches ≥ N_TOKENS. Physical repeats are integer
        # shard copies, so ceil is the right whole-number floor on the boost.
        repeat = max(1, math.ceil(weights.get(lang, 1.0) - 1e-6))
        train_n, train_secs, dev_n, dev_secs, n_dropped = build_lang(lang, repeat, exclusions)
        total_dropped += n_dropped
        # repeat is realized as `repeat` distinct hardlinked shard copies (see
        # _repeat_shard) rather than the JSON "repeat" field, which triggers a
        # webdataset group_by_keys "duplicate file name" error when the same tar
        # URL is listed twice back-to-back — so this stays 1 always.
        train_entries.append(dict(language_id=lang, manifest_path=[f"{OUT}/{lang}/data_train.lst"], repeat=1))
        dev_entries.append(dict(language_id=lang, manifest_path=[f"{OUT}/{lang}/data_dev.lst"], repeat=1))
        print(f"{lang}: train {train_n} ({train_secs/60:.1f} min) x{repeat} copies, "
              f"dev {dev_n} ({dev_secs/60:.1f} min)"
              + (f"  [-{n_dropped} defective]" if n_dropped else ""))
    os.makedirs(os.path.dirname(DATA_CONFIG), exist_ok=True)
    json.dump({"train": train_entries, "dev": dev_entries},
               open(DATA_CONFIG, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {DATA_CONFIG} ({len(train_entries)} languages)")
    print(f"   excluded {total_dropped} defective-audio utterances (work/exclusions.tsv)"
          + ("  ⚠ ZERO — has exclude_defective.py been run?" if total_dropped == 0 else ""))


if __name__ == "__main__":
    main()
