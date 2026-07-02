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
import tarfile
import numpy as np
import pandas as pd

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


def build_lang(lang, repeat):
    codes = np.load(f"{TOKENS}/codes_{lang}.npz")
    manifest = [json.loads(l) for l in open(f"{TOKENS}/manifest_{lang}.jsonl", encoding="utf-8")]
    # FLEURS 'id' is a per-SENTENCE id shared across speakers; ingest_fleurs.py keyed
    # codes by it, so only one speaker's codes survive per id (npz is deduped already).
    # Dedup the manifest to match (one row per id) BEFORE the dev/train split, else the
    # same id lands in both splits = train/dev leakage. IPA is identical across an id's
    # rows (same sentence), so keeping the first row loses nothing that training reads.
    seen = set()
    manifest = [r for r in manifest if not (r["id"] in seen or seen.add(r["id"]))]
    lang_dir = f"{OUT}/{lang}"
    os.makedirs(lang_dir, exist_ok=True)
    n_dev = min(max(round(len(manifest) * DEV_HOLDOUT_FRAC), DEV_HOLDOUT_MIN), DEV_HOLDOUT_MAX)
    n_dev = min(n_dev, len(manifest) // 10)  # never hold out more than 10%
    dev_rows, train_rows = manifest[:n_dev], manifest[n_dev:]
    train_n, train_secs = _write_shard(lang_dir, "train", train_rows, codes)
    dev_n, dev_secs = _write_shard(lang_dir, "dev", dev_rows, codes)
    # _write_shard already wrote data_train.lst / data_dev.lst as single-entry
    # manifests; stash them, then rebuild data_train.lst with `repeat` distinct
    # hardlinked copies (dev is never repeated).
    os.replace(f"{lang_dir}/data_train.lst", f"{lang_dir}/_data_train_single.lst")
    _repeat_shard(lang_dir, "train", repeat)
    return train_n, train_secs, dev_n, dev_secs


def main():
    weights = pd.read_csv(f"{ROOT}/work/sampling_weights.csv").set_index("lang")["weight"].to_dict()
    langs = sorted(w[len("codes_"):-len(".npz")] for w in os.listdir(TOKENS) if w.startswith("codes_"))
    train_entries, dev_entries = [], []
    for lang in langs:
        # CEIL, not round: a weight of W means the scarcest owned primitive sits at
        # N_TOKENS/W exposures and needs W× to clear the redundancy target. round() drops
        # sub-0.5 boosts to 1× (the first-attempt bug: ha_ng 1.31→1, ga_ie 1.46→1 got NO
        # oversampling, leaving those thin phones under target). ceil guarantees every
        # language's scarcest primitive reaches ≥ N_TOKENS. Physical repeats are integer
        # shard copies, so ceil is the right whole-number floor on the boost.
        repeat = max(1, math.ceil(weights.get(lang, 1.0) - 1e-6))
        train_n, train_secs, dev_n, dev_secs = build_lang(lang, repeat)
        # repeat is realized as `repeat` distinct hardlinked shard copies (see
        # _repeat_shard) rather than the JSON "repeat" field, which triggers a
        # webdataset group_by_keys "duplicate file name" error when the same tar
        # URL is listed twice back-to-back — so this stays 1 always.
        train_entries.append(dict(language_id=lang, manifest_path=[f"{OUT}/{lang}/data_train.lst"], repeat=1))
        dev_entries.append(dict(language_id=lang, manifest_path=[f"{OUT}/{lang}/data_dev.lst"], repeat=1))
        print(f"{lang}: train {train_n} ({train_secs/60:.1f} min) x{repeat} copies, "
              f"dev {dev_n} ({dev_secs/60:.1f} min)")
    os.makedirs(os.path.dirname(DATA_CONFIG), exist_ok=True)
    json.dump({"train": train_entries, "dev": dev_entries},
               open(DATA_CONFIG, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {DATA_CONFIG} ({len(train_entries)} languages)")


if __name__ == "__main__":
    main()
