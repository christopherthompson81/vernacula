"""Prefetch FLEURS train.tar.gz for the 24 minimal-set langs into the persistent
audio cache — download only, NO encode (so it won't contend with a running GPU job).

Once these are cached, the full multi-speaker re-ingest (ingest_fleurs.py, now keyed
by wav basename) reads them locally with no re-download. ~40 GB for the 24 langs.
"""
import os
from huggingface_hub import hf_hub_download

AUDIO_CACHE = "/mnt/data/omnivoice_ipa/corpus/audio_cache"
LANGS = ["en_us", "cmn_hans_cn", "hi_in", "es_419", "ar_eg", "fr_fr", "pt_br", "ru_ru",
         "de_de", "ja_jp", "tr_tr", "vi_vn", "ta_in", "ko_kr", "ha_ng", "th_th",
         "ff_sn", "kk_kz", "zu_za", "cs_cz", "sv_se", "ca_es", "ga_ie", "cy_gb"]

os.makedirs(AUDIO_CACHE, exist_ok=True)
for i, lang in enumerate(LANGS, 1):
    try:
        p = hf_hub_download("google/fleurs", f"data/{lang}/audio/train.tar.gz",
                            repo_type="dataset", local_dir=AUDIO_CACHE)
        mb = os.path.getsize(p) / 1e6
        print(f"[{i}/{len(LANGS)}] {lang}: {mb:.0f} MB cached", flush=True)
    except Exception as e:
        print(f"[{i}/{len(LANGS)}] {lang}: FAILED {type(e).__name__}: {e}", flush=True)
print("prefetch done")
