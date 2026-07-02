"""Download selected FLEURS language configs (audio tar + transcripts) to /mnt/data.
Usage: download_fleurs.py hi_in ga_ie cmn_hans_cn ...
"""
import sys
from huggingface_hub import snapshot_download
DEST = "/mnt/data/omnivoice_ipa/corpus/fleurs"
langs = sys.argv[1:]
patterns = []
for l in langs:
    patterns += [f"data/{l}/*"]
print("downloading FLEURS langs:", langs)
snapshot_download("google/fleurs", repo_type="dataset", local_dir=DEST,
                  allow_patterns=patterns, max_workers=4)
print("done")
