#!/usr/bin/env python
"""Download each draft model in $DRAFT_IDS to $MODELS_DIR/<name> and check it against the target's vocabulary."""
import json
import os

from huggingface_hub import snapshot_download

tv = json.load(open(os.environ["TARGET_DIR"] + "/config.json"))["text_config"]["vocab_size"]
for mid in os.environ["DRAFT_IDS"].split():
    d = os.path.join(os.environ["MODELS_DIR"], mid.split("/")[-1])
    snapshot_download(mid, local_dir=d, token=os.environ.get("HF_TOKEN"), max_workers=16)
    c = json.load(open(d + "/config.json"))
    t = c.get("text_config", c)
    verdict = "matches" if t["vocab_size"] == tv else "DOES NOT MATCH"
    print(f"DRAFT {mid}: type {c.get('model_type')} layers {t['num_hidden_layers']} hidden {t['hidden_size']} "
          f"vocab {t['vocab_size']} ({verdict} the target's {tv})", flush=True)
