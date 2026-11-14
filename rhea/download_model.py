#!/usr/bin/env python
"""Download $MODEL_ID to $MODEL_DIR and print what is needed to adapt the training code: config and tensor-name patterns."""
import collections
import glob
import json
import os
import re

from huggingface_hub import snapshot_download
from safetensors import safe_open

mid, mdir = os.environ["MODEL_ID"], os.environ["MODEL_DIR"]
snapshot_download(mid, local_dir=mdir, token=os.environ.get("HF_TOKEN"), max_workers=16)
print(sorted(os.listdir(mdir)))
cfg = json.load(open(f"{mdir}/config.json"))
print("CONFIG", json.dumps(cfg)[:6000])
pat, total = collections.Counter(), 0
shapes = {}
for f in sorted(glob.glob(f"{mdir}/*.safetensors")):
    with safe_open(f, "pt") as st:
        for k in st.keys():
            sl = st.get_slice(k)
            n = 1
            for d in sl.get_shape():
                n *= d
            total += n
            key = re.sub(r"\.\d+\.", ".N.", k)
            pat[key] += 1
            shapes.setdefault(key, sl.get_shape())
print("PARAMS", total)
for k, c in sorted(pat.items()):
    print("T", c, k, shapes[k])
tk = json.load(open(f"{mdir}/tokenizer_config.json"))
print("CHAT_TEMPLATE_TAIL", repr(str(tk.get("chat_template", ""))[-700:]))
