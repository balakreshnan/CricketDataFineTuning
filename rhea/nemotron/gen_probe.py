#!/usr/bin/env python
"""How fast does Nemotron generate in this stack, and what slows it? One GPU, fresh prompts each time.

Prints tokens/s for batch 8 x 256 new tokens with the cuDNN attention kernel on and off, and which Mamba kernels
are in use. Run by gen_probe.sbatch.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sampling  # noqa: E402

mdir, ddir = os.environ["MODEL_DIR"], os.environ["DATA_LOCAL"]
N, B = int(os.environ.get("N_TOKENS", 256)), int(os.environ.get("BATCH", 8))
tok = sampling.load_tokenizer(mdir)
model = sampling.load_model(mdir, 0).eval()
stops = sampling.eos_ids(tok, mdir)
qs = [json.loads(l) for l in open(Path(ddir) / "data" / "eval-00000.jsonl", encoding="utf-8")][:40]
print("stops", stops, [tok.convert_ids_to_tokens(s) for s in stops], "pad", tok.pad_token_id, flush=True)
print("prompt tail:", repr(sampling.prompt_text(tok, qs[0]["messages"])[-160:]), flush=True)


@torch.inference_mode()
def run(q, batch, n):
    ids = tok(sampling.prompt_text(tok, q["messages"]), add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_p=0.95,
                       max_new_tokens=n, min_new_tokens=n, num_return_sequences=batch, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (o.shape[1] - ids.shape[1]) * batch / dt, dt / (o.shape[1] - ids.shape[1]) * 1000, o, ids.shape[1]


run(qs[0], B, 32)  # warm-up
k = 1
for flag in (True, False, True, False):
    torch.backends.cuda.enable_cudnn_sdp(flag)
    for batch in (1, B):
        tps, ms, o, n0 = run(qs[k], batch, N)
        k += 1
        print(f"cudnn_sdp={flag} batch {batch}: {tps:.0f} tok/s total, {ms:.0f} ms per step, peak mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)
print("sample:", repr(tok.decode(o[0, n0:n0 + 120])), flush=True)
