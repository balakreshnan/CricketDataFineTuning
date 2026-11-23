#!/usr/bin/env python
"""Why is a hand-written decode loop slower than generate()? One process, one GPU, greedy, N tokens per variant."""
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sampling  # noqa: E402

N = int(os.environ.get("N_TOKENS", 64))
if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
dev = torch.device("cuda:0")
mdir, ddir = os.environ["MODEL_DIR"], os.environ["DATA_LOCAL"]
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
q = json.loads(open(Path(ddir) / "data" / "eval-00000.jsonl", encoding="utf-8").readline())
prompt = tok(sampling.prompt_text(tok, q["messages"]), add_special_tokens=False)["input_ids"]
print("threads", torch.get_num_threads(), "cpus", len(os.sched_getaffinity(0)), "prompt", len(prompt), flush=True)


def timed(name, fn, reps=2):
    for r in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{name} run {r}: {n} tokens in {dt:.2f} s = {n / dt:.1f} tok/s ({dt / n * 1000:.0f} ms/token)", flush=True)


@torch.inference_mode()
def hf():
    ids = torch.tensor([prompt], device=dev)
    o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=N, min_new_tokens=N, do_sample=False)
    return o.shape[1] - len(prompt)


@torch.inference_mode()
def loop(mask=False, on_gpu=False, keep1=False):
    kw = {"logits_to_keep": 1} if keep1 else {}
    ids = torch.tensor([prompt], device=dev)
    am = torch.ones_like(ids) if mask else None
    o = model(input_ids=ids, attention_mask=am, use_cache=True, **kw)
    cache, n = o.past_key_values, 0
    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    while True:
        n += 1
        if n >= N:
            return n
        if not on_gpu:
            nxt = torch.tensor([[int(nxt)]], device=dev)  # what specdec_bench does: token goes through Python
        if mask:
            am = torch.cat([am, torch.ones_like(nxt)], dim=1)
        o = model(input_ids=nxt, attention_mask=am, past_key_values=cache, use_cache=True, **kw)
        cache = o.past_key_values
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)


timed("generate()", hf)
timed("loop plain", loop)

# ---- the benchmark's own baseline loop, piece by piece ----
TOP_K, TOP_P = 20, 0.95


def warp(logits, greedy):
    if greedy:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    v, i = logits.topk(TOP_K, dim=-1)
    pr = v.softmax(-1)
    pr = pr * ((pr.cumsum(-1) - pr) < TOP_P)
    return torch.zeros_like(logits).scatter_(-1, i, pr / pr.sum(-1, keepdim=True))


@torch.inference_mode()
def bench_loop(greedy, variant="bench"):
    g = torch.Generator(device=dev).manual_seed(1)
    t = {"feed": 0.0, "float": 0.0, "warp": 0.0, "draw": 0.0}
    cache, toks, n = None, prompt, 0

    def tick(key, t0):
        torch.cuda.synchronize()
        t[key] += time.perf_counter() - t0
        return time.perf_counter()

    while n < N:
        t0 = time.perf_counter()
        o = model(input_ids=torch.tensor([toks], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        t0 = tick("feed", t0)
        lg = o.logits[0, -1:].float()
        t0 = tick("float", t0)
        if variant == "small":  # sample among the top-k values only, never build a vocabulary-sized vector
            v, i = lg.topk(TOP_K, dim=-1)
            pr = v.softmax(-1)
            t0 = tick("warp", t0)
            nxt = int(i[0, torch.multinomial(pr[0], 1, generator=g)])
        else:
            pr = warp(lg, greedy)
            t0 = tick("warp", t0)
            nxt = int(torch.multinomial(pr[-1], 1, generator=g))
        tick("draw", t0)
        toks, n = [nxt], n + 1
    print("   pieces (s):", {k: round(x, 2) for k, x in t.items()}, flush=True)
    return n


timed("bench loop sampling", lambda: bench_loop(False))
timed("bench loop greedy", lambda: bench_loop(True))
timed("bench loop sampling, top-k-only draw", lambda: bench_loop(False, "small"))
