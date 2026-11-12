#!/usr/bin/env python3
"""Tiny OpenAI-API speed probe (stdlib only): tokens/s at concurrency 1 (latency) and N (throughput).

  bench_client.py <base_url> <model> <eval.jsonl> <label> [n_questions] [concurrency] [max_tokens]
Prints one JSON line per phase: completion tokens, wall seconds, tokens/s, errors.
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, model, path, label = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
conc = int(sys.argv[6]) if len(sys.argv) > 6 else 32
max_tokens = int(sys.argv[7]) if len(sys.argv) > 7 else 2048
qs = [json.loads(l) for l in open(path, encoding="utf-8")][:n]


def ask(q):
    body = json.dumps({"model": model, "messages": q["messages"], "max_tokens": max_tokens, "temperature": 1.0,
                       "top_p": 0.95, "seed": 7}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
        return r["usage"]["completion_tokens"], time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return 0, time.time() - t0, repr(e)[:200]


def phase(name, items, workers):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(ask, items))
    wall = time.time() - t0
    toks = sum(r[0] for r in res)
    errs = [r[2] for r in res if r[2]]
    print(json.dumps({"label": label, "phase": name, "requests": len(items), "concurrency": workers,
                      "completion_tokens": toks, "wall_s": round(wall, 1), "tok_per_s": round(toks / wall, 1),
                      "errors": len(errs), "first_error": errs[0] if errs else None}), flush=True)


ask(qs[0])                         # warm-up (compiles kernels / CUDA graphs)
phase("latency", qs[:4], 1)
phase("throughput", qs, conc)
