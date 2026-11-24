#!/usr/bin/env python
"""Base vs fine-tuned benchmark: thinking-mode sampling on the dataset's eval split (600 questions, exact answers).

torchrun, one process per GPU. One model load: base eval -> merge the LoRA adapter -> fine-tuned eval.
Same seed, prompts and sampling settings for both. Per-sample records and summaries go to <run_dir>/eval/.
Resumable (questions already sampled are skipped).
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling  # noqa: E402

RANK, LOCAL_RANK = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))


def log(m):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--data_dir", required=True)
p.add_argument("--run_dir", required=True)
p.add_argument("--k", type=int, default=4)
p.add_argument("--max_new_tokens", type=int, default=4096)
p.add_argument("--batch_prompts", type=int, default=16)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

torch.cuda.set_device(LOCAL_RANK)
# ranks finish their shards at very different times; the default 10 min collective timeout kills the waiters
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"), timeout=timedelta(hours=4))
run = Path(a.run_dir)
out = run / "eval"
qs = [json.loads(l) for l in open(Path(a.data_dir) / "data" / "eval-00000.jsonl", encoding="utf-8")]
if a.limit:
    qs = qs[: a.limit]
tok = sampling.load_tokenizer(a.model_dir)
model = sampling.load_model(a.model_dir, LOCAL_RANK)
log(f"world {dist.get_world_size()}; {len(qs)} questions x {a.k} samples")


def run_eval(tag):
    t0 = time.time()
    left = sampling.sample(model, tok, qs, out, tag, k=a.k, max_new_tokens=a.max_new_tokens,
                           batch_prompts=a.batch_prompts, seed=7, log=log)
    sampling.pending_everywhere(left)
    dist.barrier()
    if RANK == 0:
        m = sampling.metrics(sampling.collect(out, tag), a.k)
        m["_wall_s"] = time.time() - t0
        (out / f"{tag}_summary.json").write_text(json.dumps(m, indent=2))
        for qt, x in m.items():
            if isinstance(x, dict):
                log(f"[{tag}] {qt}: acc={x['accuracy']:.4f} pass@{a.k}={x[f'pass@{a.k}']:.4f} "
                    f"tokens={x['mean_tokens']:.0f} truncated={x['truncated_rate']:.4f} (n={x['questions']})")


run_eval("base")
from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, str(run / "adapter")).merge_and_unload()
log("adapter merged")
run_eval("ft")
dist.barrier()
dist.destroy_process_group()
