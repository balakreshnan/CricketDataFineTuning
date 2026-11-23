#!/usr/bin/env python
"""Upload a finished speculative-decoding benchmark to a Hugging Face dataset repo (private unless --public).

Uploads <out_dir>/summary.json, the per-answer records and a README card into the folder <run_name>/ of the repo,
so several runs can live side by side.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out_dir", required=True)
p.add_argument("--run_name", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--dataset_id", default="Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full")
p.add_argument("--public", action="store_true")
a = p.parse_args()

out = Path(a.out_dir)
s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
t = s["models"]["target"]


def row(m):
    size = f"{m['draft_params'] / 1e9:.2f}B" if m["draft_params"] else ""
    acc = "" if m["acceptance"] is None else f"{m['acceptance'] * 100:.0f}%"
    return (f"| {m['config']} | {size} | {m['gamma'] or ''} | {m['tokens_per_s']:.0f} | {m['speedup']:.2f}x | {m['sec_per_answer']:.2f} | "
            f"{m['accuracy'] * 100:.1f}% | {m['mean_tokens']:.0f} | {m['tokens_per_target_pass']:.2f} | {acc} |")


rows = "\n".join(row(m) for m in s["configs"])
target_name = t["id"] or t["dir"]
target_kind = f"with a LoRA adapter merged ({Path(t['adapter']).parent.name})" if t["adapter"] else "(base)"
decoding, max_new, n_q, n_gpu = json.dumps(s["decoding"]), s["args"]["max_new_tokens"], s["questions"], s["world_size"]
card = f"""---
license: apache-2.0
tags: [speculative-decoding, cricket, benchmark]
---
# Speculative decoding on cricket reasoning questions: run `{a.run_name}`

Target model: `{target_name}` {target_kind}.
Questions: {n_q} from the eval split of [`{a.dataset_id}`](https://huggingface.co/datasets/{a.dataset_id}),
one answer per question per configuration, at most {max_new} new tokens, batch size 1,
{n_gpu} NVIDIA Vera Rubin GPUs (one process per GPU, questions split between them).
Decoding: {decoding}.

| configuration | draft size | draft tokens per round | tokens/s | speed-up | s per answer | accuracy | mean tokens | tokens per target pass | draft tokens accepted |
|---|---|---|---|---|---|---|---|---|---|
{rows}

- `baseline` is the target alone. `<draft>-g<N>`: the draft proposes N tokens per round and the target checks them in one pass.
- Tokens per target pass is about 1.0 without speculation and at most N + 1.
- Speculative sampling preserves the target's output distribution; accuracy differences are sampling noise unless they are large.
- Decoding loop: `specdec_bench.py` (hand-written, because transformers does not support assisted generation for these stateful models).

Files: `summary.json` (all numbers, per question type too), `records_rank*.jsonl` (every answer with timing and text).
"""
(out / "README.md").write_text(card, encoding="utf-8")
print(card)

from huggingface_hub import HfApi  # noqa: E402

tmp = Path(tempfile.mkdtemp()) / "repo"
(tmp / a.run_name).mkdir(parents=True)
for f in [out / "summary.json", out / "README.md", *sorted(out.glob("records_rank*.jsonl"))]:
    shutil.copy(f, tmp / a.run_name / f.name)
shutil.copy(out / "README.md", tmp / "README.md")   # repo front page = the latest run's card
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
info = api.upload_folder(folder_path=str(tmp), repo_id=a.repo_id, repo_type="dataset", commit_message=f"Speculative decoding benchmark {a.run_name}")
print("uploaded:", getattr(info, "commit_url", info))
print("private:", api.dataset_info(a.repo_id).private)
