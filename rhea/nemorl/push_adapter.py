#!/usr/bin/env python
"""Upload the LoRA adapter trained with NeMo RL to Hugging Face. Adapter only: the weights are NOT merged.

usage: push_adapter.py <adapter_dir> <repo_id> [--results results.json] [--private]
The repo is PUBLIC unless --private is given (user request, 2026-11-24).
The model card tells users that the adapter must be applied to (or merged into) the base model at inference time.
<adapter_dir> must contain adapter_model.safetensors and adapter_config.json.
results.json (optional): {"run": ..., "steps": ..., "nodes": ..., "base_accuracy": ..., "tuned_accuracy": ...,
"base_tokens": ..., "tuned_tokens": ..., "wandb": ..., "notes": ...}; missing keys are left out of the card.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
p = argparse.ArgumentParser()
p.add_argument("adapter_dir")
p.add_argument("repo_id")
p.add_argument("--results")
p.add_argument("--private", action="store_true")
a = p.parse_args()
src = Path(a.adapter_dir)
for f in ("adapter_model.safetensors", "adapter_config.json"):
    assert (src / f).is_file(), f"missing {src / f}"
r = json.load(open(a.results)) if a.results else {}


def row(label, key, fmt="{}"):
    return f"| {label} | {fmt.format(r[key])} |\n" if r.get(key) is not None else ""


results = ""
if r.get("base_accuracy") is not None and r.get("tuned_accuracy") is not None:
    results = ("## Results\n\n600 held-out questions, 4 sampled answers each (temperature 1.0, top-p 0.95).\n\n"
               "| | Base model | Base + this adapter |\n|---|---|---|\n"
               f"| Accuracy | {r['base_accuracy']:.2f}% | {r['tuned_accuracy']:.2f}% |\n")
    if r.get("base_tokens") is not None and r.get("tuned_tokens") is not None:
        results += f"| Mean output tokens | {r['base_tokens']:.0f} | {r['tuned_tokens']:.0f} |\n"
    results += "\n"

card = f"""---
base_model: {BASE}
library_name: peft
tags: [lora, reinforcement-learning, grpo, nemo-rl, nemo-gym, nemotron, cricket]
---

# Nemotron 3.5 Lightning 30B: cricket reasoning LoRA adapter (NeMo RL, GRPO)

**This repository contains LoRA adapter weights only. It is not a standalone model.**
To run inference you must load the base model [`{BASE}`](https://huggingface.co/{BASE}) and then either
apply this adapter on top of it or merge the adapter into the base weights. Loading this repository on its own
does not work. The base model weights were not modified and are not included here.

## What it is

A rank-32 LoRA adapter trained with reinforcement learning (GRPO in [NeMo RL](https://github.com/NVIDIA-NeMo/RL))
so that the model answers T20 cricket rate and milestone questions with exact numeric answers, using fewer
reasoning tokens. Rewards came from a NeMo Gym environment that checks the final number exactly:
a finished, correct answer scores `1 - 0.4 * output_tokens / 4096`, anything else scores 0.

| | |
|---|---|
| Base model | `{BASE}` |
| Method | GRPO with LoRA (NeMo RL, Automodel backend), vLLM generation, NeMo Gym rewards |
| LoRA | rank 32, alpha 64, on attention q/k/v/o projections, Mamba `in_proj`, shared-expert up/down projections |
| Dataset | `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` |
| Hardware | NVIDIA Vera Rubin, 4 GPUs per node |
{row("Nodes", "nodes")}{row("Training steps", "steps")}{row("Run", "run")}{row("Weights & Biases", "wandb")}
{results}## How to use it (the adapter must be applied or merged at inference)

Files: `adapter_model.safetensors`, `adapter_config.json` (plus `automodel_peft_config.json`, the NeMo Automodel
description of the same adapter).

**Option 1: apply the adapter at load time with PEFT**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "{BASE}"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{a.repo_id}")   # adapter applied on top of the base model
```

**Option 2: merge the adapter into the base weights, then use the result like any model**

```python
model = model.merge_and_unload()          # after the PeftModel.from_pretrained call above
model.save_pretrained("nemotron-cricket-merged")   # optional: a standalone 61 GB copy
```

**Option 3: serve with vLLM, adapter as a LoRA module**

```bash
vllm serve {BASE} --enable-lora --max-lora-rank 64 \\
    --lora-modules cricket=<local path to this repository> --reasoning-parser nemotron_v3
```

Then request the model name `cricket` (the adapter) instead of the base model name.

{r.get("notes", "")}
"""
(src / "README.md").write_text(card, encoding="utf-8")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo_id, private=a.private, exist_ok=True)
api.upload_folder(folder_path=str(src), repo_id=a.repo_id, commit_message="LoRA adapter trained with NeMo RL (GRPO)")
info = api.model_info(a.repo_id)
print(f"uploaded {sorted(f.rfilename for f in info.siblings)} to https://huggingface.co/{a.repo_id} (private={info.private})")
