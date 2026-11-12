#!/usr/bin/env python
"""Upload a finished LoRA adapter to the Hugging Face Hub (private unless --public).

Uploads adapter_model.safetensors, adapter_config.json (base model path rewritten from the local Lustre dir to the
hub id) and a model card built from train_result.json. Checkpoints and training_args.bin are not uploaded.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--base_model_id", default="Qwen/Qwen3.8-27B")
p.add_argument("--dataset_id", default="Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full")
p.add_argument("--public", action="store_true")
a = p.parse_args()

run = Path(a.run_dir)
res = json.loads((run / "train_result.json").read_text())
ev, ta = res["eval"], res["args"]
tmp = Path(tempfile.mkdtemp()) / "repo"
tmp.mkdir()
shutil.copy(run / "adapter" / "adapter_model.safetensors", tmp / "adapter_model.safetensors")
cfg = json.loads((run / "adapter" / "adapter_config.json").read_text())
cfg["base_model_name_or_path"] = a.base_model_id
(tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

card = f"""---
base_model: {a.base_model_id}
library_name: peft
pipeline_tag: text-generation
datasets: [{a.dataset_id}]
tags: [lora, reasoning, cricket, sports-analytics]
---

# Qwen3.8-27B cricket reasoning LoRA (Rhea run)

LoRA adapter for `{a.base_model_id}` fine-tuned for one epoch on the training split of
[`{a.dataset_id}`](https://huggingface.co/datasets/{a.dataset_id}): verified thinking-mode reasoning traces for
T20 International cricket questions (run rate, chase rate, milestone, projection, strike rate, legal balls).

## Training

| | |
|---|---|
| Method | LoRA r={ta['lora_r']}, alpha={ta['lora_alpha']}, dropout={ta['lora_dropout']} on the language model's attention, Gated-DeltaNet and MLP projections |
| Epochs | {ta['epochs']} (1,788 optimizer steps, global batch 256) |
| Learning rate | {ta['lr']}, cosine schedule, {ta['warmup_ratio']:.0%} warmup |
| Max sequence length | {ta['max_len']} tokens |
| Precision | bf16 |
| Hardware | 8 nodes x 4 NVIDIA Vera Rubin GPUs (32 GPUs), DDP, about 4.5 hours |
| Final dev loss | {ev['eval_loss']:.4f} (1,000 held-out training rows) |

Each training sequence is the thinking-mode chat prompt (system + user) followed by
`reasoning + "\\n</think>\\n\\n" + response`; the loss covers only the reasoning and the answer.

## Evaluation

Only the dev loss above has been measured for this adapter. Answer accuracy and reasoning length on the benchmark
split have not been evaluated yet.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{a.base_model_id}")
model = AutoModelForImageTextToText.from_pretrained("{a.base_model_id}", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{a.repo_id}")
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
```

Use thinking mode (`enable_thinking=True`); the adapter was trained only on thinking-mode outputs.
"""
(tmp / "README.md").write_text(card, encoding="utf-8")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo_id, private=not a.public, exist_ok=True)
info = api.upload_folder(folder_path=str(tmp), repo_id=a.repo_id, commit_message="Upload LoRA adapter (Rhea run)")
print("uploaded:", info.commit_url if hasattr(info, "commit_url") else info)
print("private:", api.model_info(a.repo_id).private)
for f in api.list_repo_tree(a.repo_id):
    print(" ", f.path, getattr(f, "size", ""))
