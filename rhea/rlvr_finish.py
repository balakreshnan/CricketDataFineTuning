#!/usr/bin/env python
"""After RLVR training + the benchmark: log before/after to the run's W&B page and push the adapter to the Hub.

Reads <run_dir>/eval/{base,ft}_summary.json (eval.py), <run_dir>/train_result.json and metrics.jsonl (rlvr.py).
Uploads adapter_model.safetensors, adapter_config.json (base model id instead of the Lustre path), a model card with
the measured before/after numbers and eval_summary.json. Private unless --public.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--base_model_id", default="Qwen/Qwen3.8-27B")
p.add_argument("--dataset_id", default="Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full")
p.add_argument("--public", action="store_true")
p.add_argument("--no_push", action="store_true")
a = p.parse_args()

run = Path(a.run_dir)
res = json.loads((run / "train_result.json").read_text())
B = json.loads((run / "eval" / "base_summary.json").read_text())
F = json.loads((run / "eval" / "ft_summary.json").read_text())
ta = res["args"]
types = [t for t in ("overall", "run_rate", "chase_rate", "milestone") if t in B and t in F]
summary = {"benchmark": {t: {"base": B[t], "rlvr": F[t]} for t in types}, "training": {k: res[k] for k in res if k != "args"}}

if os.environ.get("WANDB_API_KEY") and (run / "wandb_run_id.txt").exists():
    import wandb
    wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-rlvr-rhea"),
                    id=(run / "wandb_run_id.txt").read_text().strip(), resume="allow")
    tab = wandb.Table(columns=["question_type", "metric", "before (base)", "after (RLVR)", "change"])
    for t in types:
        for m in ("accuracy", "pass@4", "mean_tokens", "truncated_rate"):
            tab.add_data(t, m, B[t][m], F[t][m], F[t][m] - B[t][m])
            wb.summary[f"benchmark/{t}/{m}/before"] = B[t][m]
            wb.summary[f"benchmark/{t}/{m}/after"] = F[t][m]
    wb.log({"benchmark/before_vs_after": tab})
    wb.finish()
    print("logged benchmark to W&B run", (run / "wandb_run_id.txt").read_text().strip())

o_b, o_f = B["overall"], F["overall"]
rows = "\n".join(f"| {t} | {B[t]['accuracy'] * 100:.2f}% | {F[t]['accuracy'] * 100:.2f}% | {B[t]['mean_tokens']:.0f} | "
                 f"{F[t]['mean_tokens']:.0f} | {B[t]['truncated_rate'] * 100:.2f}% | {F[t]['truncated_rate'] * 100:.2f}% |" for t in types)
card = f"""---
base_model: {a.base_model_id}
library_name: peft
pipeline_tag: text-generation
datasets: [{a.dataset_id}]
tags: [lora, reasoning, cricket, reinforcement-learning, rlvr]
---

# Qwen3.8-27B cricket reasoning LoRA, trained with RLVR

LoRA adapter for `{a.base_model_id}` trained with reinforcement learning with verifiable rewards (RLVR) on questions
from [`{a.dataset_id}`](https://huggingface.co/datasets/{a.dataset_id}). No reasoning traces were used as targets:
the model answered the questions itself and was rewarded only when its final number matched the exact answer.

## Method

- Each step, the current policy answers questions in thinking mode, {ta['group']} samples per question, plain
  temperature-1 sampling, at most {ta['max_new_tokens']} new tokens.
- Reward: finished and correct = 1 - {ta['len_coef']} x tokens / {ta['max_new_tokens']}; wrong, no answer, or cut off = 0.
- Advantage: reward minus the mean reward of that question's samples. Loss: advantage x mean token negative log-likelihood
  of the sampled answer (policy gradient with a group baseline, one update per batch of fresh rollouts).
- {res['steps']} steps, {res['rollouts_total']:,} rollouts in total, {res['world_size']} GPUs, learning rate {ta['lr']},
  LoRA r={ta['lora_r']}, alpha={ta['lora_alpha']} on the language model's attention, Gated-DeltaNet and MLP projections.
- Training time: {res['train_hours']:.1f} hours on NVIDIA Vera Rubin GPUs.

## Before and after

600 held-out benchmark questions, 4 samples each (temperature 1.0, top-p 0.95, top-k 20, up to 4,096 new tokens).
"Before" is the base model, "after" is the base model with this adapter merged.

| question type | accuracy before | accuracy after | mean tokens before | mean tokens after | cut off before | cut off after |
|---|---|---|---|---|---|---|
{rows}

During training (rollouts on training questions, average of the first and last {res['window']} steps): accuracy
{res['first_steps']['rollout_accuracy'] * 100:.1f}% to {res['last_steps']['rollout_accuracy'] * 100:.1f}%, mean tokens
{res['first_steps']['mean_tokens']:.0f} to {res['last_steps']['mean_tokens']:.0f}.

The benchmark covers three of the dataset's six question types.

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
"""
print(f"benchmark overall: accuracy {o_b['accuracy'] * 100:.2f}% -> {o_f['accuracy'] * 100:.2f}%, tokens "
      f"{o_b['mean_tokens']:.0f} -> {o_f['mean_tokens']:.0f}, cut off {o_b['truncated_rate'] * 100:.2f}% -> {o_f['truncated_rate'] * 100:.2f}%")
(run / "MODEL_CARD.md").write_text(card, encoding="utf-8")
if a.no_push:
    raise SystemExit(0)

from huggingface_hub import HfApi  # noqa: E402

tmp = Path(tempfile.mkdtemp()) / "repo"
tmp.mkdir()
shutil.copy(run / "adapter" / "adapter_model.safetensors", tmp / "adapter_model.safetensors")
cfg = json.loads((run / "adapter" / "adapter_config.json").read_text())
cfg["base_model_name_or_path"] = a.base_model_id
(tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
(tmp / "README.md").write_text(card, encoding="utf-8")
(tmp / "eval_summary.json").write_text(json.dumps(summary, indent=1))
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo_id, private=not a.public, exist_ok=True)
info = api.upload_folder(folder_path=str(tmp), repo_id=a.repo_id, commit_message="Upload RLVR LoRA adapter (Rhea run)")
print("uploaded:", getattr(info, "commit_url", info))
print("private:", api.model_info(a.repo_id).private)
for f in api.list_repo_tree(a.repo_id):
    print(" ", f.path, getattr(f, "size", ""))
