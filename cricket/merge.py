#!/usr/bin/env python
"""Merge a LoRA adapter into Qwen3.8-27B and save a full bf16 checkpoint that vLLM can load.

The non-weight files of the base snapshot (processor, chat template, tokenizer...) are copied over so the
merged directory is a drop-in replacement for the base model path.
"""
import argparse
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, max_shard_size="5GB")
for f in Path(args.model_dir).iterdir():  # processor / tokenizer / chat template / generation config
    if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"):
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
print(f"merged {args.adapter} into {args.model_dir} -> {out} in {time.time() - t0:.0f}s")
print(sorted(x.name for x in out.iterdir()))
