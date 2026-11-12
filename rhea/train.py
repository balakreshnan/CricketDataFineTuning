#!/usr/bin/env python
"""LoRA SFT of Qwen3.8-27B on the cricket reasoning traces (HF dataset layout), torchrun + DDP, one process per GPU.

Dataset rows (data/train-*.jsonl): messages = [system, user, assistant], plus reasoning / response fields. The
training sequence is exactly what the model produces at inference in thinking mode:
  prompt  = chat_template(messages[:2], add_generation_prompt=True, enable_thinking=True)   (ends with "<think>\\n")
  target  = reasoning + "\\n</think>\\n\\n" + response + "<|im_end|>"        (loss only on the target)
Resumable: re-submitting the same run dir continues from the newest checkpoint. --deadline (epoch s) saves and stops.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

RANK, LOCAL_RANK = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))


def log(m):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True, help="dir holding data/train-*.jsonl and data/test-*.jsonl")
    p.add_argument("--out_dir", required=True, help="run dir: checkpoints, adapter, logs")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--run_name", default="cricket-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=6144)
    p.add_argument("--dev_rows", type=int, default=1000)
    p.add_argument("--train_limit", type=int, default=0, help="smoke/benchmark: first N rows only")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--group_by_length", action="store_true")
    p.add_argument("--hub_model_id", default=None, help="push the final adapter here (needs HF_TOKEN)")
    p.add_argument("--deadline", type=float, default=float(os.environ.get("TRAIN_DEADLINE", 0)))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_rows(data_dir, split, limit):
    rows = []
    for f in sorted(glob.glob(f"{data_dir}/data/{split}-*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def prompt_of(tok, r):
    return tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=True)


def tokenize_cached(tok, a, split, limit):
    """Tokenize once (rank 0) into an npz cache on Lustre; every rank loads it."""
    cache = Path(a.cache_dir) / f"tok-{split}-L{a.max_len}-n{limit}.npz"
    if RANK == 0 and not cache.exists():
        t0, ids, offs, plens, dropped, n = time.time(), [], [0], [], 0, 0
        rows = read_rows(a.data_dir, split, limit)
        for i in range(0, len(rows), 4096):
            b = rows[i:i + 4096]
            P = tok([prompt_of(tok, r) for r in b], add_special_tokens=False)["input_ids"]
            C = tok([f"{r['reasoning']}\n</think>\n\n{r['response']}<|im_end|>" for r in b],
                    add_special_tokens=False)["input_ids"]
            for p, c in zip(P, C):
                n += 1
                if len(p) + len(c) > a.max_len:
                    dropped += 1
                    continue
                ids.extend(p + c)
                offs.append(offs[-1] + len(p) + len(c))
                plens.append(len(p))
        if rows:
            log(f"prompt tail: {prompt_of(tok, rows[0])[-40:]!r}  (must end with '<think>\\n')")
        Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = Path(str(cache) + ".tmp.npz")
        np.savez(tmp, ids=np.asarray(ids, np.int32), offs=np.asarray(offs, np.int64),
                 plens=np.asarray(plens, np.int32))
        tmp.replace(cache)
        log(f"tokenized {n:,} {split} rows ({dropped} dropped > {a.max_len} tokens) in {time.time() - t0:.0f}s")
    dist.barrier()
    z = np.load(cache)
    return z["ids"], z["offs"], z["plens"]


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, offs, plens, index):
        self.ids, self.offs, self.plens, self.index = ids, offs, plens, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = self.index[i]
        x = self.ids[self.offs[j]:self.offs[j + 1]].astype(np.int64)
        y = x.copy()
        y[: self.plens[j]] = -100
        return {"input_ids": x, "labels": y}


def collate(pad):
    def fn(b):
        n = max(len(r["input_ids"]) for r in b)
        ids = torch.full((len(b), n), pad, dtype=torch.long)
        lab = torch.full((len(b), n), -100, dtype=torch.long)
        m = torch.zeros((len(b), n), dtype=torch.long)
        for i, r in enumerate(b):
            k = len(r["input_ids"])
            ids[i, :k], lab[i, :k], m[i, :k] = torch.as_tensor(r["input_ids"]), torch.as_tensor(r["labels"]), 1
        return {"input_ids": ids, "attention_mask": m, "labels": lab}
    return fn


class Deadline(TrainerCallback):
    def __init__(self, t):
        self.t = t

    def on_step_end(self, args, state, control, **kw):
        if not self.t:
            return
        f = torch.tensor([int(time.time() > self.t)], device=f"cuda:{LOCAL_RANK}")
        dist.broadcast(f, src=0)
        if f.item():
            log(f"deadline reached at step {state.global_step}: saving and stopping")
            control.should_training_stop = True
            control.should_save = True


# language model only: full attention, Gated-DeltaNet linear attention and MLP projections
LORA_TARGETS = (r"model\.language_model\.layers\.\d+\."
                r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


def main():
    a = parse_args()
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids, offs, plens = tokenize_cached(tok, a, "train", a.train_limit)
    n = len(plens)
    perm = np.random.RandomState(a.seed).permutation(n)
    dev_n = min(a.dev_rows, n // 20)
    train_ds, dev_ds = DS(ids, offs, plens, perm[dev_n:]), DS(ids, offs, plens, perm[:dev_n])
    log(f"world {dist.get_world_size()}; train {len(train_ds):,} rows, dev {len(dev_ds):,}")

    model = AutoModelForImageTextToText.from_pretrained(
        a.model_dir, dtype=torch.bfloat16, device_map={"": LOCAL_RANK}, attn_implementation="sdpa")
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    if not a.no_grad_ckpt:
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    if RANK == 0:
        model.print_trainable_parameters()
    # DDP only all-reduces the LoRA grads, not the frozen 27B base
    model._ddp_params_and_buffers_to_ignore = [k for k, p in model.named_parameters() if not p.requires_grad]
    has_wandb = bool(os.environ.get("WANDB_API_KEY")) and os.environ.get("WANDB_MODE") != "disabled"
    gbs = a.per_device_batch * dist.get_world_size() * a.grad_accum
    steps = a.max_steps if a.max_steps > 0 else int(np.ceil(np.ceil(len(train_ds) / gbs) * a.epochs))
    log(f"global batch {gbs}, {steps} optimizer steps, mean length {offs[-1] / n:.0f} tokens")
    # transformers 5.x: warmup_ratio / group_by_length are gone (warmup_steps, train_sampling_strategy)
    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), run_name=a.run_name, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.per_device_batch, per_device_eval_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(a.warmup_ratio * steps)), bf16=True, logging_steps=a.log_steps,
        logging_first_step=True, eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="steps",
        save_steps=a.save_steps, save_total_limit=3, gradient_checkpointing=not a.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length" if a.group_by_length else "random",
        label_names=["labels"], dataloader_num_workers=0,  # py3.14 forkserver cannot pickle the collate closure; data is in RAM
        remove_unused_columns=False,
        ddp_find_unused_parameters=False, report_to=["wandb"] if has_wandb else [], seed=a.seed)
    tr = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                 data_collator=collate(tok.pad_token_id if tok.pad_token_id is not None else 0),
                 callbacks=[Deadline(a.deadline)])
    ck = sorted(glob.glob(str(out / "ckpt" / "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    log(f"resuming from {ck[-1]}" if ck else "starting fresh")
    tr.train(resume_from_checkpoint=ck[-1] if ck else None)
    if a.deadline and time.time() > a.deadline:
        log("stopped at the deadline: resubmit the same run to continue")
        dist.destroy_process_group()
        return
    tr.save_model(str(out / "adapter"))
    ev = tr.evaluate()
    if RANK == 0:
        json.dump({"eval": ev, "args": vars(a)}, open(out / "train_result.json", "w"), indent=1)
        log(f"final dev loss {ev.get('eval_loss')}; adapter -> {out / 'adapter'}")
        if a.hub_model_id and os.environ.get("HF_TOKEN"):
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(a.hub_model_id, private=True, exist_ok=True)
            api.upload_folder(folder_path=str(out / "adapter"), repo_id=a.hub_model_id)
            log(f"pushed adapter to {a.hub_model_id} (private)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
