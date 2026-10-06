#!/usr/bin/env python
"""LoRA SFT of Qwen3.8-27B on the verified cricket reasoning traces, with base vs fine-tuned evals.

torchrun, one process per GPU, plain DDP (full bf16 replica per GPU). Single model load:
  base eval (thinking mode, k samples per held-out test question) -> SFT -> fine-tuned eval
  -> report -> W&B (training curves + eval tables) -> push adapter + model card + eval report to the hub.
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402
import report  # noqa: E402
import sampling  # noqa: E402

RANK, LOCAL_RANK = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))


def log(msg):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True, help="final dataset dir (train.jsonl, test.jsonl)")
    p.add_argument("--croot", required=True, help="project root on Lustre: runs/ and checkpoints/ go here")
    p.add_argument("--run_dir", default=None, help="default: $croot/runs/$RUN_NAME")
    p.add_argument("--hub_model_id", default=None)
    p.add_argument("--hub_revision", default=None)
    p.add_argument("--base_model_id", default="Qwen/Qwen3.8-27B")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=6144)
    p.add_argument("--dev_frac", type=float, default=0.02)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_k", type=int, default=2, help="samples per test question")
    p.add_argument("--eval_max_new_tokens", type=int, default=4096)
    p.add_argument("--eval_batch_prompts", type=int, default=16)
    p.add_argument("--eval_limit", type=int, default=0)
    p.add_argument("--skip_base_eval", action="store_true")
    p.add_argument("--no_eval", action="store_true", help="train only; evals run as separate vLLM jobs")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--adapter", default=None)
    p.add_argument("--deadline", type=float, default=float(os.environ.get("TRAIN_DEADLINE", 0)))
    p.add_argument("--seed", type=int, default=42)
    # throughput / scale (full 458K-row dataset)
    p.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (faster, more memory)")
    p.add_argument("--sampling", default="random", choices=["random", "group_by_length"],
                   help="group_by_length: batches of similar length (less padding)")
    p.add_argument("--max_steps", type=int, default=-1, help="benchmark: stop after N optimizer steps")
    p.add_argument("--train_limit", type=int, default=0, help="benchmark: use only the first N training rows")
    p.add_argument("--dev_rows", type=int, default=0, help="fixed dev-set size (0 = dev_frac of the data)")
    p.add_argument("--eval_steps", type=int, default=0, help="dev-loss interval (0 = save_steps)")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_steps", type=int, default=2)
    return p.parse_args()


class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        if not self.deadline:
            return
        flag = torch.tensor([int(time.time() > self.deadline)], device=f"cuda:{LOCAL_RANK}")
        dist.broadcast(flag, src=0)
        if flag.item():
            log(f"deadline reached at step {state.global_step}: saving and stopping")
            control.should_training_stop = True
            control.should_save = True


class JsonlLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, "time": time.time(), **logs}) + "\n")


class TokDataset(torch.utils.data.Dataset):
    """Rows stored as one flat int32 token array + offsets; labels = tokens with the prompt part masked."""

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


def load_tokenized(tok, train_path, max_len, limit, cache_dir):
    """Tokenize once (global rank 0) into a cache on Lustre; every rank loads the result.
    prompt = exact thinking-mode prompt used at generation; target = the model's own verified output + <|im_end|>."""
    st = Path(train_path).stat()
    key = f"{Path(train_path).parent.parent.name}-{st.st_size}-{int(st.st_mtime)}-L{max_len}-n{limit}"
    cache = Path(cache_dir) / f"tok-{key}.npz"
    if RANK == 0 and not cache.exists():
        t0 = time.time()
        ids, offs, plens, batch = [], [0], [], []
        stats = {"dropped": 0, "rows": 0}

        def flush():
            prompts = tok([sampling.prompt_text(tok, r["messages"][:2]) for r in batch], add_special_tokens=False)["input_ids"]
            comps = tok([r["completion_text"] + "<|im_end|>" for r in batch], add_special_tokens=False)["input_ids"]
            for pr, co in zip(prompts, comps):
                if len(pr) + len(co) > max_len:
                    stats["dropped"] += 1
                    continue
                ids.extend(pr + co)
                offs.append(offs[-1] + len(pr) + len(co))
                plens.append(len(pr))
            batch.clear()

        with open(train_path, encoding="utf-8") as f:
            for line in f:
                if limit and stats["rows"] >= limit:
                    break
                batch.append(json.loads(line))
                stats["rows"] += 1
                if len(batch) == 4096:
                    flush()
        if batch:
            flush()
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(cache) + ".tmp.npz")
        np.savez(tmp, ids=np.asarray(ids, dtype=np.int32), offs=np.asarray(offs, dtype=np.int64),
                 plens=np.asarray(plens, dtype=np.int32), dropped=np.asarray([stats["dropped"]]))
        tmp.replace(cache)
        log(f"tokenized {stats['rows']:,} rows -> {cache} in {time.time() - t0:.0f}s")
    dist.barrier()
    z = np.load(cache)
    return z["ids"], z["offs"], z["plens"], int(z["dropped"][0])


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.as_tensor(b["input_ids"]), torch.as_tensor(b["labels"]), 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return fn


def tokenize(tok, rows, max_len):
    """prompt = exact thinking-mode prompt used at generation; target = the model's own verified output + <|im_end|>."""
    out, dropped = [], 0
    for r in rows:
        p = tok(sampling.prompt_text(tok, r["messages"][:2]), add_special_tokens=False)["input_ids"]
        c = tok(r["completion_text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        out.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return out, dropped


def run_eval(model, tok, test, run_dir, tag, args):
    left = sampling.sample(model, tok, test, run_dir / "eval", tag, k=args.eval_k,
                           max_new_tokens=args.eval_max_new_tokens, batch_prompts=args.eval_batch_prompts,
                           seed=7, log=log)  # same seed for base and fine-tuned
    sampling.pending_everywhere(left)
    dist.barrier()
    if RANK != 0:
        return None
    m = sampling.metrics(sampling.collect(run_dir / "eval", tag), args.eval_k)
    (run_dir / "eval" / f"{tag}_summary.json").write_text(json.dumps(m, indent=2))
    for qt, x in m.items():
        log(f"[{tag} eval] {qt}: acc={x['accuracy']:.4f} pass@{args.eval_k}={x[f'pass@{args.eval_k}']:.4f} "
            f"tokens={x['mean_tokens']:.0f} truncated={x['truncated_rate']:.3f} (n={x['questions']})")
    return m


def wandb_eval(tag, m):
    import wandb
    for qt, x in m.items():
        wandb.summary.update({f"{tag}/{qt}/{k}": v for k, v in x.items()})


def finalize(args, run_dir, use_wandb, adapter_dir):
    comp = report.generate(run_dir, args.eval_k) if (run_dir / "eval" / "base_summary.json").exists() else None
    if use_wandb:
        import wandb
        if comp:
            t = wandb.Table(columns=["question_type", "metric", "base", "finetuned", "delta"])
            for r in comp["rows"]:
                t.add_data(r["qtype"], r["metric"], r["base"], r["finetuned"], r["delta"])
            wandb.log({"eval/base_vs_finetuned": t})
        for png in sorted((run_dir / "plots").glob("*.png")):
            wandb.log({f"plots/{png.stem}": wandb.Image(str(png))})
    if not (args.hub_model_id and adapter_dir and Path(adapter_dir).is_dir()):
        return
    from huggingface_hub import HfApi
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = args.base_model_id
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    card = ["---", f"base_model: {args.base_model_id}", "library_name: peft", "pipeline_tag: text-generation",
            "tags: [lora, reasoning, cricket, sports-analytics]", "---", "",
            f"LoRA adapter for `{args.base_model_id}` fine-tuned on verified cricket reasoning traces (T20I ball-by-ball "
            "situations from Valarmathy/CricketData; run-rate, chase-rate, milestone, projection, strike-rate and "
            "legal-ball questions with exact answers, distilled from the base model in thinking mode). Use thinking mode "
            "(`enable_thinking=True`) with temperature 1.0, top_p 0.95, top_k 20.", ""]
    if (run_dir / "train_result.json").exists():
        tr = json.loads((run_dir / "train_result.json").read_text())
        card.append(f"Training: {tr.get('train_rows', 0):,} rows, {tr.get('completed_epochs', 0):.2f} epochs, "
                    f"{tr.get('global_step')} steps, global batch {tr.get('global_batch', '?')}, "
                    f"{tr.get('world_size', '?')} Rubin GPUs, {tr.get('train_runtime', 0) / 3600:.2f} h.\n")
    if (run_dir / "report.md").exists():
        card.append((run_dir / "report.md").read_text().replace("](plots/", "](eval_report/plots/"))
    (adapter_dir / "README.md").write_text("\n".join(card) + "\n")
    api = HfApi()
    api.create_repo(args.hub_model_id, private=True, exist_ok=True)
    if args.hub_revision:
        api.create_branch(args.hub_model_id, branch=args.hub_revision, exist_ok=True)
    api.upload_folder(repo_id=args.hub_model_id, folder_path=str(adapter_dir), revision=args.hub_revision,
                      ignore_patterns=["checkpoint-*", "checkpoint-*/**"], commit_message=f"LoRA adapter, run {run_dir.name}")
    api.upload_folder(repo_id=args.hub_model_id, folder_path=str(run_dir), path_in_repo="eval_report",
                      revision=args.hub_revision,
                      allow_patterns=["report.md", "comparison.json", "config.json", "train_result.json", "metrics.jsonl",
                                      "eval/*_summary.json", "plots/*.png"], commit_message=f"Eval report, run {run_dir.name}")
    log(f"pushed adapter + eval report to https://huggingface.co/{args.hub_model_id}")


def main():
    args = parse_args()
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"), timeout=timedelta(hours=2))
    world = dist.get_world_size()
    run_name = os.environ.get("RUN_NAME", "local")
    run_dir = Path(args.run_dir or Path(args.croot) / "runs" / run_name)
    out_dir = Path(args.croot) / "checkpoints" / run_dir.name
    if RANK == 0:
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (run_dir / ("config_eval_only.json" if args.eval_only else "config.json")).write_text(json.dumps(
            {**vars(args), "world_size": world, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "nodes": os.environ.get("SLURM_JOB_NODELIST")}, indent=2))

    if not args.eval_only and (run_dir / "train_result.json").exists():  # resume-safety job after a finished run
        log(f"{run_dir / 'train_result.json'} exists - training already complete, nothing to do")
        dist.destroy_process_group()
        return

    use_wandb = os.environ.get("WANDB_MODE") != "disabled"
    if use_wandb and RANK == 0:
        import wandb
        kw = dict(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-reasoning"), name=run_dir.name, id=run_dir.name,
                  resume="allow", dir=str(run_dir), job_type="eval" if args.eval_only else "sft",
                  config={"eval_only_args": vars(args)} if args.eval_only else vars(args))
        try:
            wandb.init(**kw)
        except Exception as e:
            log(f"W&B init failed ({e}); logging offline")
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(mode="offline", **kw)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    test = [{**q, "gold": q["gold_answer"]} for q in Q.read_jsonl(Path(args.data_dir) / "test.jsonl")]
    if args.eval_limit:
        test = test[: args.eval_limit]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map={"": LOCAL_RANK}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.name_or_path = model.config._name_or_path = args.base_model_id
    log(f"world {world}; model loaded; {len(test)} test questions x {args.eval_k} samples per eval")

    from peft import LoraConfig, PeftModel, get_peft_model

    if args.eval_only:
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
        tag = "ft" if args.adapter else "base"
        m = run_eval(model, tok, test, run_dir, tag, args)
        if RANK == 0:
            if use_wandb:
                wandb_eval(tag, m)
            if tag == "ft":
                finalize(args, run_dir, use_wandb, args.adapter)
            if use_wandb:
                import wandb
                wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
        return

    if not (args.skip_base_eval or args.no_eval):
        m = run_eval(model, tok, test, run_dir, "base", args)
        if RANK == 0 and use_wandb:
            wandb_eval("base", m)
    dist.barrier()

    ids, offs, plens, dropped = load_tokenized(tok, Path(args.data_dir) / "train.jsonl", args.max_len,
                                               args.train_limit, Path(args.croot) / "cache")
    n_rows = len(plens)
    perm = torch.randperm(n_rows, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_dev = args.dev_rows or max(1, int(n_rows * args.dev_frac))
    dev, train = TokDataset(ids, offs, plens, perm[:n_dev]), TokDataset(ids, offs, plens, perm[n_dev:])
    gbs = args.per_device_batch * world * args.grad_accum
    steps = args.max_steps if args.max_steps > 0 else math.ceil(math.ceil(len(train) / gbs) * args.epochs)
    log(f"SFT: {len(train):,} train / {len(dev):,} dev rows ({dropped} over {args.max_len} tokens dropped), "
        f"global batch {gbs}, {steps} steps, mean length {(offs[-1] / n_rows):.0f} tokens, "
        f"grad ckpt {not args.no_grad_ckpt}, sampling {args.sampling}")

    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=_lora_targets(), task_type="CAUSAL_LM"))
    if RANK == 0:
        model.print_trainable_parameters()
    model._ddp_params_and_buffers_to_ignore = [n for n, p in model.named_parameters() if not p.requires_grad]
    targs = TrainingArguments(
        output_dir=str(out_dir), run_name=run_dir.name, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(args.warmup_ratio * steps)), bf16=True, gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False,
        max_steps=args.max_steps, train_sampling_strategy=args.sampling,
        logging_steps=args.log_steps, logging_first_step=True, eval_strategy="steps",
        eval_steps=args.eval_steps or args.save_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        label_names=["labels"], remove_unused_columns=False, dataloader_num_workers=2,
        report_to=["wandb"] if use_wandb else [], seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                      data_collator=collate(tok.pad_token_id), processing_class=tok,
                      callbacks=[DeadlineCallback(args.deadline), JsonlLogger(run_dir / "metrics.jsonl")])
    resume = any(out_dir.glob("checkpoint-*"))  # continuation job: pick up from the latest checkpoint
    if resume:
        log(f"resuming from the latest checkpoint in {out_dir}")
    out = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model()
    if RANK == 0:
        (run_dir / "train_result.json").write_text(json.dumps(
            {**out.metrics, "global_step": trainer.state.global_step, "completed_epochs": trainer.state.epoch,
             "train_rows": len(train), "dev_rows": len(dev), "dropped_too_long": dropped, "adapter_dir": str(out_dir),
             "global_batch": gbs, "world_size": world}, indent=2))

    if args.no_eval:
        if RANK == 0 and use_wandb:
            import wandb
            wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
        log(f"done (training only). run dir {run_dir}, adapter {out_dir}")
        return
    torch.cuda.empty_cache()
    m = run_eval(model, tok, test, run_dir, "ft", args)
    if RANK == 0:
        if use_wandb:
            wandb_eval("ft", m)
        finalize(args, run_dir, use_wandb, out_dir)
        if use_wandb:
            import wandb
            wandb.finish()
    dist.barrier()
    dist.destroy_process_group()
    log(f"done. run dir {run_dir}, adapter {out_dir}")


def _lora_targets():
    # language model only: full attention, Gated-DeltaNet linear attention and MLP projections
    return (r"model\.language_model\.layers\.\d+\."
            r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


if __name__ == "__main__":
    main()
