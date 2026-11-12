"""Distributed thinking-mode sampling with Qwen3.8-27B - builds the reasoning dataset and runs the evals.

Every rank takes questions[rank::world], samples k completions per question, verifies each against the
exact answer, and appends one JSON line per completion to its own file. Resumable: questions already in a
rank's file are skipped, and a wall-clock deadline stops new batches so a SLURM job never dies mid-write.
"""
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

import cricket_qa as Q

# Qwen3.8 model card, thinking mode: temperature=1.0, top_p=0.95, top_k=20 (greedy decoding can loop)
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def prompt_text(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def eos_ids(tokenizer):
    return [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


@torch.inference_mode()
def sample(model, tokenizer, questions, out_dir, tag, *, k, max_new_tokens, batch_prompts, deadline=0.0,
           seed=0, log=print, on_batch=None):
    """Append k verified samples per question to {out_dir}/{tag}_rank{R}.jsonl. Returns #questions left on this rank."""
    from transformers import GenerationConfig

    rank, world = rank_world()
    out_path = Path(out_dir) / f"{tag}_rank{rank}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    shard = [q for i, q in enumerate(questions) if i % world == rank and q["id"] not in done]
    prompts = [tokenizer(prompt_text(tokenizer, q["messages"]), add_special_tokens=False)["input_ids"] for q in shard]
    order = sorted(range(len(shard)), key=lambda i: -len(prompts[i]))
    stops = eos_ids(tokenizer)
    gen_cfg = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, num_return_sequences=k, use_cache=True,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=stops, **THINKING_SAMPLING)
    torch.manual_seed(seed * 1000 + rank)
    device = next(model.parameters()).device
    model.eval()
    n_batches = math.ceil(len(order) / batch_prompts)
    t0, gen_tokens, left = time.time(), 0, len(order)
    if done:
        log(f"[{tag}] resuming: {len(done)} questions already sampled on rank 0's shard")
    for b in range(n_batches):
        if deadline and time.time() > deadline:
            log(f"[{tag}] deadline reached with {left} questions left on rank 0 - resubmit to continue")
            break
        idx = order[b * batch_prompts:(b + 1) * batch_prompts]
        n = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), n), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), n), dtype=torch.long)
        for row, i in enumerate(idx):  # left padding
            ids[row, n - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[row, n - len(prompts[i]):] = 1
        out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=gen_cfg)
        recs = []
        for j, toks in enumerate(out[:, n:].tolist()):  # rows are [q0 s0..s(k-1), q1 s0.., ...]
            q = shard[idx[j // k]]
            cut = next((p for p, t in enumerate(toks) if t in stops), None)
            finished = cut is not None
            toks = toks[:cut] if finished else toks
            text = tokenizer.decode(toks, skip_special_tokens=False)
            pred = Q.parse_answer(text)
            recs.append({"id": q["id"], "qtype": q["qtype"], "sample": j % k, "text": text, "n_tokens": len(toks),
                         "finished": finished, "pred": pred, "gold": q["gold"],
                         "correct": bool(finished and Q.is_correct(pred, q["gold"], q["answer_kind"]))})
            gen_tokens += len(toks)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        left -= len(idx)
        el = time.time() - t0
        if on_batch:
            on_batch({"batch": b + 1, "batches": n_batches, "elapsed_s": el, "tokens_per_s_rank": gen_tokens / el,
                      "batch_accuracy": sum(r["correct"] for r in recs) / len(recs),
                      "batch_mean_tokens": sum(r["n_tokens"] for r in recs) / len(recs)})
        log(f"[{tag}] batch {b + 1}/{n_batches} on rank 0: {el / 60:.1f} min, {gen_tokens / el:,.0f} tok/s/GPU, "
            f"batch acc {sum(r['correct'] for r in recs) / len(recs):.2f}, "
            f"mean {sum(r['n_tokens'] for r in recs) / len(recs):,.0f} tok, ~{el / (b + 1) * (n_batches - b - 1) / 60:.0f} min left")
    return left


def pending_everywhere(left):
    """All-reduce: total questions still pending across ranks."""
    t = torch.tensor([left], device=f"cuda:{torch.cuda.current_device()}")
    if dist.is_initialized():
        dist.all_reduce(t)
    return int(t.item())


def collect(out_dir, tag):
    recs = []
    for p in sorted(Path(out_dir).glob(f"{tag}_rank*.jsonl")):
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def metrics(records, k):
    """accuracy = mean over all samples (avg@k); pass@k = any sample correct; tokens = mean completion length."""
    by_q = defaultdict(list)
    for r in records:
        by_q[r["id"]].append(r)

    def agg(qids):
        rs = [r for q in qids for r in by_q[q]]
        if not rs:
            return None
        return {"questions": len(qids), "samples": len(rs),
                "accuracy": sum(r["correct"] for r in rs) / len(rs),
                f"pass@{k}": sum(any(r["correct"] for r in by_q[q]) for q in qids) / len(qids),
                "mean_tokens": sum(r["n_tokens"] for r in rs) / len(rs),
                "mean_tokens_correct": (sum(r["n_tokens"] for r in rs if r["correct"]) / max(1, sum(r["correct"] for r in rs))),
                "truncated_rate": sum(not r["finished"] for r in rs) / len(rs),
                "no_answer_rate": sum(r["finished"] and r["pred"] is None for r in rs) / len(rs)}

    out = {qt: agg([q for q, rs in by_q.items() if rs[0]["qtype"] == qt]) for qt in Q.QTYPES}
    out["overall"] = agg(list(by_q))
    return {qt: m for qt, m in out.items() if m}  # empty when nothing was sampled yet
