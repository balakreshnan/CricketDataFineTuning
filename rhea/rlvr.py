#!/usr/bin/env python
"""RLVR (reinforcement learning with verifiable rewards) for Qwen3.8-27B on the cricket questions. LoRA, torchrun.

One process per GPU. Every step, on every GPU:
  1. rollouts : the current policy answers Q questions, G samples each (thinking mode, plain temperature-1 sampling)
  2. reward   : the exact-answer checker scores each sample: finished and correct -> 1 - len_coef * tokens/max_new_tokens,
                otherwise 0 (wrong, no answer, or cut off at the token cap)
  3. advantage: reward minus the mean reward of that question's G samples (group baseline, as in GRPO)
  4. update   : policy gradient, loss = advantage * mean token NLL of the sampled answer; gradients are averaged over all
                GPUs by DDP, clipped, one AdamW step
Only the LoRA weights train. Questions come from the dataset's training split (gold answers are in the rows).
Resumable: checkpoints every --save_steps; --deadline (epoch seconds) saves and stops; re-submitting continues.
Metrics go to W&B (rank 0) and to <out_dir>/metrics.jsonl.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402
import sampling  # noqa: E402

RANK, LOCAL_RANK = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
QTYPES = ["run_rate", "chase_rate", "milestone", "projection", "strike_rate", "legal_balls"]
LORA_TARGETS = (r"model\.language_model\.layers\.\d+\."
                r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")


def log(m):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--data_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--run_name", default="cricket-rlvr")
p.add_argument("--steps", type=int, default=120)
p.add_argument("--questions_per_gpu", type=int, default=4)
p.add_argument("--group", type=int, default=8, help="samples per question")
p.add_argument("--max_new_tokens", type=int, default=1024)
p.add_argument("--len_coef", type=float, default=0.1, help="reward for a correct answer = 1 - len_coef * tokens/max_new_tokens")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup_steps", type=int, default=10)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--train_batch", type=int, default=8, help="sequences per backward pass")
p.add_argument("--lora_r", type=int, default=32)
p.add_argument("--lora_alpha", type=int, default=64)
p.add_argument("--save_steps", type=int, default=20)
p.add_argument("--pool", type=int, default=100_000, help="training rows read to draw questions from")
p.add_argument("--deadline", type=float, default=float(os.environ.get("TRAIN_DEADLINE", 0)))
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"), timeout=timedelta(hours=2))
WORLD, dev = dist.get_world_size(), torch.device(f"cuda:{LOCAL_RANK}")
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)

# ---------------- questions: same shuffled pool on every rank, disjoint slices per rank and step ----------------
rows = []
for f in sorted(glob.glob(f"{a.data_dir}/data/train-*.jsonl")):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"id": r["id"], "qtype": r["qtype"], "messages": r["messages"][:2], "gold": r["gold_answer"],
                         "kind": r["answer_kind"]})
            if len(rows) >= a.pool:
                break
    if len(rows) >= a.pool:
        break
random.Random(a.seed).shuffle(rows)
need = a.steps * a.questions_per_gpu * WORLD
assert need <= len(rows), f"need {need} questions, pool has {len(rows)}"

tok = AutoTokenizer.from_pretrained(a.model_dir)
model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map={"": LOCAL_RANK},
                                                    attn_implementation="sdpa")
model.enable_input_require_grads()          # needed for gradient checkpointing with a frozen base
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

ckpts = sorted(glob.glob(str(out / "ckpt" / "step-*")), key=lambda s: int(s.rsplit("-", 1)[1]))
start = 0
if ckpts:
    model = PeftModel.from_pretrained(model, ckpts[-1], is_trainable=True)
    start = int(ckpts[-1].rsplit("-", 1)[1])
else:
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
if RANK == 0:
    model.print_trainable_parameters()
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
if ckpts and (Path(ckpts[-1]) / "optimizer.pt").exists():
    opt.load_state_dict(torch.load(Path(ckpts[-1]) / "optimizer.pt", map_location=dev))
model._ddp_params_and_buffers_to_ignore = [k for k, q in model.named_parameters() if not q.requires_grad]
ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
stops = sampling.eos_ids(tok)
eos = tok.convert_tokens_to_ids("<|im_end|>")
gen_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new_tokens,
                           num_return_sequences=a.group, use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=stops)
log(f"world {WORLD}; {a.questions_per_gpu} questions x {a.group} samples per GPU = {a.questions_per_gpu * a.group * WORLD} "
    f"rollouts per step; steps {start} -> {a.steps}; resume from {ckpts[-1] if ckpts else 'scratch'}")

wb = None
if RANK == 0 and os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
    import wandb
    idf = out / "wandb_run_id.txt"
    rid = idf.read_text().strip() if idf.exists() else __import__("uuid").uuid4().hex[:8]
    idf.write_text(rid)
    wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-rlvr-rhea"), name=a.run_name, id=rid,
                    resume="allow", config=vars(a) | {"world_size": WORLD, "method": "RLVR (group-baseline policy gradient, LoRA)"})


def save(step, final=False):
    if RANK != 0:
        return
    d = out / ("adapter" if final else f"ckpt/step-{step}")
    model.save_pretrained(str(d))
    if not final:
        torch.save(opt.state_dict(), d / "optimizer.pt")
        for old in sorted(glob.glob(str(out / "ckpt" / "step-*")), key=lambda s: int(s.rsplit("-", 1)[1]))[:-3]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)
    log(f"saved {d}")


@torch.inference_mode()
def rollouts(qs):
    model.eval()
    prompts = [tok(sampling.prompt_text(tok, q["messages"]), add_special_tokens=False)["input_ids"] for q in qs]
    n = max(len(x) for x in prompts)
    ids = torch.full((len(qs), n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(qs), n), dtype=torch.long)
    for i, x in enumerate(prompts):                    # left padding
        ids[i, n - len(x):], mask[i, n - len(x):] = torch.tensor(x), 1
    gen = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), generation_config=gen_cfg)
    recs = []
    for j, toks in enumerate(gen[:, n:].tolist()):
        q = qs[j // a.group]
        cut = next((k for k, t in enumerate(toks) if t in stops), None)
        finished = cut is not None
        comp = toks[:cut] if finished else [t for t in toks if t != tok.pad_token_id]
        correct = bool(finished and Q.is_correct(Q.parse_answer(tok.decode(comp, skip_special_tokens=False)), q["gold"], q["kind"]))
        reward = 1.0 - a.len_coef * len(comp) / a.max_new_tokens if correct else 0.0
        recs.append({"g": j // a.group, "qtype": q["qtype"], "prompt": prompts[j // a.group],
                     "comp": comp + ([eos] if finished else []), "n": len(comp), "finished": finished, "correct": correct,
                     "reward": reward})
    for g in range(len(qs)):
        grp = [r for r in recs if r["g"] == g]
        m = sum(r["reward"] for r in grp) / len(grp)
        for r in grp:
            r["adv"] = r["reward"] - m
    return recs


def update(recs):
    """Same number of backward passes on every rank (DDP averages the gradients at each one)."""
    model.train()
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["prompt"]) + len(recs[i]["comp"]))
    total = 0.0
    for b in range(0, len(order), a.train_batch):
        batch = [recs[i] for i in order[b:b + a.train_batch]]
        n = max(len(r["prompt"]) + len(r["comp"]) for r in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, r in enumerate(batch):
            x = r["prompt"] + r["comp"]
            ids[i, :len(x)], mask[i, :len(x)] = torch.tensor(x), 1
            lab[i, len(r["prompt"]):len(x)] = torch.tensor(r["comp"])
        ids, lab, mask = ids.to(dev), lab.to(dev), mask.to(dev)
        logits = ddp(input_ids=ids, attention_mask=mask).logits[:, :-1]
        tgt = lab[:, 1:]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none",
                              ignore_index=-100).view(tgt.shape)
        seq_nll = nll.sum(1) / (tgt != -100).sum(1).clamp(min=1)
        adv = torch.tensor([r["adv"] for r in batch], device=dev, dtype=seq_nll.dtype)
        loss = (adv * seq_nll).sum() / len(recs)        # minimise A * NLL  ==  maximise A * log-prob
        loss.backward()
        total += float(loss)
        del logits, nll, loss
    return total


def reduce(vals):
    t = torch.tensor(vals, device=dev, dtype=torch.float64)
    dist.all_reduce(t)
    return t.tolist()


per_step = a.questions_per_gpu * WORLD
t_run = time.time()
for step in range(start + 1, a.steps + 1):
    stop = torch.tensor([int(bool(a.deadline) and time.time() > a.deadline)], device=dev)
    dist.broadcast(stop, src=0)
    if stop.item():
        log(f"deadline reached before step {step}: saving and stopping")
        save(step - 1)
        break
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1.0, step / max(1, a.warmup_steps))
    base = (step - 1) * per_step + RANK * a.questions_per_gpu
    qs = rows[base: base + a.questions_per_gpu]
    t0 = time.time()
    recs = rollouts(qs)
    t1 = time.time()
    opt.zero_grad(set_to_none=True)
    loss = update(recs)
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
    opt.step()
    t2 = time.time()

    groups = [[r for r in recs if r["g"] == g] for g in range(len(qs))]
    vals = [len(recs), sum(r["correct"] for r in recs), sum(r["finished"] for r in recs), sum(r["n"] for r in recs),
            sum(r["reward"] for r in recs), sum(0 < sum(r["correct"] for r in g) < len(g) for g in groups), len(groups),
            sum(all(r["correct"] for r in g) for g in groups), loss, t1 - t0, t2 - t1,
            sum(r["n"] for r in recs if r["correct"])]
    for qt in QTYPES:
        vals += [sum(r["qtype"] == qt for r in recs), sum(r["correct"] for r in recs if r["qtype"] == qt)]
    v = reduce(vals)
    if RANK == 0:
        n = v[0]
        m = {"step": step, "rollouts": int(n), "reward_mean": v[4] / n, "rollout_accuracy": v[1] / n,
             "finished_rate": v[2] / n, "truncated_rate": 1 - v[2] / n, "mean_tokens": v[3] / n,
             "mean_tokens_correct": v[11] / max(1, v[1]), "groups_mixed_rate": v[5] / v[6], "groups_all_correct_rate": v[7] / v[6],
             "loss": v[8] / WORLD, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "sec_rollout": v[9] / WORLD,
             "sec_update": v[10] / WORLD, "sec_step": t2 - t0, "questions_seen": step * per_step,
             "elapsed_h": (time.time() - t_run) / 3600}
        for i, qt in enumerate(QTYPES):
            if v[12 + 2 * i]:
                m[f"accuracy/{qt}"] = v[13 + 2 * i] / v[12 + 2 * i]
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(m) + "\n")
        if wb:
            wb.log(m, step=step)
        log(f"step {step}/{a.steps}: acc {m['rollout_accuracy']:.3f} reward {m['reward_mean']:.3f} tokens {m['mean_tokens']:.0f} "
            f"cut-off {m['truncated_rate']:.3f} mixed groups {m['groups_mixed_rate']:.2f} loss {m['loss']:+.4f} gnorm {gn:.3f} "
            f"| rollout {m['sec_rollout']:.0f}s update {m['sec_update']:.0f}s")
    if step % a.save_steps == 0 and step < a.steps:
        save(step)
else:
    save(a.steps, final=True)
    if RANK == 0:
        ms = [json.loads(l) for l in open(out / "metrics.jsonl")]
        k = max(1, min(10, len(ms) // 2))
        avg = lambda xs, key: sum(x[key] for x in xs) / len(xs)
        json.dump({"finished": True, "steps": a.steps, "world_size": WORLD, "rollouts_total": sum(x["rollouts"] for x in ms),
                   "first_steps": {key: avg(ms[:k], key) for key in ("rollout_accuracy", "reward_mean", "mean_tokens", "truncated_rate")},
                   "last_steps": {key: avg(ms[-k:], key) for key in ("rollout_accuracy", "reward_mean", "mean_tokens", "truncated_rate")},
                   "window": k, "train_hours": sum(x["sec_step"] for x in ms) / 3600, "args": vars(a)},
                  open(out / "train_result.json", "w"), indent=1)
        log(f"finished; adapter -> {out / 'adapter'}")
if wb:
    wb.finish()
dist.barrier()
dist.destroy_process_group()
