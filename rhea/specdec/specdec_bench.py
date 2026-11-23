#!/usr/bin/env python
"""Speculative decoding benchmark: the target model alone vs the target with each draft model, on the cricket eval questions.

One independent process per GPU (RANK and WORLD from the environment, GPU chosen with CUDA_VISIBLE_DEVICES;
no torch.distributed). Every rank loads the target (optionally with a LoRA adapter merged) and all drafts, takes
questions[rank::world], and answers each question once per configuration, one prompt at a time:
  baseline              target only, one token per target pass
  <draft>-g<gamma>      target + draft: the draft proposes <gamma> tokens, the target checks them in one pass
  hf-generate           (--hf_reference) target only through transformers' generate(), to check this loop's speed
Same prompts, sampling settings and seed for every configuration.

Why a hand-written loop: Qwen3.5-family models are "stateful" (their linear-attention layers keep a recurrent state
that cannot be cropped), and transformers refuses assisted generation for them. Here a rejected round is undone by
restoring a snapshot of the cache taken before the round:
  - target: the snapshot is restored and the accepted tokens are simply re-fed, together with the next round's
    draft tokens, in that round's single pass (no extra pass; after --max_pending such tokens the state is advanced).
  - draft: the snapshot is restored and the accepted tokens are re-fed in one draft pass.
Acceptance uses speculative sampling (accept x with probability min(1, p(x)/q(x)), else resample from max(0, p - q)),
on the top-k/top-p filtered distributions, so the output distribution is the target's own. With --greedy a draft token
is accepted exactly when it is the target's argmax.

Measured per answer: wall seconds, tokens, correctness, target passes, draft passes, draft tokens proposed and accepted.
Derived per configuration: tokens/s, speed-up over baseline, acceptance rate, tokens per target pass.

Records: <out_dir>/records_rank<R>.jsonl (resumable). Summary: <out_dir>/summary.json. W&B: one run per benchmark.
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cricket_qa as Q  # noqa: E402
import sampling  # noqa: E402

RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD", 1))

# Precaution, not a measured fix: multi-token passes use transformers' plain PyTorch gated-delta-rule instead of the
# fla package's kernel, in case that kernel specialises on sequence length (speculative decoding feeds a different
# number of tokens on almost every target pass). Single-token passes keep the fused kernel.
import importlib  # noqa: E402

for _name in ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
    try:
        _mod = importlib.import_module(_name)
    except ImportError:
        continue
    _f = getattr(_mod, "torch_chunk_gated_delta_rule", None)
    if _f is not None and hasattr(_f, "__wrapped__"):
        _mod.torch_chunk_gated_delta_rule = _f.__wrapped__


def log(m):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--target_id", default="")
p.add_argument("--adapter", default="", help="LoRA adapter merged into the target; empty = base target")
p.add_argument("--drafts", default="", help="name=dir,name=dir")
p.add_argument("--data_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--run_name", default="specdec")
p.add_argument("--limit", type=int, default=200, help="questions in total (spread over the GPUs); 0 = all 600")
p.add_argument("--max_new_tokens", type=int, default=1024)
p.add_argument("--gammas", default="8", help="draft tokens per round, colon-separated, e.g. 4:8:16")
p.add_argument("--max_pending", type=int, default=48, help="re-fed tokens the target may carry before its state is advanced")
p.add_argument("--greedy", action="store_true", help="greedy decoding instead of the model card's thinking-mode sampling")
p.add_argument("--hf_reference", action="store_true", help="also time the target through transformers generate()")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--cudnn_sdp", action="store_true", help="keep the cuDNN attention kernel (slow on every new sequence length)")
p.add_argument("--profile", action="store_true", help="diagnostic: profile one baseline and one speculative answer, then stop")
p.add_argument("--aggregate", action="store_true", help="no decoding: build summary.json and log to W&B from the records")
a = p.parse_args()

dev = torch.device("cuda:0")
# The cuDNN scaled-dot-product-attention kernel pays about 5 ms per call the first time it sees a (query, key) length.
# During decoding the key length grows by one every token, so every pass on a new prompt pays it in all 16
# full-attention layers. Measured here: 8 tokens/s with it, about 40 without, for the target alone.
if not a.cudnn_sdp:
    torch.backends.cuda.enable_cudnn_sdp(False)
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
qs = [json.loads(l) for l in open(Path(a.data_dir) / "data" / "eval-00000.jsonl", encoding="utf-8")]
if a.limit:
    qs = qs[: a.limit]
shard = qs[RANK::WORLD]

if not a.aggregate:
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    load = lambda d: AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16, device_map={"": 0},  # noqa: E731
                                                                 attn_implementation="sdpa").eval()
    target = load(a.model_dir)
    if a.adapter:
        from peft import PeftModel
        target = PeftModel.from_pretrained(target, a.adapter).merge_and_unload().eval()
        log(f"target adapter merged: {a.adapter}")
    nparams = lambda m: sum(x.numel() for x in m.parameters())  # noqa: E731
    drafts, info = {}, {"target": {"dir": a.model_dir, "id": a.target_id, "adapter": a.adapter, "params": nparams(target)}, "drafts": {}}
    for spec in filter(None, a.drafts.split(",")):
        name, d = spec.split("=", 1)
        if AutoTokenizer.from_pretrained(d).get_vocab() != tok.get_vocab():
            raise SystemExit(f"draft {name} does not share the target's vocabulary; speculative decoding needs identical token ids")
        drafts[name] = load(d)
        info["drafts"][name] = {"dir": d, "params": nparams(drafts[name])}
    gammas = [int(g) for g in a.gammas.split(":")]
    configs = [("baseline", None, 0)] + [(f"{n}-g{g}", n, g) for n in drafts for g in gammas]
    if a.hf_reference:
        configs.append(("hf-generate", None, -1))
    log(f"world {WORLD}; {len(qs)} questions; configs {[c[0] for c in configs]}; "
        f"target {info['target']['params'] / 1e9:.1f}B, drafts " + ", ".join(f"{n} {v['params'] / 1e9:.2f}B" for n, v in info["drafts"].items()))
    stops = set(sampling.eos_ids(tok))
    TOP_K, TOP_P = sampling.THINKING_SAMPLING["top_k"], sampling.THINKING_SAMPLING["top_p"]
    assert sampling.THINKING_SAMPLING["temperature"] == 1.0


    class Stream:
        """One model plus its cache. feed() consumes tokens and returns the logits of the last `keep` positions."""

        def __init__(self, model):
            self.m, self.cache, self.n, self.calls = model, None, 0, 0

        def feed(self, toks, keep=1):
            o = self.m(input_ids=torch.tensor([toks], device=dev), past_key_values=self.cache, use_cache=True)
            self.cache, self.n, self.calls = o.past_key_values, self.n + len(toks), self.calls + 1
            return o.logits[0, -keep:].float() if keep else None

        def snap(self):
            return copy.deepcopy(self.cache), self.n

        def restore(self, s):
            self.cache, self.n = s


    def warp(logits):
        """[n, vocab] logits -> the probabilities tokens are actually drawn from (top-k then top-p; one-hot if greedy)."""
        if a.greedy:
            return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        v, i = logits.topk(TOP_K, dim=-1)
        pr = v.softmax(-1)
        pr = pr * ((pr.cumsum(-1) - pr) < TOP_P)
        return torch.zeros_like(logits).scatter_(-1, i, pr / pr.sum(-1, keepdim=True))


    def draw(probs, g):
        return int(torch.multinomial(probs, 1, generator=g))


    @torch.inference_mode()
    def generate(prompt, dname, gamma, seed):
        """Returns (new tokens, stats). Stops after a stop token or max_new_tokens."""
        g = torch.Generator(device=dev).manual_seed(seed)
        T, new, st = Stream(target), [], {"proposed": 0, "accepted": 0, "rounds": 0, "readvance": 0}
        if not dname:  # baseline: one target pass per token
            lg = T.feed(prompt)
            while True:
                new.append(draw(warp(lg)[-1], g))
                if new[-1] in stops or len(new) >= a.max_new_tokens:
                    break
                lg = T.feed(new[-1:])
            return new, {**st, "target_calls": T.calls, "draft_calls": 0}
        D, seq, n0 = Stream(drafts[dname]), list(prompt), len(prompt)
        T.feed(prompt[:-1], keep=0)  # prefill both models once, so that a rejected round never re-reads the prompt
        D.feed(prompt[:-1], keep=0)
        while True:
            k_max = min(gamma, a.max_new_tokens - (len(seq) - n0))
            snap_d, xs, qs_ = D.snap(), [], []
            lg = D.feed(seq[D.n:])
            while True:  # the draft proposes up to k_max tokens, one pass each
                qs_.append(warp(lg)[-1])
                xs.append(draw(qs_[-1], g))
                if xs[-1] in stops or len(xs) >= k_max:
                    break
                lg = D.feed(xs[-1:])
            k = len(xs)
            snap_t = T.snap()
            P = warp(T.feed(seq[T.n:] + xs, keep=k + 1))  # row i: target distribution for the token after xs[:i]
            acc, fix = 0, None
            for i in range(k):
                px, qx = P[i, xs[i]], qs_[i][xs[i]]
                if torch.rand((), device=dev, generator=g) < px / qx:
                    acc += 1
                else:
                    resid = (P[i] - qs_[i]).clamp(min=0)
                    fix = draw(resid / resid.sum(), g)
                    break
            st["proposed"] += k
            st["accepted"] += acc
            st["rounds"] += 1
            if fix is None:  # everything accepted: the same pass also gives the next token for free
                seq += xs
                if xs[-1] not in stops and len(seq) - n0 < a.max_new_tokens:
                    seq.append(draw(P[k], g))
            else:
                seq += xs[:acc] + [fix]
                T.restore(snap_t)  # the accepted tokens are re-fed with the next round
                if acc < k - 1:  # the draft consumed xs[:k-1]; keep its state only if that is exactly what was accepted
                    D.restore(snap_d)
            new = seq[n0:]
            if any(t in stops for t in new) or len(new) >= a.max_new_tokens:
                break
            if len(seq) - 1 - T.n > a.max_pending:
                T.feed(seq[T.n:-1], keep=0)
                st["readvance"] += 1
        cut = next((i for i, t in enumerate(new) if t in stops), None)
        new = new[: a.max_new_tokens] if cut is None else new[: cut + 1]
        return new, {**st, "target_calls": T.calls, "draft_calls": D.calls}


    @torch.inference_mode()
    def hf_generate(prompt, seed):
        ids = torch.tensor([prompt], device=dev)
        torch.manual_seed(seed)
        kw = dict(do_sample=False) if a.greedy else dict(do_sample=True, **sampling.THINKING_SAMPLING)
        o = target.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new_tokens,
                            eos_token_id=sorted(stops), pad_token_id=tok.pad_token_id, use_cache=True, **kw)
        new = o[0, len(prompt):].tolist()
        cut = next((i for i, t in enumerate(new) if t in stops), None)
        new = new if cut is None else new[: cut + 1]
        return new, {"proposed": 0, "accepted": 0, "rounds": 0, "readvance": 0, "target_calls": len(new), "draft_calls": 0}


    def answer(q, cfg, seed):
        name, dname, gamma = cfg
        prompt = tok(sampling.prompt_text(tok, q["messages"]), add_special_tokens=False)["input_ids"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new, st = hf_generate(prompt, seed) if gamma < 0 else generate(prompt, dname, gamma, seed)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        finished = new[-1] in stops
        text = tok.decode(new[:-1] if finished else new, skip_special_tokens=False)
        pred = Q.parse_answer(text)
        return {"id": q["id"], "qtype": q["qtype"], "config": name, "draft": dname or "", "gamma": max(gamma, 0), "sec": sec,
                "n_tokens": len(new), "prompt_tokens": len(prompt), "finished": finished, "pred": pred, "gold": q["gold"],
                "correct": bool(finished and Q.is_correct(pred, q["gold"], q["answer_kind"])), **st, "text": text}


    rec_path = out / f"records_rank{RANK}.jsonl"
    done = set()
    if rec_path.exists():
        done = {(r["id"], r["config"]) for r in map(json.loads, filter(str.strip, open(rec_path, encoding="utf-8")))}
    if shard:  # warm-up: compile kernels for every configuration before timing anything
        keep, a.max_new_tokens = a.max_new_tokens, 48
        for cfg in configs:
            answer(shard[0], cfg, 0)
        a.max_new_tokens = keep
        log("warm-up done")
    if a.profile:
        prompt_ = tok(sampling.prompt_text(tok, shard[0]["messages"]), add_special_tokens=False)["input_ids"]

        @torch.inference_mode()
        def raw(n_tok, pos=False, mask=False):
            ids = torch.tensor([prompt_], device=dev)
            kw = {}
            if pos:
                kw["position_ids"] = torch.arange(ids.shape[1], device=dev).view(1, 1, -1).expand(4, 1, -1)
            if mask:
                kw["attention_mask"] = torch.ones_like(ids)
            o = target(input_ids=ids, use_cache=True, **kw)
            cache, n, seen = o.past_key_values, 0, ids.shape[1]
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            while n < n_tok:
                n += 1
                nxt = torch.tensor([[int(nxt)]], device=dev)
                if pos:
                    kw["position_ids"] = torch.full((4, 1, 1), seen, device=dev)
                if mask:
                    kw["attention_mask"] = torch.ones((1, seen + 1), dtype=torch.long, device=dev)
                o = target(input_ids=nxt, past_key_values=cache, use_cache=True, **kw)
                cache, seen = o.past_key_values, seen + 1
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            return n

        def t_(name, fn):
            for _ in range(2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                n = fn()
                torch.cuda.synchronize()
                log(f"PROFILE {name}: {n / (time.perf_counter() - t0):.1f} tok/s")

        t_("raw loop", lambda: raw(64))
        t_("Stream baseline", lambda: len(generate(prompt_, None, 0, 1)[0]))
        t_("hf generate", lambda: len(hf_generate(prompt_, 1)[0]))
        for q2 in shard[1:2]:  # a prompt whose lengths have not been seen yet
            p2 = tok(sampling.prompt_text(tok, q2["messages"]), add_special_tokens=False)["input_ids"]
            t_("new prompt, Stream baseline", lambda: len(generate(p2, None, 0, 1)[0]))
            for cfg in configs[1:-1]:
                t_(f"new prompt, {cfg[0]}", lambda: len(generate(p2, cfg[1], cfg[2], 1)[0]))
        raise SystemExit(0)
    t_start = time.time()
    for qi, q in enumerate(shard):
        recs = [answer(q, cfg, a.seed * 100003 + qi * WORLD + RANK) for cfg in configs if (q["id"], cfg[0]) not in done]
        with open(rec_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if recs and (qi % 5 == 0 or qi == len(shard) - 1):
            base = next((r for r in recs if r["config"] == "baseline"), None)
            log(f"question {qi + 1}/{len(shard)} on rank 0, {(time.time() - t_start) / 60:.1f} min: " + "; ".join(
                f"{r['config']} {r['n_tokens'] / r['sec']:.0f} tok/s" + (f" ({r['n_tokens'] / r['sec'] / (base['n_tokens'] / base['sec']):.2f}x)" if base and r is not base else "")
                for r in recs))

    (out / f"models_rank{RANK}.json").write_text(json.dumps({"info": info, "configs": configs}), encoding="utf-8")
    log(f"rank {RANK} finished its {len(shard)} questions")
    raise SystemExit(0)


# ---- aggregate: summary, W&B ----
WORLD = len(list(out.glob("records_rank*.jsonl")))  # the aggregate step runs on one node; count the ranks that wrote records
_m = json.loads((out / "models_rank0.json").read_text(encoding="utf-8"))
info, configs = _m["info"], [tuple(c) for c in _m["configs"]]
recs = [json.loads(l) for f in sorted(out.glob("records_rank*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
by = defaultdict(list)
for r in recs:
    by[r["config"]].append(r)
base_text = {r["id"]: r["text"] for r in by["baseline"]}


def agg(rs):
    tk, sec, tc = sum(r["n_tokens"] for r in rs), sum(r["sec"] for r in rs), sum(r["target_calls"] for r in rs)
    prop = sum(r["proposed"] for r in rs)
    return {"answers": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs), "mean_tokens": tk / len(rs),
            "truncated_rate": sum(not r["finished"] for r in rs) / len(rs), "tokens_per_s": tk / sec, "sec_per_answer": sec / len(rs),
            "tokens_per_target_pass": tk / max(1, tc), "draft_passes_per_token": sum(r["draft_calls"] for r in rs) / max(1, tk),
            "acceptance": sum(r["accepted"] for r in rs) / prop if prop else None,
            "same_text_as_baseline": sum(r["text"] == base_text.get(r["id"]) for r in rs) / len(rs)}


rows = []
for name, dname, gamma in configs:
    if not by[name]:
        continue
    m = agg(by[name])
    m.update(config=name, draft=dname or "", gamma=max(gamma, 0), draft_params=info["drafts"].get(dname, {}).get("params"),
             by_qtype={qt: agg([r for r in by[name] if r["qtype"] == qt]) for qt in Q.QTYPES if any(r["qtype"] == qt for r in by[name])})
    rows.append(m)
b = next((m for m in rows if m["config"] == "baseline"), None)
for m in rows:
    m["speedup"] = m["tokens_per_s"] / b["tokens_per_s"] if b else None
    m["latency_ratio"] = m["sec_per_answer"] / b["sec_per_answer"] if b else None
summary = {"run": a.run_name, "args": vars(a), "world_size": WORLD, "questions": len(qs), "models": info, "configs": rows,
           "decoding": "greedy" if a.greedy else {"do_sample": True, **sampling.THINKING_SAMPLING}}
(out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
for m in rows:
    acc = "n/a" if m["acceptance"] is None else f"{m['acceptance'] * 100:.0f}%"
    log(f"[{m['config']}] {m['tokens_per_s']:.0f} tok/s, speed-up {m['speedup']:.2f}x, {m['sec_per_answer']:.2f} s/answer, "
        f"accuracy {m['accuracy'] * 100:.1f}%, tokens {m['mean_tokens']:.0f}, tokens/target pass {m['tokens_per_target_pass']:.2f}, "
        f"acceptance {acc}, same text as baseline {m['same_text_as_baseline'] * 100:.0f}% (n={m['answers']})")

if os.environ.get("WANDB_MODE") != "disabled" and os.environ.get("WANDB_API_KEY"):
    import wandb
    wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "cricket-specdec-rhea"), name=a.run_name, dir=os.environ.get("WANDB_DIR"),
                    config={**vars(a), "world_size": WORLD, "models": info})
    cols = ["config", "draft", "gamma", "answers", "tokens_per_s", "speedup", "sec_per_answer", "latency_ratio", "accuracy",
            "mean_tokens", "truncated_rate", "tokens_per_target_pass", "acceptance", "draft_passes_per_token", "draft_params",
            "same_text_as_baseline"]
    wb.log({"specdec/results": wandb.Table(columns=cols, data=[[m[c] for c in cols] for m in rows])})
    for m in rows:
        for k in ("tokens_per_s", "speedup", "sec_per_answer", "accuracy", "mean_tokens", "tokens_per_target_pass", "acceptance"):
            if m[k] is not None:
                wb.summary[f"{m['config']}/{k}"] = m[k]
    (out / "wandb_run.txt").write_text(f"{wb.project}/{wb.id}\n{wb.url}\n")
    log(f"logged to W&B run {wb.id}")
    wb.finish()
log("specdec benchmark finished")
