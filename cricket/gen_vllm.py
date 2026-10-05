#!/usr/bin/env python
"""vLLM sampling worker - one process per GPU, each owning shard i of n (data parallel, no NCCL).

Samples k thinking-mode completions per question, verifies them against the exact answers and appends
records to {out_dir}/{tag}_rank{shard}.jsonl in the same format as sampling.py (so collect/metrics and
the dataset builder work unchanged). Resumable: questions already in the shard file are skipped; a
wall-clock deadline stops between chunks. Runs inside the NGC vLLM container (needs only vllm + stdlib).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402

THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}  # Qwen3.8 model card, thinking mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--questions", required=True, help="jsonl with id, qtype, messages, gold|gold_answer, answer_kind")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="train")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=6144)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256, help="questions per generate() call (resume granularity)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline", type=float, default=float(os.environ.get("TRAIN_DEADLINE", 0)))
    args = p.parse_args()


    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [shard {args.shard}/{args.num_shards}] {msg}", flush=True)


    qs = Q.read_jsonl(args.questions)
    if args.limit:
        qs = qs[: args.limit]
    for q in qs:
        q.setdefault("gold", q.get("gold_answer"))
    out = Path(args.out_dir) / f"{args.tag}_rank{args.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    shard = [q for i, q in enumerate(qs) if i % args.num_shards == args.shard and q["id"] not in done]
    log(f"{len(shard)} questions to sample (k={args.k}), {len(done)} already done; model {args.model_dir}")
    if not shard:
        return

    from vllm import LLM, SamplingParams  # noqa: E402

    t0 = time.time()
    llm = LLM(model=args.model_dir, tensor_parallel_size=1, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=args.seed,
              limit_mm_per_prompt={"image": 0, "video": 0})  # text only: skip the vision encoder
    tok = llm.get_tokenizer()
    log(f"engine ready in {time.time() - t0:.0f}s")
    sp = SamplingParams(n=args.k, max_tokens=args.max_tokens, seed=args.seed + args.shard, skip_special_tokens=False,
                        **THINKING_SAMPLING)

    t0, n_tok, n_done = time.time(), 0, 0
    for c in range(0, len(shard), args.chunk):
        if args.deadline and time.time() > args.deadline:
            log(f"deadline reached, {len(shard) - c} questions left - resubmit to continue")
            break
        batch = shard[c:c + args.chunk]
        prompts = [tok.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                   for q in batch]
        results = llm.generate(prompts, sp, use_tqdm=False)
        with open(out, "a", encoding="utf-8") as f:
            for q, res in zip(batch, results):
                for j, o in enumerate(res.outputs):
                    text = o.text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                    finished = o.finish_reason == "stop"
                    pred = Q.parse_answer(text)
                    n_tok += len(o.token_ids)
                    f.write(json.dumps({"id": q["id"], "qtype": q["qtype"], "sample": j, "text": text,
                                        "n_tokens": len(o.token_ids), "finished": finished, "pred": pred, "gold": q["gold"],
                                        "correct": bool(finished and Q.is_correct(pred, q["gold"], q["answer_kind"]))},
                                       ensure_ascii=False) + "\n")
        n_done += len(batch)
        el = time.time() - t0
        log(f"{n_done}/{len(shard)} questions, {el / 60:.1f} min, {n_tok / el:,.0f} generated tok/s on this GPU")
    log("done")


# vLLM starts its engine in a spawned child that re-imports this file: everything must sit behind the guard.
if __name__ == "__main__":
    main()
