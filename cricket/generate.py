#!/usr/bin/env python
"""Generate the reasoning dataset: Qwen3.8-27B (thinking mode) answers every training question k times,
answers are verified against the exact values, and the shortest correct trace per question is kept.

torchrun, one process per GPU (each GPU holds a full bf16 replica). Resumable across SLURM jobs: rerun
with the same --out_dir and it continues where the last job stopped; the dataset is assembled when all
questions are done. Metrics go to W&B and $out_dir/stats.json.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402
import sampling  # noqa: E402

RANK, LOCAL_RANK = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))


def log(msg):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_dataset(questions, test_questions, records, out_dir, k, args):
    """One row per solved question: the shortest verified-correct, fully finished reasoning trace."""
    by_id = {}
    for r in records:
        by_id.setdefault(r["id"], []).append(r)
    rows, unsolved = [], []
    for q in questions:
        rs = by_id.get(q["id"], [])
        good = [r for r in rs if r["correct"]]
        if not good:
            unsolved.append(q["id"])
            continue
        best = min(good, key=lambda r: r["n_tokens"])
        reasoning, answer, _ = Q.split_reasoning(best["text"])
        rows.append({
            "id": q["id"], "qtype": q["qtype"], "question": q["messages"][1]["content"],
            "messages": q["messages"] + [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}],
            "reasoning": reasoning, "response": answer, "final_answer": best["pred"], "gold_answer": q["gold"],
            "answer_kind": q["answer_kind"], "reasoning_tokens": best["n_tokens"],
            "samples_generated": len(rs), "samples_correct": len(good), "source": q["source"],
            "completion_text": best["text"],  # exact model output after the prompt (training target); not pushed
        })
    final = Path(out_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    Q.write_jsonl(final / "train.jsonl", rows)
    Q.write_jsonl(final / "test.jsonl", [{"id": q["id"], "qtype": q["qtype"], "question": q["messages"][1]["content"],
                                          "messages": q["messages"], "gold_answer": q["gold"],
                                          "answer_kind": q["answer_kind"], "source": q["source"]} for q in test_questions])
    per_source = Counter(r["id"].rsplit("-", 1)[0] for r in rows)
    m = sampling.metrics(records, k)
    stats = {
        "questions": len(questions), "rows_kept": len(rows), "unsolved": len(unsolved),
        "source_rows": len({q["id"].rsplit("-", 1)[0] for q in questions}),
        "source_rows_with_all_3": sum(1 for v in per_source.values() if v == 3),
        "rows_by_qtype": dict(Counter(r["qtype"] for r in rows)),
        "kept_mean_reasoning_tokens": sum(r["reasoning_tokens"] for r in rows) / max(1, len(rows)),
        "test_questions": len(test_questions), "k": k, "max_new_tokens": args.max_new_tokens,
        "sampling": sampling.THINKING_SAMPLING, "generator": "Qwen/Qwen3.8-27B (thinking mode)",
        "base_model_on_train_questions": m,
    }
    (final / "stats.json").write_text(json.dumps(stats, indent=2))
    # human review file: 2 examples per question type
    md = ["# Cricket reasoning dataset - review samples", "", "```json", json.dumps({k2: v for k2, v in stats.items()
          if k2 != "base_model_on_train_questions"}, indent=2), "```", ""]
    for qt in Q.QTYPES:
        for r in [r for r in rows if r["qtype"] == qt][:2]:
            md += [f"## {qt} - `{r['id']}`", "", "**Question**", "", r["question"], "",
                   f"**Reasoning** ({r['reasoning_tokens']} tokens; {r['samples_correct']}/{r['samples_generated']} samples correct)",
                   "", "```", r["reasoning"][:4000], "```", "", f"**Response:** {r['response']}", "",
                   f"**Gold answer:** {r['gold_answer']}", ""]
    (final / "samples.md").write_text("\n".join(md), encoding="utf-8")
    return stats, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--questions_dir", required=True)
    p.add_argument("--out_dir", required=True, help="raw samples + final dataset; reuse to resume")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--batch_prompts", type=int, default=16, help="questions per generate() call (x k sequences)")
    p.add_argument("--limit", type=int, default=0, help="pilot: only the first N training questions")
    p.add_argument("--deadline", type=float, default=float(os.environ.get("TRAIN_DEADLINE", 0)))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"), timeout=timedelta(hours=2))
    out_dir = Path(args.out_dir)
    questions = Q.read_jsonl(Path(args.questions_dir) / "train.jsonl")
    test_questions = Q.read_jsonl(Path(args.questions_dir) / "test.jsonl")
    if args.limit:
        questions = questions[: args.limit]
    log(f"{len(questions)} training questions, k={args.k}, world {dist.get_world_size()}, out {out_dir}")

    use_wandb = RANK == 0 and os.environ.get("WANDB_MODE") != "disabled"
    if use_wandb:
        import wandb
        qstats = json.loads((Path(args.questions_dir) / "stats.json").read_text())
        kw = dict(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-reasoning"), job_type="datagen",
                  name=f"datagen-{out_dir.name}-{os.environ.get('SLURM_JOB_ID', 'local')}",
                  config={**vars(args), "questions": qstats, "sampling": sampling.THINKING_SAMPLING})
        try:
            wandb.init(**kw)
        except Exception as e:
            log(f"W&B init failed ({e}); logging offline")
            wandb.init(mode="offline", **kw)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map={"": LOCAL_RANK}, attn_implementation="sdpa")
    log(f"model loaded, {torch.cuda.memory_allocated() / 2**30:.0f} GiB/GPU")

    def on_batch(m):
        if use_wandb:
            import wandb
            wandb.log({f"datagen/{k2}": v for k2, v in m.items()})

    left = sampling.sample(model, tok, questions, out_dir / "samples", "train", k=args.k,
                           max_new_tokens=args.max_new_tokens, batch_prompts=args.batch_prompts,
                           deadline=args.deadline, seed=args.seed, log=log, on_batch=on_batch)
    pending = sampling.pending_everywhere(left)
    dist.barrier()
    if RANK == 0:
        records = sampling.collect(out_dir / "samples", "train")
        m = sampling.metrics(records, args.k)
        log("base model on training questions: " + json.dumps({qt: {k2: round(v, 3) for k2, v in x.items()} for qt, x in m.items()}))
        if pending:
            log(f"{pending} questions still pending across ranks - resubmit the same command to continue")
        else:
            stats, rows = build_dataset(questions, test_questions, records, out_dir, args.k, args)
            log(f"dataset: {stats['rows_kept']} rows from {stats['questions']} questions "
                f"({stats['unsolved']} unsolved), written to {out_dir / 'final'}")
        if use_wandb:
            import wandb
            for qt, x in m.items():
                wandb.summary.update({f"datagen/{qt}/{k2}": v for k2, v in x.items()})
            wandb.summary["datagen/pending"] = pending
            if not pending:
                wandb.summary.update({f"dataset/{k2}": v for k2, v in stats.items() if isinstance(v, (int, float))})
                t = wandb.Table(columns=["id", "qtype", "question", "reasoning", "response", "gold", "tokens"])
                for r in rows[:: max(1, len(rows) // 30)][:30]:
                    t.add_data(r["id"], r["qtype"], r["question"], r["reasoning"][:3000], r["response"], r["gold_answer"], r["reasoning_tokens"])
                wandb.log({"dataset/samples": t})
            wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
