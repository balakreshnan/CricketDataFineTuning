#!/usr/bin/env python
"""After the vLLM workers: merge shard files, compute metrics, build the dataset (when every question is
sampled) and log everything to W&B. Runs in the PyTorch container (has wandb); single process, no GPU."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402
import sampling  # noqa: E402
from generate import build_dataset  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--questions_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--k", type=int, default=4)
p.add_argument("--max_tokens", type=int, default=6144)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()

out_dir = Path(args.out_dir)
questions = Q.read_jsonl(Path(args.questions_dir) / "train.jsonl")
test_questions = Q.read_jsonl(Path(args.questions_dir) / "test.jsonl")
if args.limit:
    questions = questions[: args.limit]
records = sampling.collect(out_dir / "samples", "train")
sampled = {r["id"] for r in records}
pending = sum(q["id"] not in sampled for q in questions)
m = sampling.metrics(records, args.k)
print(f"{len(sampled)}/{len(questions)} questions sampled ({len(records)} samples); pending {pending}")
for qt, x in m.items():
    print(f"  {qt:10} acc/sample {x['accuracy']:.3f}  pass@{args.k} {x[f'pass@{args.k}']:.3f}  mean tok {x['mean_tokens']:,.0f}"
          f"  truncated {x['truncated_rate']:.3f}  no-answer {x['no_answer_rate']:.3f}")
stats, rows = None, []
if not pending:
    stats, rows = build_dataset(questions, test_questions, records, out_dir, args.k,
                                SimpleNamespace(max_new_tokens=args.max_tokens))
    print(f"dataset: {stats['rows_kept']} rows from {stats['questions']} questions ({stats['unsolved']} unsolved); "
          f"{stats['source_rows_with_all_3']}/{stats['source_rows']} source rows have all 3 -> {out_dir / 'final'}")
else:
    print("resubmit the same stage to finish sampling; the dataset is built once nothing is pending")

if os.environ.get("WANDB_MODE") != "disabled":
    import wandb
    kw = dict(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-reasoning"), job_type="datagen",
              name=f"datagen-{out_dir.name}-{os.environ.get('SLURM_JOB_ID', 'local')}",
              config={**vars(args), "engine": "vLLM (NGC container)", "sampling": sampling.THINKING_SAMPLING,
                      "questions": json.loads((Path(args.questions_dir) / "stats.json").read_text())})
    try:
        wandb.init(**kw)
    except Exception as e:
        print(f"W&B init failed ({e}); logging offline")
        wandb.init(mode="offline", **kw)
    for qt, x in m.items():
        wandb.summary.update({f"datagen/{qt}/{k}": v for k, v in x.items()})
    wandb.summary["datagen/pending"] = pending
    tok = [r["n_tokens"] for r in records]
    wandb.log({"datagen/output_tokens_hist": wandb.Histogram(tok)})
    if stats:
        wandb.summary.update({f"dataset/{k}": v for k, v in stats.items() if isinstance(v, (int, float))})
        t = wandb.Table(columns=["id", "qtype", "question", "reasoning", "response", "gold", "tokens"])
        for r in rows[:: max(1, len(rows) // 30)][:30]:
            t.add_data(r["id"], r["qtype"], r["question"], r["reasoning"][:3000], r["response"], r["gold_answer"],
                       r["reasoning_tokens"])
        wandb.log({"dataset/samples": t})
    wandb.finish()
