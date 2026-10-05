#!/usr/bin/env python
"""After the vLLM evals of the base and fine-tuned (merged) models: metrics, report, W&B (resuming the
training run), and push of the adapter + model card + eval report to the hub. Runs in the PyTorch container."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling  # noqa: E402
from train import finalize, wandb_eval  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--adapter_dir", required=True)
p.add_argument("--k", type=int, default=4)
p.add_argument("--hub_model_id", default=None)
p.add_argument("--hub_revision", default=None)
p.add_argument("--base_model_id", default="Qwen/Qwen3.8-27B")
args = p.parse_args()

run_dir = Path(args.run_dir)
summaries = {}
for tag in ("base", "ft"):
    recs = sampling.collect(run_dir / "eval", tag)
    if not recs:
        sys.exit(f"no {tag} eval samples in {run_dir / 'eval'}")
    summaries[tag] = sampling.metrics(recs, args.k)
    (run_dir / "eval" / f"{tag}_summary.json").write_text(json.dumps(summaries[tag], indent=2))
    for qt, x in summaries[tag].items():
        print(f"[{tag}] {qt:10} acc {x['accuracy']:.4f}  pass@{args.k} {x[f'pass@{args.k}']:.4f}  "
              f"tokens {x['mean_tokens']:,.0f}  truncated {x['truncated_rate']:.3f}  (n={x['questions']})")

use_wandb = os.environ.get("WANDB_MODE") != "disabled"
if use_wandb:
    import wandb
    kw = dict(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-reasoning"), id=run_dir.name, name=run_dir.name,
              resume="allow", dir=str(run_dir), job_type="eval",
              config={"eval": {"engine": "vLLM", "k": args.k, "sampling": sampling.THINKING_SAMPLING}})
    try:
        wandb.init(**kw)
    except Exception as e:
        print(f"W&B init failed ({e}); logging offline")
        wandb.init(mode="offline", **kw)
    for tag, m in summaries.items():
        wandb_eval(tag, m)
finalize(SimpleNamespace(eval_k=args.k, hub_model_id=args.hub_model_id, hub_revision=args.hub_revision,
                         base_model_id=args.base_model_id), run_dir, use_wandb, args.adapter_dir)
if use_wandb:
    import wandb
    wandb.finish()
print((run_dir / "report.md").read_text() if (run_dir / "report.md").exists() else "no report")
