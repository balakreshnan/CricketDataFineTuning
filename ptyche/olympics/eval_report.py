"""Base vs fine-tuned on the held-out benchmarks (both sampled through Dynamo), logged to W&B.
   python eval_report.py <run>
Base: $PROJ/gen/full/test.* (550 table questions) and $PROJ/gen/rows/rows_test.* (26,997 row questions, held-out
athletes). Fine-tuned: $PROJ/gen/eval-<run>/{test,rows_test}.*. Writes $PROJ/runs/<run>/eval/report.json and
$PROJ/runs/<run>/card_facts.json (model card table)."""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

run = sys.argv[1]
P = os.environ["PROJ"]


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)]


def summarize(rs):
    by = defaultdict(list)
    for r in rs:
        by["all"].append(r)
        by[r["qtype"]].append(r)
    return {qt: {"samples": len(v), "pass@1": round(100 * sum(r["correct"] for r in v) / len(v), 2),
                 "mean_tokens": round(statistics.mean(r["completion_tokens"] or 0 for r in v)),
                 "truncated_pct": round(100 * sum(r["finish_reason"] == "length" for r in v) / len(v), 2)}
            for qt, v in by.items()}


rep = {}
for bench, base_glob, ft_glob in (("table_benchmark", f"{P}/gen/full/test.shard*.jsonl", f"{P}/gen/eval-{run}/test.shard*.jsonl"),
                                  ("row_benchmark", f"{P}/gen/rows/rows_test.shard*.jsonl", f"{P}/gen/eval-{run}/rows_test.shard*.jsonl")):
    base, ft = load(base_glob), load(ft_glob)
    # same questions on both sides (the reworded ev_noc_medal_count rows have no unbiased first-pass base samples)
    common = {r["id"] for r in base} & {r["id"] for r in ft}
    rep[bench] = {"base": summarize([r for r in base if r["id"] in common]),
                  "finetuned": summarize([r for r in ft if r["id"] in common]),
                  "questions_compared": len(common)}
os.makedirs(f"{P}/runs/{run}/eval", exist_ok=True)
json.dump(rep, open(f"{P}/runs/{run}/eval/report.json", "w"), indent=1)

for bench, d in rep.items():
    print(f"\n{bench}: {'qtype':20s} {'base':>8s} {'ft':>8s} {'base_tok':>9s} {'ft_tok':>7s} {'base_tr%':>8s} {'ft_tr%':>7s}")
    for qt in d["base"]:
        b, f = d["base"][qt], d["finetuned"].get(qt, {})
        print(f"{'':16s}{qt:20s} {b['pass@1']:8.2f} {f.get('pass@1', float('nan')):8.2f} {b['mean_tokens']:9d} "
              f"{f.get('mean_tokens', 0):7d} {b['truncated_pct']:8.2f} {f.get('truncated_pct', float('nan')):7.2f}")

sft_dir = os.environ.get("SFT_DATA") or f"{P}/data/sft"   # frozen training copy carries its own stats
stats = json.load(open(f"{sft_dir}/rows_dataset_stats.json" if os.path.exists(f"{sft_dir}/rows_dataset_stats.json")
                       else f"{P}/data/rows_dataset_stats.json"))
facts = {"training data": f"{os.environ.get('HF_DATASET_REPO')} train split: {stats['sft_training']:,} rows "
                          f"(snapshot at {stats['coverage_pct']}% source-row coverage), 1 epoch",
         "hardware": "NVIDIA GB200 NVL72 (Ptyche), Megatron-Bridge, HybridEP MoE dispatcher"}
for bench, label in (("row_benchmark", f"row benchmark ({rep['row_benchmark']['questions_compared']:,} q, held-out athletes)"),
                     ("table_benchmark", "table benchmark (550 q)")):
    b, f = rep[bench]["base"]["all"], rep[bench]["finetuned"]["all"]
    facts[f"{label}: pass@1"] = f"{b['pass@1']}% -> **{f['pass@1']}%**"
    facts[f"{label}: mean tokens"] = f"{b['mean_tokens']:,} -> {f['mean_tokens']:,}"
    facts[f"{label}: truncated"] = f"{b['truncated_pct']}% -> {f['truncated_pct']}%"
json.dump(facts, open(f"{P}/runs/{run}/card_facts.json", "w"), indent=1)

if os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc")):
    import wandb
    w = wandb.init(project=os.environ.get("WANDB_PROJECT"), entity=os.environ.get("WANDB_ENTITY"),
                   name=f"eval-{run}", job_type="eval", config={"run": run, "served_with": "NVIDIA Dynamo + SGLang"})
    for bench, d in rep.items():
        t = wandb.Table(columns=["qtype", "base_pass@1", "ft_pass@1", "base_tokens", "ft_tokens", "base_trunc%", "ft_trunc%"])
        for qt, b in d["base"].items():
            f = d["finetuned"].get(qt, {})
            t.add_data(qt, b["pass@1"], f.get("pass@1"), b["mean_tokens"], f.get("mean_tokens"),
                       b["truncated_pct"], f.get("truncated_pct"))
        w.log({f"{bench}/table": t})
        w.summary.update({f"{bench}/{side}/{k}": v for side in ("base", "finetuned")
                          for k, v in d[side].get("all", {}).items()})
    w.finish()
print(json.dumps(facts, indent=1))
