#!/usr/bin/env python
"""Assemble the one-to-one reasoning dataset from the vLLM sample files (streaming: ~2M samples).

For every question keep the shortest verified-correct, finished trace. Writes, once nothing is pending:
  final/train.jsonl, final/test.jsonl  reasoning rows (train / held-out-match split)
  final/eval.jsonl                     the fixed 600-question benchmark (questions + gold only)
  final/stats.json, final/samples.md   statistics incl. source-row coverage, review examples
  retry.jsonl                          questions with no correct sample yet (run `submit.sh retry-full`)
  final/uncovered_rows.jsonl           source rows still without any reasoning row (should end empty)
Logs to W&B. Runs in the PyTorch container (single process, no GPU).
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--questions_dir", required=True)
p.add_argument("--questions_file", default="all.jsonl", help="all.jsonl (full run) or pilot.jsonl")
p.add_argument("--out_dir", required=True)
p.add_argument("--k", type=int, default=4)
p.add_argument("--max_tokens", type=int, default=6144)
args = p.parse_args()

qdir, out = Path(args.questions_dir), Path(args.out_dir)
questions = Q.read_jsonl(qdir / args.questions_file)
qmap = {q["id"]: q for q in questions}
print(f"{len(questions):,} questions in {args.questions_file}", flush=True)

# ---- stream every sample file (first pass and retries)
best, n_s, n_c = {}, defaultdict(int), defaultdict(int)
agg = defaultdict(lambda: defaultdict(float))
for f in sorted((out / "samples").glob("*_rank*.jsonl")):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in qmap:
                continue
            qid, a = r["id"], agg[r["qtype"]]
            n_s[qid] += 1
            a["samples"] += 1
            a["tokens"] += r["n_tokens"]
            a["truncated"] += not r["finished"]
            a["no_answer"] += r["finished"] and r["pred"] is None
            if r["correct"]:
                n_c[qid] += 1
                a["correct"] += 1
                a["tokens_correct"] += r["n_tokens"]
                if qid not in best or r["n_tokens"] < best[qid]["n_tokens"]:
                    best[qid] = {"text": r["text"], "n_tokens": r["n_tokens"], "pred": r["pred"]}

pending = [q for q in questions if q["id"] not in n_s]
unsolved = [q for q in questions if q["id"] in n_s and q["id"] not in best]
metrics = {}
for qt in Q.ALL_QTYPES:
    qs = [q["id"] for q in questions if q["qtype"] == qt and q["id"] in n_s]
    a = agg[qt]
    if a["samples"]:
        metrics[qt] = {"questions": len(qs), "samples": int(a["samples"]), "accuracy": a["correct"] / a["samples"],
                       "pass@k": sum(1 for x in qs if x in best) / len(qs), "mean_tokens": a["tokens"] / a["samples"],
                       "mean_tokens_correct": a["tokens_correct"] / max(1, a["correct"]),
                       "truncated_rate": a["truncated"] / a["samples"], "no_answer_rate": a["no_answer"] / a["samples"]}
tot = {k: sum(agg[qt][k] for qt in agg) for k in ("samples", "correct", "tokens", "truncated")}
if tot["samples"]:
    metrics["overall"] = {"questions": len(n_s), "samples": int(tot["samples"]), "accuracy": tot["correct"] / tot["samples"],
                          "pass@k": len(best) / max(1, len(n_s)), "mean_tokens": tot["tokens"] / tot["samples"],
                          "truncated_rate": tot["truncated"] / tot["samples"]}
print(f"sampled {len(n_s):,}/{len(questions):,} questions ({int(tot['samples']):,} samples); pending {len(pending):,}; "
      f"unsolved so far {len(unsolved):,}")
for qt, m in metrics.items():
    print(f"  {qt:12} acc/sample {m['accuracy']:.3f}  pass@k {m['pass@k']:.4f}  mean tok {m['mean_tokens']:,.0f}  "
          f"truncated {m['truncated_rate']:.4f}")

stats = None
if not pending:
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: defaultdict(int))
    covered = defaultdict(set)
    with open(final / "train.jsonl", "w", encoding="utf-8") as ftr, open(final / "test.jsonl", "w", encoding="utf-8") as fte:
        for q in questions:
            b = best.get(q["id"])
            if not b:
                continue
            reasoning, answer, _ = Q.split_reasoning(b["text"])
            split = q.get("split", "train")
            row = {"id": q["id"], "split": split, "qtype": q["qtype"], "question": q["messages"][1]["content"],
                   "messages": q["messages"] + [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}],
                   "reasoning": reasoning, "response": answer, "final_answer": b["pred"], "gold_answer": q["gold"],
                   "answer_kind": q["answer_kind"], "reasoning_tokens": b["n_tokens"], "samples_generated": n_s[q["id"]],
                   "samples_correct": n_c[q["id"]], "source": q["source"], "completion_text": b["text"]}
            (ftr if split == "train" else fte).write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split][q["qtype"]] += 1
            covered[split].add(q["source"]["row_index"])
    src_rows = defaultdict(set)
    for q in questions:
        src_rows[q.get("split", "train")].add(q["source"]["row_index"])
    uncovered = sorted(r for s in src_rows for r in src_rows[s] - covered[s])
    Q.write_jsonl(final / "uncovered_rows.jsonl", [{"row_index": r} for r in uncovered])
    Q.write_jsonl(out / "retry.jsonl", unsolved)
    if (qdir / "eval.jsonl").exists():
        shutil.copy(qdir / "eval.jsonl", final / "eval.jsonl")
    n_src = sum(len(v) for v in src_rows.values())
    stats = {"source_dataset": "Valarmathy/CricketData", "source_rows_in_question_set": n_src,
             "source_rows_covered": n_src - len(uncovered), "source_rows_uncovered": len(uncovered),
             "rows_train": sum(counts["train"].values()), "rows_test": sum(counts["test"].values()),
             "rows_total": sum(sum(c.values()) for c in counts.values()),
             "rows_by_split_and_type": {s: dict(c) for s, c in counts.items()},
             "questions": len(questions), "unsolved_questions": len(unsolved),
             "kept_mean_reasoning_tokens": sum(b["n_tokens"] for b in best.values()) / max(1, len(best)),
             "k": args.k, "max_new_tokens": args.max_tokens, "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
             "generator": "Qwen/Qwen3.8-27B (thinking mode)", "base_model_on_questions": metrics}
    (final / "stats.json").write_text(json.dumps(stats, indent=2))
    md = ["# Cricket reasoning dataset (one-to-one) - review samples", "", "```json",
          json.dumps({k: v for k, v in stats.items() if k != "base_model_on_questions"}, indent=2), "```", ""]
    shown = defaultdict(int)
    for q in questions:
        b = best.get(q["id"])
        if not b or shown[q["qtype"]] >= 2:
            continue
        shown[q["qtype"]] += 1
        reasoning, answer, _ = Q.split_reasoning(b["text"])
        md += [f"## {q['qtype']} - `{q['id']}` ({q.get('split', 'train')})", "", "**Question**", "", q["messages"][1]["content"],
               "", f"**Reasoning** ({b['n_tokens']} tokens; {n_c[q['id']]}/{n_s[q['id']]} samples correct)", "",
               "```text", reasoning[:4000], "```", "", f"**Response:** {answer}", "", f"**Gold answer:** {q['gold']}", ""]
    (final / "samples.md").write_text("\n".join(md), encoding="utf-8")
    print(f"dataset: {stats['rows_total']:,} rows (train {stats['rows_train']:,}, test {stats['rows_test']:,}); "
          f"source rows covered {stats['source_rows_covered']:,}/{n_src:,}; unsolved questions {len(unsolved):,} "
          f"-> {out / 'retry.jsonl'}" + ("  (run `submit.sh retry-full`)" if unsolved else ""))
else:
    print("pending questions remain - resubmit the same generate stage to continue")

if os.environ.get("WANDB_MODE") != "disabled":
    import wandb
    kw = dict(project=os.environ.get("WANDB_PROJECT", "qwen38-cricket-reasoning"), job_type="datagen",
              name=f"datagen-{out.name}-{os.environ.get('SLURM_JOB_ID', 'local')}",
              config={**vars(args), "engine": "vLLM", "mode": "one-to-one",
                      "questions": json.loads((qdir / "stats.json").read_text())})
    try:
        wandb.init(**kw)
    except Exception as e:
        print(f"W&B init failed ({e}); logging offline")
        wandb.init(mode="offline", **kw)
    for qt, m in metrics.items():
        wandb.summary.update({f"datagen/{qt}/{k}": v for k, v in m.items()})
    wandb.summary.update({"datagen/pending": len(pending), "datagen/unsolved": len(unsolved)})
    if stats:
        wandb.summary.update({f"dataset/{k}": v for k, v in stats.items() if isinstance(v, (int, float))})
    wandb.finish()
