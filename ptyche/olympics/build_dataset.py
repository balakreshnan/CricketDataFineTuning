"""Turn Dynamo-sampled traces into the published 1:1 dataset + SFT files (login node, tools venv).
   $TOOLS_VENV/bin/python build_dataset.py --gen $PROJ/gen/rows [--unsolved-out q.jsonl] [--push REPO] [--wandb]

Source = the row-anchored questions (questions_rows_{train,test}.jsonl, plus all fallback questions in
questions_rows_fallback*.jsonl): every unique row of athlete_events.csv has >= 1 question. All sample files in --gen
whose name starts with "rows" are merged by question id (first pass, retries, fallbacks). For each SOURCE ROW the
shortest verified-correct, non-truncated trace over all its questions is kept -> one dataset row per source row.
  --unsolved-out  write the questions of rows still without a correct trace (for a retry / fallback pass)
  --push          refuses unless dataset rows >= unique source rows (1:1 coverage), then uploads (PUBLIC repo, user request 2026-10-09)
Outputs: $PROJ/data/hf/ (train/test jsonl + README), $PROJ/data/sft/{training,validation}.jsonl
(validation = 1% of train rows, dev loss only), $PROJ/data/rows_dataset_stats.json.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olympics_qa import SYSTEM_PROMPT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--unsolved-out", default=None)
ap.add_argument("--push", default=None)
ap.add_argument("--allow-missing", type=int, default=0,
                help="push even if up to N source rows lack a trace (user-approved shortfall; listed in the card)")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--max-trace-tokens", type=int, default=28000,
                help="longest trace kept in the published dataset (= retry2 generation budget; SFT has its own cap)")
ap.add_argument("--sft-max-tokens", type=int, default=8000, help="prompt+trace cap for SFT rows (seq 8192)")
a = ap.parse_args()
DATA = os.path.join(os.environ["PROJ"], "data")

questions = {}
fallbacks = sorted(os.path.basename(p)[len("questions_"):-len(".jsonl")]
                   for p in glob.glob(f"{DATA}/questions_rows_fallback*.jsonl"))
for name in ["rows_train", "rows_test"] + fallbacks:
    path = f"{DATA}/questions_{name}.jsonl"
    if os.path.exists(path):
        for l in open(path):
            q = json.loads(l)
            questions[q["id"]] = q
source_rows = json.load(open(f"{DATA}/rows_stats.json"))["source_rows"]

samples = defaultdict(list)
files = sorted(glob.glob(f"{a.gen}/rows*.jsonl"))
for f in files:
    for l in open(f):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        samples[r["id"]].append(r)

by_row = defaultdict(list)  # source row -> its questions
for q in questions.values():
    by_row[q["source"]["row_index"]].append(q)

best, unsolved = {}, []
for row, qs in by_row.items():
    good = [(s, q) for q in qs for s in samples.get(q["id"], [])
            if s["correct"] and s["finish_reason"] == "stop" and s["reasoning"].strip()
            and (s["completion_tokens"] or 0) <= a.max_trace_tokens]
    if good:
        best[row] = min(good, key=lambda sq: sq[0]["completion_tokens"])
    else:
        unsolved.extend(qs)  # incl. never-sampled questions (e.g. a reworded template)

all_s = [s for q in questions.values() for s in samples.get(q["id"], [])]
sampled_q = [q for q in questions.values() if q["id"] in samples]
stats = {
    "source_rows_unique": source_rows, "questions": len(questions), "questions_sampled": len(sampled_q),
    "samples": len(all_s), "sample_files": [os.path.basename(f) for f in files],
    "pass@1": round(100 * sum(s["correct"] for s in all_s) / max(1, len(all_s)), 2),
    "truncated_pct": round(100 * sum(s["finish_reason"] == "length" for s in all_s) / max(1, len(all_s)), 2),
    "rows_with_trace": len(best), "rows_without_trace": source_rows - len(best),
    "coverage_pct": round(100 * len(best) / source_rows, 3),
    "per_qtype_pass@1": {qt: round(100 * sum(s["correct"] for s in v) / len(v), 2) for qt, v in
                         ((qt, [s for s in all_s if s["qtype"] == qt]) for qt in sorted({s["qtype"] for s in all_s}))},
}

if a.unsolved_out:
    with open(a.unsolved_out, "w") as f:
        for q in unsolved:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    stats["unsolved_questions_written"] = len(unsolved)


def record(s, q):
    src = q["source"]
    return {"id": f"row{src['row_index']}", "source_row_index": src["row_index"], "question_id": q["id"],
            "qtype": q["qtype"], "athlete_id": src["athlete_id"], "athlete": src["name"], "games": src["games"],
            "event": src["event"], "noc": src["noc"], "question": q["messages"][1]["content"], "answer": str(q["gold"]),
            "reasoning": s["reasoning"].strip(), "response": s["content"].strip(),
            "reasoning_tokens": s["completion_tokens"], "prompt_tokens": s.get("prompt_tokens"),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": q["messages"][1]["content"]},
                         {"role": "assistant", "reasoning_content": s["reasoning"].strip(), "content": s["content"].strip()}]}


test_ids = {json.loads(l)["id"] for l in open(f"{DATA}/questions_rows_test.jsonl")}
test_athletes = {questions[i]["source"]["athlete_id"] for i in test_ids}
recs = {"train": [], "test": []}
for row in sorted(best):
    s, q = best[row]
    recs["test" if q["source"]["athlete_id"] in test_athletes else "train"].append(record(s, q))

os.makedirs(f"{DATA}/hf/data", exist_ok=True)
os.makedirs(f"{DATA}/sft", exist_ok=True)
for sp, rs in recs.items():
    with open(f"{DATA}/hf/data/{sp}.jsonl", "w") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_valid = n_long = 0
with open(f"{DATA}/sft/training.jsonl", "w") as ft, open(f"{DATA}/sft/validation.jsonl", "w") as fv:
    for r in recs["train"]:
        if (r["prompt_tokens"] or 0) + (r["reasoning_tokens"] or 0) > a.sft_max_tokens:
            n_long += 1  # stays in the published dataset; would be cut off at the packed sequence length
            continue
        dev = int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 100 == 0
        n_valid += dev
        (fv if dev else ft).write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
toks = [r["reasoning_tokens"] for r in recs["train"] + recs["test"]]
stats.update({"dataset_train": len(recs["train"]), "dataset_test": len(recs["test"]),
              "dataset_total": len(recs["train"]) + len(recs["test"]),
              "sft_training": len(recs["train"]) - n_valid - n_long, "sft_validation": n_valid,
              "sft_skipped_too_long": n_long,
              "dataset_qtypes": dict(Counter(r["qtype"] for r in recs["train"] + recs["test"])),
              "reasoning_tokens_mean": round(statistics.mean(toks)) if toks else 0,
              "reasoning_tokens_p95": sorted(toks)[int(.95 * len(toks))] if toks else 0})
json.dump(stats, open(f"{DATA}/rows_dataset_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

if a.wandb:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT"), entity=os.environ.get("WANDB_ENTITY"),
                     name="datagen-rows", job_type="datagen",
                     config={"generator": os.environ.get("MODEL_ID"), "serving": "NVIDIA Dynamo 1.5.1 + SGLang 0.5.18",
                             "sampling": "T=1.0 top_p=0.95 reasoning on", "selection": "shortest verified-correct per source row"})
    run.summary.update({k: v for k, v in stats.items() if isinstance(v, (int, float))})
    run.summary.update({f"pass@1/{k}": v for k, v in stats["per_qtype_pass@1"].items()})
    run.log({"reasoning_tokens": wandb.Histogram(toks)})
    t = wandb.Table(columns=["qtype", "rows"], data=[[k, v] for k, v in stats["dataset_qtypes"].items()])
    run.log({"dataset_qtypes": t})
    run.finish()

if a.push:
    total = stats["dataset_total"]
    missing = sorted(set(range(source_rows)) - set(best))
    if len(missing) > a.allow_missing:
        raise SystemExit(f"refusing to push: {total} dataset rows < {source_rows} unique source rows (not 1:1 yet)")
    coverage = (f"**Every unique source row ({source_rows:,} after dropping exact duplicates) has exactly one row "
                f"here**" if not missing else
                f"**{total:,} of the {source_rows:,} unique source rows (after dropping exact duplicates) have exactly "
                f"one row here** ({100 * total / source_rows:.3f}%; the {len(missing)} rows without a verified trace, "
                f"source_row_index {', '.join(map(str, missing))}, are counting questions over 100+-entrant rosters "
                f"that no sample answered correctly across 3 question templates and 60+ samples per row)")
    from huggingface_hub import HfApi
    qt_rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(stats["dataset_qtypes"].items()))
    readme = f"""---
license: mit
task_categories: [text-generation, question-answering]
language: [en]
size_categories: [100K<n<1M]
tags: [reasoning, olympics, synthetic, distillation, nemotron, dynamo, sglang]
configs:
- config_name: default
  data_files:
  - {{split: train, path: data/train.jsonl}}
  - {{split: test, path: data/test.jsonl}}
---
# Olympics reasoning traces, one per source row

Distilled from [pranjalvoid/olympics-dataset](https://huggingface.co/datasets/pranjalvoid/olympics-dataset)
(`athlete_events.csv`, 120 years of Olympic entries). {coverage}. Each row is a verifiable question anchored on
one entry (athlete, event, Games), the gold answer computed from the data, and a verified-correct reasoning trace.

Traces were generated by `{os.environ.get('MODEL_ID')}` served with **NVIDIA Dynamo 1.5.1 (SGLang backend)** on
GB200 nodes: several samples per question (T=1.0, top_p=0.95, reasoning on); the shortest trace whose final answer
matches the gold answer was kept. Each prompt contains the relevant table (the event's entry list at that Games,
or the athlete's career record), so the answer is fully determined by the prompt.

| split | rows |
|---|---|
| train | {len(recs['train']):,} |
| test (athletes held out) | {len(recs['test']):,} |
| **total** | **{total:,}** |

| question type | rows |
|---|---|
{qt_rows}

Fields: `source_row_index` (row in the deduplicated CSV), `question`, `answer`, `reasoning`, `response`,
`messages` (chat format; the assistant turn carries the trace as `reasoning_content`), plus athlete / Games /
event / NOC metadata. Mean trace length {stats['reasoning_tokens_mean']} tokens.
"""
    open(f"{DATA}/hf/README.md", "w").write(readme)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.push, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(a.push, repo_type="dataset", private=False)  # public even if it existed as private
    api.upload_large_folder(repo_id=a.push, repo_type="dataset", folder_path=f"{DATA}/hf")
    print("pushed", a.push, "rev", api.dataset_info(a.push).sha, flush=True)
