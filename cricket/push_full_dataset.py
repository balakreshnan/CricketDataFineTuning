#!/usr/bin/env python
"""Push the one-to-one cricket reasoning dataset (every Valarmathy/CricketData row covered) to Hugging Face.

Refuses unless the build is complete and every source row has at least one verified reasoning row (override with
--allow_uncovered). Train is written in shards of --shard_rows rows; internal fields (completion_text) are dropped.
The repo is created private if it does not exist; visibility of an existing repo is left unchanged.
"""
import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--final_dir", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--shard_rows", type=int, default=100_000)
p.add_argument("--allow_uncovered", action="store_true")
args = p.parse_args()

final = Path(args.final_dir)
stats = json.loads((final / "stats.json").read_text())
if stats["source_rows_uncovered"] and not args.allow_uncovered:
    raise SystemExit(f"{stats['source_rows_uncovered']} source rows have no reasoning row yet - run retry-full first")
tmp = Path(tempfile.mkdtemp()) / "repo"
(tmp / "data").mkdir(parents=True)


def write_split(name):
    n, shard, fo = 0, 0, None
    with open(final / f"{name}.jsonl", encoding="utf-8") as fi:
        for line in fi:
            if n % args.shard_rows == 0:
                if fo:
                    fo.close()
                fo = open(tmp / "data" / f"{name}-{shard:05d}.jsonl", "w", encoding="utf-8")
                shard += 1
            r = json.loads(line)
            r.pop("completion_text", None)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    if fo:
        fo.close()
    return n


counts = {s: write_split(s) for s in ("train", "test")}
(tmp / "data" / "eval-00000.jsonl").write_text((final / "eval.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
(tmp / "stats.json").write_text(json.dumps(stats, indent=2))
m = stats["base_model_on_questions"].get("overall", {})
by = stats["rows_by_split_and_type"]
types = sorted({t for s in by.values() for t in s})
table = "\n".join(f"| `{t}` | {by.get('train', {}).get(t, 0):,} | {by.get('test', {}).get(t, 0):,} |" for t in types)
card = f"""---
license: cc0-1.0
task_categories: [question-answering, text-generation]
tags: [reasoning, cricket, t20, synthetic, chain-of-thought, qwen, distillation]
size_categories: [100K<n<1M]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.jsonl
  - split: test
    path: data/test-*.jsonl
  - split: eval
    path: data/eval-*.jsonl
---

# Cricket T20I reasoning traces - full one-to-one coverage (Qwen3.8-27B, verified)

Every one of the **{stats['source_rows_in_question_set']:,} rows** of
[Valarmathy/CricketData](https://huggingface.co/datasets/Valarmathy/CricketData) (`raw/ball_by_ball_it20.csv`,
T20 Internationals ball by ball, CC0) has at least one verified reasoning row here
({stats['source_rows_covered']:,} covered). **{stats['rows_total']:,} rows** in total: one question per source row plus a
second, different question on a random subset of training rows.

**Question types** (answers computed exactly from the row):

| type | train rows | test rows |
|---|---|---|
{table}

- `run_rate`: current run rate (overs in cricket notation, 14.3 = 14 overs 3 balls)
- `chase_rate`: run rate needed to win (2nd innings) / scored over the rest of the innings (1st innings, full 20 overs)
- `milestone`: balls the striker needs to reach the next 50 at their current strike rate
- `projection`: projected 20-over total at the current run rate (1st innings)
- `strike_rate`: the striker's strike rate
- `legal_balls`: innings opened with a wide / no-ball - legal deliveries still to come

**How it was made**: `Qwen/Qwen3.8-27B` answered each question {stats['k']} times in thinking mode (temperature 1.0,
top_p 0.95, top_k 20, up to {stats['max_new_tokens']} tokens; extra samples for hard questions). Answers were checked
against the exact values and the **shortest correct** trace was kept. Base-model accuracy while generating:
{100 * m.get('accuracy', 0):.1f}% per sample, {100 * m.get('pass@k', 0):.2f}% of questions solved.

**Splits**: `train` ({counts['train']:,} rows) and `test` ({counts['test']:,} rows) are split **by match** (177 held-out
matches in test; rain-revised-target matches are in train and get no target-based questions). `eval` is the fixed
600-question benchmark (questions + gold only) from the held-out matches, shared with
[Balab2021/CricketData-T20-Reasoning-Qwen3.8](https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8).

**Fields**: `id`, `split`, `qtype`, `question`, `messages` (system, user, assistant with `<think>` reasoning), `reasoning`,
`response`, `final_answer`, `gold_answer`, `answer_kind`, `reasoning_tokens`, `samples_generated`, `samples_correct`,
`source` (the originating row: row_index, match, innings, over, ball, score, batter, bowler, extras, wicket ...).
"""
(tmp / "README.md").write_text(card, encoding="utf-8")
api = HfApi()
api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(tmp),
                  commit_message=f"One-to-one cricket reasoning dataset: {stats['rows_total']:,} rows covering "
                                 f"{stats['source_rows_covered']:,} source rows")
print(f"pushed {counts} + eval to https://huggingface.co/datasets/{args.repo_id}")
