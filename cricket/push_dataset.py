#!/usr/bin/env python
"""Push the generated cricket reasoning dataset to a private Hugging Face dataset repo.

Run only after the dataset has been reviewed and approved (submit.sh push-dataset passes --confirm).
Internal fields (exact completion text used for training) are dropped; train/test go up as JSONL.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--final_dir", required=True)
p.add_argument("--repo_id", required=True)
p.add_argument("--confirm", action="store_true", help="required: the dataset was reviewed and approved")
args = p.parse_args()
if not args.confirm:
    raise SystemExit("refusing to push without --confirm (dataset must be reviewed and approved first)")

final = Path(args.final_dir)
stats = json.loads((final / "stats.json").read_text())
tmp = Path(tempfile.mkdtemp())
(tmp / "data").mkdir()
with open(final / "train.jsonl", encoding="utf-8") as fi, open(tmp / "data" / "train.jsonl", "w", encoding="utf-8") as fo:
    for line in fi:
        r = json.loads(line)
        r.pop("completion_text", None)
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
shutil.copy(final / "test.jsonl", tmp / "data" / "test.jsonl")
shutil.copy(final / "stats.json", tmp / "stats.json")
base = stats.get("base_model_on_train_questions", {}).get("overall", {})
card = f"""---
license: cc0-1.0
task_categories: [question-answering, text-generation]
tags: [reasoning, cricket, t20, synthetic, chain-of-thought, qwen]
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: test
    path: data/test.jsonl
---

# Cricket T20I reasoning traces (Qwen3.8-27B, verified)

Reasoning dataset built from the ball-by-ball T20 International data in
[Valarmathy/CricketData](https://huggingface.co/datasets/Valarmathy/CricketData) (CC0).

**How it was made**
- Matches with rain-revised targets were removed; matches were split into train and held-out test sets.
- Each sampled delivery (source row) became **3 questions** with answers computed exactly from the data:
  - `run_rate`: current run rate, with overs given in cricket notation (14.3 = 14 overs 3 balls)
  - `chase_rate`: required run rate to win (second innings) or the rate scored over the rest of the innings (first innings)
  - `milestone`: balls the striker needs to reach their next 50 at their current strike rate
- `Qwen/Qwen3.8-27B` answered each training question {stats['k']} times in thinking mode (temperature 1.0, top_p 0.95,
  top_k 20, up to {stats['max_new_tokens']} tokens). Answers were checked against the exact values and the
  **shortest correct** reasoning trace was kept.

**Size**: {stats['rows_kept']} training rows from {stats['questions']} questions ({stats['source_rows']} source rows;
{stats['source_rows_with_all_3']} contribute all 3 questions); {stats['test_questions']} test questions (questions and
gold answers only, from matches not used for training). Base-model accuracy on the training questions while generating:
{100 * base.get('accuracy', 0):.1f}% per sample, pass@{stats['k']} {100 * base.get(f"pass@{stats['k']}", 0):.1f}%.

**Fields (train)**: `id`, `qtype`, `question`, `messages` (system, user, assistant with `<think>` reasoning),
`reasoning`, `response`, `final_answer`, `gold_answer`, `answer_kind`, `reasoning_tokens`, `samples_generated`,
`samples_correct`, `source` (the originating ball-by-ball row: match, innings, over, ball, score, batter...).
"""
(tmp / "README.md").write_text(card, encoding="utf-8")
api = HfApi()
api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(tmp),
                  commit_message=f"Cricket reasoning dataset: {stats['rows_kept']} train rows, {stats['test_questions']} test questions")
print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")
