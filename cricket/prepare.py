#!/usr/bin/env python
"""Download Valarmathy/CricketData and build the verifiable questions (train/test split by match).

Writes $DATA_DIR/raw/..., $DATA_DIR/questions/{train,test}.jsonl and questions/stats.json.
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=os.environ.get("DATA_DIR"))
p.add_argument("--n_train", type=int, default=1500, help="source rows for training (3 questions each)")
p.add_argument("--n_test", type=int, default=200, help="source rows from held-out matches (3 questions each)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_name", default="questions", help="output folder under data_dir (keeps question sets separate)")
p.add_argument("--mode", choices=["sample", "full"], default="sample",
               help="sample: n_train/n_test source rows x 3 questions; full: every source row >= 1 question")
p.add_argument("--target_total", type=int, default=500_000, help="full mode: total questions (rows beyond one-to-one get a 2nd)")
args = p.parse_args()

SOURCE_REPO, SOURCE_FILE, SOURCE_ROWS = "Valarmathy/CricketData", "raw/ball_by_ball_it20.csv", 425_119

data = Path(args.data_dir)
csv_path = hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset", local_dir=str(data / "raw"))
with open(csv_path, newline="", encoding="utf-8") as f:
    n_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header; csv.reader handles quoted newlines
if n_rows != SOURCE_ROWS:
    raise SystemExit(f"{SOURCE_REPO}/{SOURCE_FILE} has {n_rows:,} rows, expected {SOURCE_ROWS:,} - source changed, stopping")
source_rev = HfApi().dataset_info(SOURCE_REPO).sha
print(f"source: {SOURCE_REPO} @ {source_rev}, {SOURCE_FILE}: {n_rows:,} rows "
      f"({Path(csv_path).stat().st_size / 2**20:.0f} MiB) - verified", flush=True)
qdir = data / args.out_name
qdir.mkdir(parents=True, exist_ok=True)
if args.mode == "full":
    # one-to-one: every source row gets >= 1 question; train + test both get reasoning traces
    train, test, stats = Q.build_full(csv_path, target_total=args.target_total, seed=args.seed)
    _, bench, _ = Q.build_splits(csv_path, 1500, 200, seed=args.seed)  # the fixed 600-question benchmark
    rng = random.Random(args.seed + 2)
    pilot = []
    for qt in Q.ALL_QTYPES:  # stratified pilot: 40 per type (all 40 legal_balls rows are in train/test)
        pool = [q for q in train + test if q["qtype"] == qt]
        pilot += rng.sample(pool, min(40, len(pool)))
    Q.write_jsonl(qdir / "train.jsonl", train)
    Q.write_jsonl(qdir / "test.jsonl", test)
    Q.write_jsonl(qdir / "all.jsonl", train + test)
    Q.write_jsonl(qdir / "eval.jsonl", bench)
    Q.write_jsonl(qdir / "pilot.jsonl", pilot)
    stats.update({"benchmark_questions": len(bench), "pilot_questions": len(pilot), "target_total": args.target_total})
else:
    train, test, stats = Q.build_splits(csv_path, args.n_train, args.n_test, seed=args.seed)
    Q.write_jsonl(qdir / "train.jsonl", train)
    Q.write_jsonl(qdir / "test.jsonl", test)
stats.update({"seed": args.seed, "source_dataset": SOURCE_REPO, "source_revision": source_rev,
              "source_file": SOURCE_FILE, "source_rows": n_rows})
(qdir / "stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print("example:", json.dumps(train[0], ensure_ascii=False)[:600])
