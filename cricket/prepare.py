#!/usr/bin/env python
"""Download Valarmathy/CricketData and build the verifiable questions (train/test split by match).

Writes $DATA_DIR/raw/..., $DATA_DIR/questions/{train,test}.jsonl and questions/stats.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cricket_qa as Q  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=os.environ.get("DATA_DIR"))
p.add_argument("--n_train", type=int, default=1500, help="source rows for training (3 questions each)")
p.add_argument("--n_test", type=int, default=200, help="source rows from held-out matches (3 questions each)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

data = Path(args.data_dir)
csv_path = hf_hub_download("Valarmathy/CricketData", "raw/ball_by_ball_it20.csv", repo_type="dataset",
                           local_dir=str(data / "raw"))
print(f"source csv: {csv_path} ({Path(csv_path).stat().st_size / 2**20:.0f} MiB)", flush=True)
train, test, stats = Q.build_splits(csv_path, args.n_train, args.n_test, seed=args.seed)
qdir = data / "questions"
qdir.mkdir(parents=True, exist_ok=True)
Q.write_jsonl(qdir / "train.jsonl", train)
Q.write_jsonl(qdir / "test.jsonl", test)
stats.update({"seed": args.seed, "source_dataset": "Valarmathy/CricketData", "source_file": "raw/ball_by_ball_it20.csv"})
(qdir / "stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
print("example:", json.dumps(train[0], ensure_ascii=False)[:600])
