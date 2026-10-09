"""Build the question sets (login node, tools venv):
   $TOOLS_VENV/bin/python prep_questions.py [--train-per-type 2000] [--test-per-type 100]
Writes $PROJ/data/questions_{train,test}.jsonl and question_stats.json."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olympics_qa import QTYPES, build_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-per-type", type=int, default=2000)
ap.add_argument("--test-per-type", type=int, default=100)
ap.add_argument("--out", default=os.path.join(os.environ["PROJ"], "data"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
train, test, stats = build_splits(os.path.join(os.environ["RAW_DIR"], "athlete_events.csv"),
                                  a.train_per_type, a.test_per_type)
for name, rows in (("train", train), ("test", test)):
    with open(os.path.join(a.out, f"questions_{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
assert not {r["id"] for r in train} & {r["id"] for r in test}
json.dump(stats, open(os.path.join(a.out, "question_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
for qt in QTYPES:
    ex = next(r for r in test if r["qtype"] == qt)
    print(f"\n===== {qt}  gold={ex['gold']}\n{ex['messages'][1]['content'][:1200]}")
