"""Build the 1:1 row-anchored question set:  python prep_rows.py [--fallback [N]]
Writes $PROJ/data/questions_rows_{train,test}.jsonl + rows_stats.json (every unique source row -> one question).
--fallback N writes questions_rows_fallback[N].jsonl instead (no suffix for N=1): each row's template N steps after
its usual one, for rows still without a correct trace after sampling (build_dataset.py --unsolved-out lists them).
Runs on a compute node (the login node kills long CPU-heavy processes): see rows.sbatch."""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rows_qa import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fallback", type=int, nargs="?", const=1, default=0, help="template offset (1 = next template)")
a = ap.parse_args()
out = os.path.join(os.environ["PROJ"], "data")
csv_path = os.path.join(os.environ["RAW_DIR"], "athlete_events.csv")

if not a.fallback:
    with open(f"{out}/questions_rows_train.jsonl", "w") as ftr, open(f"{out}/questions_rows_test.jsonl", "w") as fte:
        stats = build(csv_path, ftr, fte)
    json.dump(stats, open(f"{out}/rows_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    with open(f"{out}/questions_rows_train.jsonl") as f:
        seen = set()
        for l in f:
            r = json.loads(l)
            if r["qtype"] not in seen:
                seen.add(r["qtype"])
                c = r["messages"][1]["content"]
                print(f"\n===== {r['qtype']} gold={r['gold']}\n{c[:400]}\n...\n{c[-220:]}")
else:
    want = {json.loads(l)["source"]["row_index"] for l in open(f"{out}/unsolved_rows.jsonl")}
    buf = io.StringIO()
    stats = build(csv_path, buf, buf, skip_first=a.fallback)
    sfx = "" if a.fallback == 1 else str(a.fallback)
    n = 0
    with open(f"{out}/questions_rows_fallback{sfx}.jsonl", "w") as f:
        for l in buf.getvalue().splitlines():
            r = json.loads(l)
            if r["source"]["row_index"] in want:
                r["id"] += f"-fb{sfx}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"fallback questions for {n} of {len(want)} unsolved rows")
