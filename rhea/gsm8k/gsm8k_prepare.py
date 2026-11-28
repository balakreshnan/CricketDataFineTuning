#!/usr/bin/env python
"""Download GSM8K and write it in the layout of the cricket dataset, so the same training and evaluation code runs on it.

usage: gsm8k_prepare.py <out_dir>
Source: the original release (github.com/openai/grade-school-math, MIT licence): 7,473 training and 1,319 test problems.
Writes <out_dir>/data/train-00000.jsonl (gold in "gold_answer") and <out_dir>/data/eval-00000.jsonl (gold in "gold"),
rows {"id", "qtype": "gsm8k", "answer_kind": "int", "messages": [system, user]}. The gold answer is the number after
"####" in the reference solution; the checker compares numbers exactly. Standard library only.
"""
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"
SYSTEM_PROMPT = ("You are a careful problem solver. Work through the problem step by step, then give the final answer "
                 "on its own line as 'Final answer: <number>', a plain number without units, symbols or commas.")
out = Path(sys.argv[1])
(out / "data").mkdir(parents=True, exist_ok=True)
(out / "raw").mkdir(exist_ok=True)
for split, name, key in (("train", "train-00000.jsonl", "gold_answer"), ("test", "eval-00000.jsonl", "gold")):
    raw = out / "raw" / f"{split}.jsonl"
    if not raw.is_file():
        urllib.request.urlretrieve(URL.format(split), raw)
    rows, bad = [], 0
    for i, line in enumerate(open(raw, encoding="utf-8")):
        r = json.loads(line)
        try:
            gold = float(r["answer"].split("####")[-1].strip().replace(",", "").replace("$", ""))
        except ValueError:
            bad += 1
            continue
        rows.append({"id": f"gsm8k-{split}-{i}", "qtype": "gsm8k", "answer_kind": "int", key: gold,
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": r["question"].strip()}]})
    with open(out / "data" / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} problems written to {out / 'data' / name} ({bad} skipped: gold answer not a number)")
