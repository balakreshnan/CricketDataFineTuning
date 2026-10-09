"""Small SFT set for the training throughput/memory benchmark only (not published, not the real run):
shortest correct trace per question from an earlier generation dir -> $PROJ/data/sft-bench/{training,validation}.jsonl
   python3 mk_bench_sft.py $PROJ/gen/full"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olympics_qa import SYSTEM_PROMPT  # noqa: E402

gen = sys.argv[1]
proj = os.environ["PROJ"]
qs = {json.loads(l)["id"]: json.loads(l) for l in open(f"{proj}/data/questions_train.jsonl")}
best = defaultdict(lambda: None)
for f in glob.glob(f"{gen}/train.shard*.jsonl"):
    for l in open(f):
        s = json.loads(l)
        if s["correct"] and s["finish_reason"] == "stop" and s["completion_tokens"] < 6000:
            b = best[s["id"]]
            if b is None or s["completion_tokens"] < b["completion_tokens"]:
                best[s["id"]] = s
out = f"{proj}/data/sft-bench"
os.makedirs(out, exist_ok=True)
with open(f"{out}/training.jsonl", "w") as ft, open(f"{out}/validation.jsonl", "w") as fv:
    for i, (qid, s) in enumerate(sorted(best.items())):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": qs[qid]["messages"][1]["content"]},
                {"role": "assistant", "reasoning_content": s["reasoning"].strip(), "content": s["content"].strip()}]
        (fv if i % 50 == 0 else ft).write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
print(len(best), "rows ->", out)
