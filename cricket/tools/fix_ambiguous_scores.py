#!/usr/bin/env python
"""Clarify ambiguous low scores in unsolved questions, in place.

A score written "1/3" is read as runs/wickets in most countries but wickets/runs in Australia; when runs <= wickets
both readings are plausible and Qwen3.8 consistently picks wickets/runs. For every question listed in retry.jsonl
whose score has runs <= wickets, rewrite "they are R/W." as "they are R/W (R run(s) for the loss of W wicket(s))."
in the question set (all/train/test.jsonl) and in retry.jsonl, so the re-sampled traces match the stored question.
"""
import json
import sys
from pathlib import Path

qdir, gdir = Path(sys.argv[1]), Path(sys.argv[2])
retry = [json.loads(l) for l in open(gdir / "retry.jsonl", encoding="utf-8") if l.strip()]
fix = {q["id"] for q in retry if q["source"]["innings_runs"] <= q["source"]["innings_wickets"]}
print(f"{len(retry)} unsolved questions, {len(fix)} with an ambiguous score (runs <= wickets)")


def clarify(q):
    if q["id"] not in fix or q.get("note") == "score_clarified":
        return q
    r, w = q["source"]["innings_runs"], q["source"]["innings_wickets"]
    old = f"they are {r}/{w}. "
    new = f"they are {r}/{w} ({r} run{'s' if r != 1 else ''} for the loss of {w} wicket{'s' if w != 1 else ''}). "
    text = q["messages"][1]["content"]
    assert text.count(old) == 1, (q["id"], old)
    q["messages"][1]["content"] = text.replace(old, new)
    q["note"] = "score_clarified"
    return q


for path in [qdir / "all.jsonl", qdir / "train.jsonl", qdir / "test.jsonl", gdir / "retry.jsonl"]:
    rows = [clarify(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for q in rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"{path.name}: {sum(q.get('note') == 'score_clarified' for q in rows)} clarified")
