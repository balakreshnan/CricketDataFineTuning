#!/usr/bin/env python3
"""Condense <run_dir>/eval/{base,ft}_rank*.jsonl into one small JSON for the HTML report (stdlib only).

Prints JSON: summaries, token histogram, paired outcomes, a bootstrap CI for the accuracy change (resampling
questions, since the 4 samples of a question are not independent) and one side-by-side example.
"""
import glob
import json
import random
import sys
from collections import defaultdict

run = sys.argv[1]


def load(tag):
    recs = []
    for p in sorted(glob.glob(f"{run}/eval/{tag}_rank*.jsonl")):
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


base, ft = load("base"), load("ft")
out = {"base": json.load(open(f"{run}/eval/base_summary.json")), "ft": json.load(open(f"{run}/eval/ft_summary.json")),
       "n_base": len(base), "n_ft": len(ft)}

edges = [0, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4096]


def hist(recs):
    h = [0] * (len(edges) - 1)
    for r in recs:
        i = max(k for k, e in enumerate(edges[:-1]) if r["n_tokens"] >= e)
        h[i] += 1
    return h


out["hist"] = {"edges": edges, "base": hist(base), "ft": hist(ft)}

kb = {(r["id"], r["sample"]): r for r in base}
kf = {(r["id"], r["sample"]): r for r in ft}
pairs = {"both": 0, "fixed": 0, "broken": 0, "neither": 0}
for key in kb.keys() & kf.keys():
    b, f = kb[key]["correct"], kf[key]["correct"]
    pairs["both" if b and f else "fixed" if f else "broken" if b else "neither"] += 1
out["pairs"] = pairs

# per-question accuracy -> paired bootstrap over questions for the overall accuracy change (in points)
qb, qf = defaultdict(list), defaultdict(list)
for r in base:
    qb[r["id"]].append(r["correct"])
for r in ft:
    qf[r["id"]].append(r["correct"])
ids = sorted(qb.keys() & qf.keys())
diff = [sum(qf[i]) / len(qf[i]) - sum(qb[i]) / len(qb[i]) for i in ids]
rng = random.Random(0)
boots = sorted(sum(rng.choice(diff) for _ in ids) / len(ids) for _ in range(4000))
out["ci"] = [100 * boots[int(0.025 * len(boots))], 100 * boots[int(0.975 * len(boots))]]
out["questions_changed"] = {"better": sum(d > 0 for d in diff), "worse": sum(d < 0 for d in diff),
                            "same": sum(d == 0 for d in diff)}
out["tokens_p"] = {t: {str(q): sorted(r["n_tokens"] for r in rs)[int(q / 100 * (len(rs) - 1))] for q in (50, 90, 99)}
                   for t, rs in (("base", base), ("ft", ft))}

# example: both correct on sample 0, largest token saving among mid-length base answers (readable size)
qs = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2], encoding="utf-8")}
cands = [(kb[k]["n_tokens"] - kf[k]["n_tokens"], k) for k in kb.keys() & kf.keys()
         if kb[k]["correct"] and kf[k]["correct"] and 500 <= kb[k]["n_tokens"] <= 1200 and k[1] == 0]
if cands:
    cands.sort()
    _, k = cands[int(len(cands) * 0.75)]   # a typical-to-good saving, not the single most extreme case
    q = qs[k[0]]
    pick = lambda r: {x: r[x] for x in ("text", "n_tokens", "pred", "correct")}
    out["example"] = {"qtype": q["qtype"], "gold": q["gold"], "question": q["messages"][1]["content"],
                      "base": pick(kb[k]), "ft": pick(kf[k]),
                      "why": "Picked automatically: sample 0 of a question both models answered correctly, at the 75th "
                             "percentile of token savings among base answers of 500 to 1,200 tokens."}
print(json.dumps(out))
