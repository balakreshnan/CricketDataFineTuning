#!/usr/bin/env python3
"""Build the single-file Hecate vs Rhea comparison report.

  python report/build_compare.py      (run from rhea/)   ->  hecate-vs-rhea-report.html
Inputs (all local copies of what the two runs wrote):
  ../pulled/cricket/runs/general_sa-cricket.sft-full-10070553/   Hecate: metrics.jsonl (with timestamps), eval records
  pulled/log_history.json, pulled/rhea_step_times.json, pulled/eval.json   Rhea: trainer log, tqdm timings, eval digest
Wall-clock facts (job start/end times) come from the Slurm records quoted in cricket/README.md section 13.8 and the
Rhea job log; they are typed in FACTS below. Everything else is computed.
"""
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent.parent
HEC = root.parent / "pulled" / "cricket" / "runs" / "general_sa-cricket.sft-full-10070553"
STEPS, K = 1788, 4
TYPES = ["overall", "run_rate", "chase_rate", "milestone"]
EDGES = [0, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4096]

# ---------------- training curves ----------------
hm = [json.loads(l) for l in open(HEC / "metrics.jsonl")]
h_tr = [x for x in hm if "loss" in x]
h_dev = [{"s": x["step"], "v": x["eval_loss"]} for x in hm if "eval_loss" in x]
iv = [(b["time"] - a["time"]) / (b["step"] - a["step"]) for a, b in zip(h_tr, h_tr[1:])]
typical = statistics.median(iv)
t, h_train = typical * h_tr[0]["step"], []
for i, x in enumerate(h_tr):
    if i:
        per = iv[i - 1]
        # the run was split over two allocations: the 6 h queue wait + reload between them is not training time
        t += (typical if per > 120 else per) * (x["step"] - h_tr[i - 1]["step"])
    h_train.append({"s": x["step"], "loss": x["loss"], "lr": x["learning_rate"], "t": t})
h_total = t + typical * (STEPS - h_tr[-1]["step"])

rl = json.loads((root / "pulled" / "log_history.json").read_text())
rt = dict(json.loads((root / "pulled" / "rhea_step_times.json").read_text()))
keys = sorted(rt)


def r_time(s):
    lo = max(k for k in keys if k <= s)
    hi = min(k for k in keys if k >= s)
    return rt[lo] if lo == hi else rt[lo] + (rt[hi] - rt[lo]) * (s - lo) / (hi - lo)


r_train = [{"s": x["step"], "loss": x["loss"], "lr": x["learning_rate"], "t": r_time(x["step"])} for x in rl if "loss" in x]
r_dev = [{"s": x["step"], "v": x["eval_loss"]} for x in rl if "eval_loss" in x]
r_total = rt[STEPS]


def pace(train, total, win=60):
    """seconds per step over a sliding window of `win` steps (dev-loss pauses included, as in the real run)"""
    pts = [(0, 0.0)] + [(p["s"], p["t"]) for p in train] + [(STEPS, total)]
    out = []
    for i, (s, tt) in enumerate(pts):
        j = max(k for k in range(i + 1) if pts[k][0] <= s - win) if s >= win else None
        if j is not None and s > pts[j][0]:
            out.append({"s": s, "v": (tt - pts[j][1]) / (s - pts[j][0])})
    return out


def seg(train, total, a, b):
    pts = [(0, 0.0)] + [(p["s"], p["t"]) for p in train] + [(STEPS, total)]
    f = lambda s: next(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * (s - pts[i][0]) / (pts[i + 1][0] - pts[i][0])
                       for i in range(len(pts) - 1) if pts[i][0] <= s <= pts[i + 1][0])
    return (f(b) - f(a)) / (b - a)


# ---------------- benchmark ----------------
def digest_records(run):
    def load(tag):
        recs = []
        for p in sorted(glob.glob(str(run / "eval" / f"{tag}_rank*.jsonl"))):
            recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return recs

    base, ft = load("base"), load("ft")

    def hist(recs):
        h = [0] * (len(EDGES) - 1)
        for r in recs:
            h[max(k for k, e in enumerate(EDGES[:-1]) if r["n_tokens"] >= e)] += 1
        return h

    kb = {(r["id"], r["sample"]): r for r in base}
    kf = {(r["id"], r["sample"]): r for r in ft}
    pairs = {"both": 0, "fixed": 0, "broken": 0, "neither": 0}
    for key in kb.keys() & kf.keys():
        b, f = kb[key]["correct"], kf[key]["correct"]
        pairs["both" if b and f else "fixed" if f else "broken" if b else "neither"] += 1
    qb, qf = defaultdict(list), defaultdict(list)
    for r in base:
        qb[r["id"]].append(r["correct"])
    for r in ft:
        qf[r["id"]].append(r["correct"])
    ids = sorted(qb.keys() & qf.keys())
    diff = [sum(qf[i]) / len(qf[i]) - sum(qb[i]) / len(qb[i]) for i in ids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diff) for _ in ids) / len(ids) for _ in range(4000))
    pct = lambda rs, q: sorted(r["n_tokens"] for r in rs)[int(q / 100 * (len(rs) - 1))]
    return {"base": json.load(open(run / "eval" / "base_summary.json")), "ft": json.load(open(run / "eval" / "ft_summary.json")),
            "hist": {"edges": EDGES, "base": hist(base), "ft": hist(ft)}, "pairs": pairs,
            "ci": [100 * boots[int(0.025 * len(boots))], 100 * boots[int(0.975 * len(boots))]],
            "questions_changed": {"better": sum(d > 0 for d in diff), "worse": sum(d < 0 for d in diff), "same": sum(d == 0 for d in diff)},
            "tokens_p": {t: {str(q): pct(rs, q) for q in (50, 90, 99)} for t, rs in (("base", base), ("ft", ft))}}


EH = digest_records(HEC)
ER = json.loads((root / "pulled" / "eval.json").read_text())
assert ER["hist"]["edges"] == EDGES
slim = lambda s: {t: {k: v for k, v in s[t].items()} for t in TYPES}

# ---------------- wall-clock facts (minutes) ----------------
FACTS = {
    "hecate": {"name": "Hecate", "date": "2026-10-07", "gpus": 32, "nodes": 8, "lr": 2e-4,
               # job 711070 05:58-06:48 (training from about 06:05), queue 06:48-12:48, job 711522 12:48-16:06
               "segments": [["startup", 7], ["training", 43], ["queue", 360], ["training", 198]],
               "allocations": 2, "eval_engine": "vLLM", "eval_minutes": "3 to 4 minutes per model on one node"},
    "rhea": {"name": "Rhea", "date": "2026-11-11", "gpus": 32, "nodes": 8, "lr": 1e-4,
             # job 23649: submitted 06:36:00, started 06:39:03, training 06:46:33-11:16:33, ended 11:16:44
             "segments": [["queue", 3], ["startup", 7.5], ["training", 270]],
             "allocations": 1, "eval_engine": "Hugging Face generate", "eval_minutes": "25 to 40 minutes per model on four nodes"},
}
runs = {
    "hecate": {**FACTS["hecate"], "train": h_train, "dev": h_dev, "total_s": h_total, "pace": pace(h_train, h_total),
               "eval": {"base": slim(EH["base"]), "ft": slim(EH["ft"]), **{k: EH[k] for k in ("hist", "pairs", "ci", "questions_changed", "tokens_p")}}},
    "rhea": {**FACTS["rhea"], "train": r_train, "dev": r_dev, "total_s": r_total, "pace": pace(r_train, r_total),
             "eval": {"base": slim(ER["base"]), "ft": slim(ER["ft"]), **{k: ER[k] for k in ("hist", "pairs", "ci", "questions_changed", "tokens_p")}}},
}
for r in runs.values():
    r["train_h"] = r["total_s"] / 3600
    r["gpu_h"] = r["train_h"] * r["gpus"]
    r["wall_h"] = sum(m for _, m in r["segments"]) / 60
    r["early_pace"] = seg(r["train"], r["total_s"], 0, 272)
    r["late_pace"] = seg(r["train"], r["total_s"], 272, STEPS)

H, R = runs["hecate"], runs["rhea"]
acc = lambda r, m, t="overall": r["eval"][m][t]["accuracy"] * 100
tok = lambda r, m, t="overall": r["eval"][m][t]["mean_tokens"]
gain = lambda r: acc(r, "ft") - acc(r, "base")
cut = lambda r: (tok(r, "ft") / tok(r, "base") - 1) * 100
p = lambda x, d=2: f"{x:+.{d}f}"
base_gap = abs(acc(H, "base") - acc(R, "base"))
ft_gap = abs(acc(H, "ft") - acc(R, "ft"))
slower = (R["train_h"] / H["train_h"] - 1) * 100
ci_txt = lambda r: f"{p(r['eval']['ci'][0])} to {p(r['eval']['ci'][1])}"
tname = {"run_rate": "run rate", "chase_rate": "chase rate", "milestone": "milestone"}

insights = [
    f"<b>Both clusters produced the same adapter quality.</b> Fine-tuned accuracy is {acc(H, 'ft'):.2f}% on Hecate and "
    f"{acc(R, 'ft'):.2f}% on Rhea ({ft_gap:.2f} points apart), mean reasoning length {tok(H, 'ft'):,.0f} and {tok(R, 'ft'):,.0f} "
    f"tokens, and neither fine-tuned model hit the token cap. This held even though the learning rate differed "
    f"(2e-4 on Hecate, 1e-4 on Rhea) and the benchmark ran on different inference engines.",
    f"<b>The accuracy gain looks bigger on Rhea, but the difference is within noise.</b> Hecate measured {p(gain(H))} points (95% interval "
    f"{ci_txt(H)}) and Rhea {p(gain(R))} ({ci_txt(R)}). The base model is the same file on both clusters, yet it scored "
    f"{acc(H, 'base'):.2f}% on Hecate and {acc(R, 'base'):.2f}% on Rhea: a {base_gap:.2f}-point spread from sampling alone, "
    f"about the size of the difference between the two gains.",
    f"<b>The dependable effect is shorter reasoning: {cut(H):.0f}% on Hecate and {cut(R):.0f}% on Rhea.</b> The 99th-percentile "
    f"answer fell from {H['eval']['tokens_p']['base']['99']:,} to {H['eval']['tokens_p']['ft']['99']:,} tokens on Hecate and from "
    f"{R['eval']['tokens_p']['base']['99']:,} to {R['eval']['tokens_p']['ft']['99']:,} on Rhea.",
    f"<b>Hecate trained faster; Rhea finished sooner.</b> Training took {H['train_h']:.1f} h on Hecate ({H['gpu_h']:.0f} GPU-hours) and "
    f"{R['train_h']:.1f} h on Rhea ({R['gpu_h']:.0f} GPU-hours, {slower:.0f}% more) on the same 32 Vera Rubin GPUs. But Hecate's run "
    f"was split across two allocations with a 6-hour queue wait between them, so it took {H['wall_h']:.1f} h from start to "
    f"finish against {R['wall_h']:.1f} h on Rhea in a single allocation.",
    f"<b>Most of Rhea's extra time came from a slow start.</b> Over the first 272 steps Rhea averaged "
    f"{R['early_pace']:.1f} s per step against {H['early_pace']:.1f} s on Hecate. For the remaining steps the gap narrowed to "
    f"{R['late_pace']:.1f} s against {H['late_pace']:.1f} s, so Rhea stayed {(R['late_pace'] / H['late_pace'] - 1) * 100:.0f}% slower "
    f"even after warming up. The cause was not diagnosed on either count.",
    "<b>By question type:</b> " + "; ".join(
        f"{tname[t]} {acc(H, 'base', t):.2f} to {acc(H, 'ft', t):.2f}% on Hecate and {acc(R, 'base', t):.2f} to {acc(R, 'ft', t):.2f}% on Rhea"
        for t in TYPES[1:]) + ". Chase rate is the weakest type on both clusters, before and after; milestone improved on "
    "both; run rate moved in opposite directions by a fraction of a point.",
    f"<b>The inference engine mattered more than the cluster for evaluation time.</b> Hecate benchmarked with vLLM in about "
    f"{FACTS['hecate']['eval_minutes']}; Rhea used Hugging Face generate and needed about {FACTS['rhea']['eval_minutes']}.",
    "<b>What is not comparable:</b> the dev-loss values. Each run held out a different random 1,000 rows, and Rhea rebuilt "
    "the training targets from the published dataset fields, so Hecate's 0.318 and Rhea's 0.313 are not a ranking. "
    "The benchmark covers three of the six question types on both clusters.",
]

rows = lambda a, b, c: [a, b, c, b == c]
config = [
    rows("Model", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    rows("Training data", "458,702 rows (457,689 train, 1,000 dev, 13 dropped)", "458,702 rows (457,689 train, 1,000 dev, 13 dropped)"),
    rows("Target text", "the model's raw completion (completion_text)", "rebuilt from the reasoning and response fields"),
    rows("LoRA", "r 32, alpha 64, dropout 0.05; 217.6M trainable (0.79%)", "r 32, alpha 64, dropout 0.05; 217.6M trainable (0.79%)"),
    rows("Batch and steps", "8 per GPU x 32 GPUs = 256; 1,788 steps", "8 per GPU x 32 GPUs = 256; 1,788 steps"),
    rows("Learning rate", "2e-4, 3% warmup, cosine", "1e-4, 3% warmup, cosine"),
    rows("GPUs", "8 nodes x 4 Vera Rubin, about 280 GiB each", "8 nodes x 4 Vera Rubin, about 280 GiB each"),
    rows("Allocations", "2 (stopped at step 272, resumed 6 h later)", "1"),
    rows("Benchmark", "600 questions x 4 samples, vLLM, adapter merged", "600 questions x 4 samples, Hugging Face generate, adapter merged"),
    rows("Adapter on Hugging Face", "Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA (public)", "Balab2021/Qwen3.8-27B-Cricket-Reasoning-Rhea-LoRA (private)"),
]

data = {"steps": STEPS, "k": K, "types": TYPES, "runs": runs, "insights": insights, "config": config,
        "lede": (f"The same LoRA fine-tune of Qwen3.8-27B on the cricket reasoning dataset, run on Hecate in October and on Rhea in "
                 f"November. Both reached about {acc(R, 'ft'):.1f}% accuracy with answers roughly {abs(cut(R)):.0f}% shorter; they "
                 f"differ in how long it took."),
        "foot": ("Hecate: run general_sa-cricket.sft-full-10070553 (metrics.jsonl with timestamps, per-sample eval records). "
                 "Rhea: run cricket-lora-full-1111 (trainer log history, progress-bar timings from the job log, per-sample eval "
                 "records). Hecate's training time excludes the queue wait between its two allocations.")}
html = (root / "report" / "compare_template.html").read_text(encoding="utf-8")
assert "/*DATA*/null" in html
out = root / "hecate-vs-rhea-report.html"
out.write_text(html.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8", newline="\n")
print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")
for k, r in runs.items():
    print(k, f"train {r['train_h']:.2f} h, wall {r['wall_h']:.2f} h, gpu-h {r['gpu_h']:.0f}, early {r['early_pace']:.1f} late {r['late_pace']:.1f} s/step,",
          f"acc {acc(r, 'base'):.2f}->{acc(r, 'ft'):.2f} ci {r['eval']['ci']}, tok {tok(r, 'base'):.0f}->{tok(r, 'ft'):.0f}, q {r['eval']['questions_changed']}")
for i in insights:
    print("-", i[:170])
