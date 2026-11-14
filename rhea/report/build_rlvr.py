#!/usr/bin/env python
"""Build rhea/cricket-rlvr-report.html from pulled/rlvr/{metrics.jsonl,summary.json,gpu_usage.json}.

gpu_usage.json comes from export_sysmetrics.py: W&B sampled only the node where process 0 ran (4 of the 32 GPUs).
"""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "pulled" / "rlvr"
ms = [json.loads(l) for l in open(src / "metrics.jsonl", encoding="utf-8") if l.startswith("{")]
sm = json.loads((src / "summary.json").read_text(encoding="utf-8"))
g = json.loads((src / "gpu_usage.json").read_text(encoding="utf-8"))

T0 = g["t"][0] - 1                      # training loop start (first sample lands ~1 s after it)
rel = [t - T0 for t in g["t"]]
gpus = g["gpus"]
util = {i: g["series"][f"{i}.util"] for i in gpus}
mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731

steps, roll_u, upd_u = [], [], []
for m in ms:
    end = m["elapsed_h"] * 3600
    start = end - m["sec_step"]
    mid = start + m["sec_rollout"]
    idx = [k for k, t in enumerate(rel) if start <= t <= end]
    if not idx:
        idx = [min(range(len(rel)), key=lambda k: abs(rel[k] - (start + end) / 2))]
    per = [mean([util[i][k] for k in idx if util[i][k] is not None]) for i in gpus]
    for k in idx:
        v = mean([util[i][k] for i in gpus if util[i][k] is not None])
        if start <= rel[k] <= end:
            (roll_u if rel[k] < mid else upd_u).append(v)
    steps.append({"s": m["step"], "acc": m["rollout_accuracy"], "tok": m["mean_tokens"], "cut": m["truncated_rate"],
                  "mixed": m["groups_mixed_rate"], "rew": m["reward_mean"], "ro": m["sec_rollout"], "up": m["sec_update"],
                  "st": m["sec_step"], "el": end, "util": [None if x is None else round(x, 1) for x in per]})

gb = 1024 ** 3
ser = g["series"]
D = {
    "steps": steps,
    "job": {"id": 39000, "elapsed_s": 4925, "limit_h": 5, "start": "06:16:06", "end": "07:38:11", "gpus": 32, "gpu_gb": 286,
            "nodes": ["rhea0271", "rhea0272", "rhea0273", "rhea0274", "rhea0275", "rhea0276", "rhea0281", "rhea0288"],
            "eval_gpus": 16, "eval_s": 243},
    "gpu": {
        "sampled_node": "rhea0271", "samples": len(rel), "interval_s": 15,
        "util_avg": [round(mean([x for x in util[i] if x is not None]), 1) for i in gpus],
        "util_rollout": round(mean(roll_u), 1), "util_update": round(mean(upd_u), 1),
        "mem_peak_gb": [round(max(x for x in ser[f"{i}.mem_bytes"] if x is not None) / gb, 1) for i in gpus],
        "mem_avg_gb": [round(mean([x for x in ser[f"{i}.mem_bytes"] if x is not None]) / gb, 1) for i in gpus],
        "power_avg_w": round(mean([mean([x for x in ser[f"{i}.power_w"] if x is not None]) for i in gpus])),
        "power_peak_w": round(max(max(x for x in ser[f"{i}.power_w"] if x is not None) for i in gpus)),
        "t_min": [round(t / 60, 2) for t in rel],
        "util": [ser[f"{i}.util"] for i in gpus],
        "mem_gb": [[None if x is None else round(x / gb, 1) for x in ser[f"{i}.mem_bytes"]] for i in gpus],
        "power_w": [[None if x is None else round(x) for x in ser[f"{i}.power_w"]] for i in gpus],
        "temp_peak": [round(max(x for x in ser[f"{i}.temp_c"] if x is not None)) for i in gpus],
        "temp_peak_c": round(max(max(x for x in ser[f"{i}.temp_c"] if x is not None) for i in gpus)),
    },
    "bench": {"base": sm["base"], "ft": sm["ft"]},
    "train": {k: v for k, v in sm["train"].items() if k != "args"},
    "args": {k: sm["train"]["args"][k] for k in ("questions_per_gpu", "group", "max_new_tokens", "len_coef", "lr", "lora_r", "steps")},
}
print("util avg per sampled GPU", D["gpu"]["util_avg"], "rollout", D["gpu"]["util_rollout"], "update", D["gpu"]["util_update"],
      "samples in phases", len(roll_u), len(upd_u))
print("mem peak GB", D["gpu"]["mem_peak_gb"], "power avg/peak", D["gpu"]["power_avg_w"], D["gpu"]["power_peak_w"])
html = (here / "rlvr_template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(D, separators=(",", ":")))
out = here.parent / "cricket-rlvr-report.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("wrote", out, out.stat().st_size, "bytes")
