#!/usr/bin/env python
"""Build rhea/qwen36-rlvr-report.html from pulled/rlvr36/{metrics.jsonl,summary.json,gpulog/*.csv}.

gpulog/<node>-<job>.csv: nvidia-smi every 15 s on every node (timestamp, index, util %, memory.used MiB, memory.total MiB,
power W, temp C). Memory is shown in thousands of MiB so that a full GPU reads 286.5, matching the "286 GB" label.
"""
import datetime as dt
import glob
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "pulled" / "rlvr36"
ms = [json.loads(l) for l in open(src / "metrics.jsonl", encoding="utf-8") if l.startswith("{")]
sm = json.loads((src / "summary.json").read_text(encoding="utf-8"))
prev = json.loads((here.parent / "pulled" / "rlvr" / "summary.json").read_text(encoding="utf-8"))

JOB = 40421
T0 = dt.datetime(2026, 11, 13, 15, 1, 57)            # "world 32; ..." log line: the training loop starts here
ELAPSED = 2 * 3600 + 38 * 60 + 46
mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
r1 = lambda x: None if x is None else round(x, 1)  # noqa: E731

nodes = sorted(Path(f).name.split("-")[0] for f in glob.glob(str(src / "gpulog" / f"*-{JOB}.csv")))
loop_end = ms[-1]["elapsed_h"] * 3600
NB = math.ceil(loop_end / 60)
# samples[g] = list of (t, util, mem_gb, power, temp) for global GPU g = node_index * 4 + gpu_index
samples = [[] for _ in range(len(nodes) * 4)]
for ni, n in enumerate(nodes):
    for line in open(src / "gpulog" / f"{n}-{JOB}.csv", encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        try:
            t = (dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f") - T0).total_seconds()
            row = (t, float(p[2]), float(p[3]) / 1000, float(p[5]), float(p[6]))
        except ValueError:
            continue
        if 0 <= t <= loop_end:
            samples[ni * 4 + int(p[1])].append(row)
NG = len(samples)


def binned(g, col):
    out = [[] for _ in range(NB)]
    for r in samples[g]:
        out[min(NB - 1, int(r[0] // 60))].append(r[col])
    return [mean(v) for v in out]


util_b = [binned(g, 1) for g in range(NG)]
mem_b = [binned(g, 2) for g in range(NG)]
pow_b = [binned(g, 3) for g in range(NG)]


def node_avg(arr, ni):
    return [r1(mean([arr[g][m] for g in range(ni * 4, ni * 4 + 4) if arr[g][m] is not None])) for m in range(NB)]


steps, roll_u, upd_u = [], [], []
for m in ms:
    end = m["elapsed_h"] * 3600
    start = end - m["sec_step"]
    mid = start + m["sec_rollout"]
    per = []
    for g in range(NG):
        v = [r[1] for r in samples[g] if start <= r[0] <= end]
        per.append(None if not v else round(mean(v)))
        for r in samples[g]:
            if start <= r[0] <= end:
                (roll_u if r[0] < mid else upd_u).append(r[1])
    steps.append({"s": m["step"], "acc": m["rollout_accuracy"], "tok": m["mean_tokens"], "cut": m["truncated_rate"],
                  "mixed": m["groups_mixed_rate"], "rew": m["reward_mean"], "ro": m["sec_rollout"], "up": m["sec_update"],
                  "st": m["sec_step"], "el": end, "util": per})

gpu_util = [mean([r[1] for r in s]) for s in samples]
B, F = sm["base"], sm["ft"]
lo = min(B[t]["accuracy"] for t in ("overall", "run_rate", "chase_rate", "milestone"))
lo = min(lo, min(F[t]["accuracy"] for t in ("overall", "run_rate", "chase_rate", "milestone")))
b0 = math.floor((lo - 0.005) * 50) / 50
nt = round((1 - b0) / 0.02)
stp = 0.02 if nt <= 6 else 0.04 if nt <= 12 else 0.1
b0 = 1 - math.ceil((1 - b0) / stp) * stp
bacc = [round(b0 + i * stp, 2) for i in range(round((1 - b0) / stp) + 1)]
tmaxb = max(max(B[t]["mean_tokens"], F[t]["mean_tokens"]) for t in ("overall", "run_rate", "chase_rate", "milestone"))
bstep = 250 if tmaxb <= 1000 else 500 if tmaxb <= 2500 else 1000
btok = [i * bstep for i in range(math.ceil(tmaxb / bstep) + 1)]
second = sorted(s["st"] for s in steps)[-2]
tmax = math.ceil(second / 180) * 180

pa = prev["train"]["args"]
D = {
    "model": "Qwen3.6-35B-A3B", "trainable": "38.3M (0.11%)",
    "steps": steps,
    "axes": {"steps": [1, 10, 20, 30, 40, 50, 60], "tok": [0, 600, 1200, 1800, 2400], "mix": [0, 10, 20, 30, 40], "tmax": tmax,
             "tticks": [round(tmax * i / 4) for i in range(5)], "bacc": bacc, "btok": btok},
    "job": {"id": JOB, "elapsed_s": ELAPSED, "limit_h": 5, "start": "15:00:23", "end": "17:39:09", "gpus": NG, "gpu_gb": 286.5,
            "nodes": nodes},
    "gpu": {
        "units": nodes, "samples": sum(len(s) for s in samples), "interval_s": 15,
        "util_all": round(mean([r[1] for s in samples for r in s]), 1),
        "util_gpu_min": round(min(gpu_util)), "util_gpu_max": round(max(gpu_util)),
        "util_avg": [round(mean(gpu_util[i * 4:i * 4 + 4]), 1) for i in range(len(nodes))],
        "util_rollout": round(mean(roll_u), 1), "util_update": round(mean(upd_u), 1),
        "mem_peak_gb": [round(max(r[2] for g in range(i * 4, i * 4 + 4) for r in samples[g]), 1) for i in range(len(nodes))],
        "mem_avg_gb": [round(mean([r[2] for g in range(i * 4, i * 4 + 4) for r in samples[g]]), 1) for i in range(len(nodes))],
        "power_avg_w": round(mean([r[3] for s in samples for r in s])),
        "power_peak_w": round(max(r[3] for s in samples for r in s)),
        "temp_peak": [round(max(r[4] for g in range(i * 4, i * 4 + 4) for r in samples[g])) for i in range(len(nodes))],
        "temp_peak_c": round(max(r[4] for s in samples for r in s)),
        "t_min": [m + 0.5 for m in range(NB)],
        "util": [node_avg(util_b, i) for i in range(len(nodes))],
        "mem_gb": [node_avg(mem_b, i) for i in range(len(nodes))],
        "power_w": [[None if x is None else round(x) for x in node_avg(pow_b, i)] for i in range(len(nodes))],
        "heat": [[None if x is None else round(x) for x in util_b[g]] for g in range(NG)],
    },
    "bench": {"base": B, "ft": F},
    "train": {k: v for k, v in sm["train"].items() if k != "args"},
    "args": {k: sm["train"]["args"][k] for k in ("questions_per_gpu", "group", "max_new_tokens", "len_coef", "lr", "lora_r", "steps", "train_batch")},
    "prev": {"name": "Qwen3.8-27B", "base": prev["base"]["overall"], "ft": prev["ft"]["overall"], "steps": pa["steps"],
             "rollouts": prev["train"]["rollouts_total"], "max_new_tokens": pa["max_new_tokens"], "trainable": "217.6M (0.79%)",
             "elapsed_s": 4925},
}
g = D["gpu"]
print("nodes", nodes, "samples", g["samples"], "util all", g["util_all"], "per-GPU range", g["util_gpu_min"], g["util_gpu_max"],
      "rollout/update", g["util_rollout"], g["util_update"])
print("mem peak per node", g["mem_peak_gb"], "power avg/peak", g["power_avg_w"], g["power_peak_w"], "axes", D["axes"])
html = (here / "rlvr36_template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(D, separators=(",", ":")))
out = here.parent / "qwen36-rlvr-report.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("wrote", out, out.stat().st_size, "bytes")
