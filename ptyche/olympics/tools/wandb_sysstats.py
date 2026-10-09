"""Per-GPU system metrics W&B recorded for a run (the node that hosted the W&B logger).
   python wandb_sysstats.py balabala76/nemotron35-olympics-reasoning/s6mmt4cz"""
import re
import statistics as st
import sys

import wandb

run = wandb.Api().run(sys.argv[1])
print("run:", run.name, "| state:", run.state, "| host:", run.metadata.get("host"), "| gpus:", run.metadata.get("gpu_count"),
      run.metadata.get("gpu"))
rows = run.history(stream="system", samples=100000, pandas=False)
print("system samples:", len(rows))
keys = sorted({k for r in rows for k in r if k.startswith("system.gpu.")})
by = {}
for k in keys:
    m = re.match(r"system\.gpu\.(\d+)\.(\w+)$", k)
    if m:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if len(vals) > 10:
            by.setdefault(m.group(2), {})[int(m.group(1))] = vals
for metric in ("gpu", "smActive", "smOccupancy", "pipeTensorActive", "dramActive", "memoryAllocatedBytes", "powerWatts",
               "enforcedPowerLimitWatts", "smClock", "temp", "nvlinkTxBytes", "nvlinkRxBytes"):
    if metric in by:
        for g, v in sorted(by[metric].items()):
            # steady state: drop the first/last 10 % (startup, final save/teardown)
            s = sorted(v)[len(v) // 10: -len(v) // 10 or None]
            print(f"{metric:22s} GPU{g}: median {st.median(v):10.1f}  mean {st.mean(v):10.1f}  p10-p90 {s[0]:.1f}-{s[-1]:.1f}")
print("other gpu metric names:", sorted(by.keys()))
