#!/usr/bin/env python
"""Export the GPU system metrics W&B sampled during a run to <run_dir>/gpu_usage.json (compact, for the HTML report).

W&B samples only the node where rank 0 ran, so this covers that node's GPUs, not the whole job.
Tries the W&B API first, then falls back to reading the local .wandb file.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", "."))
a = p.parse_args()
run = Path(a.run_dir)
rid = (run / "wandb_run_id.txt").read_text().strip()
rows, source = [], None

try:
    import wandb
    api = wandb.Api()
    ent = os.environ.get("WANDB_ENTITY") or api.default_entity
    r = api.run(f"{ent}/{os.environ['WANDB_PROJECT']}/{rid}")
    for stream in ("system", "events"):
        try:
            rows = list(r.history(stream=stream, pandas=False, samples=3000))
        except Exception as e:  # noqa: BLE001
            print("api stream", stream, "failed:", repr(e)[:200])
            rows = []
        if rows:
            source = f"wandb api ({stream})"
            break
except Exception as e:  # noqa: BLE001
    print("api failed:", repr(e)[:300])

if not rows:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore
    files = sorted(glob.glob(f"{a.wandb_dir}/wandb/run-*-{rid}/run-{rid}.wandb"), key=os.path.getsize)
    ds = DataStore()
    ds.open_for_scan(files[-1])
    while True:
        try:
            d = ds.scan_data()
        except Exception as e:  # noqa: BLE001
            print("scan stopped:", repr(e)[:200])
            break
        if d is None:
            break
        rec = pb.Record()
        rec.ParseFromString(d)
        if rec.WhichOneof("record_type") == "stats":
            row = {"_timestamp": rec.stats.timestamp.seconds}
            for it in rec.stats.item:
                try:
                    row["system." + it.key] = json.loads(it.value_json)
                except Exception:  # noqa: BLE001
                    pass
            rows.append(row)
    source = "local .wandb file"

print("source:", source, "rows:", len(rows))
keys = sorted({k for r_ in rows for k in r_})
print("keys:", [k for k in keys if "gpu" not in k][:40])
print("gpu keys (gpu 0):", [k for k in keys if re.search(r"gpu\.0\.", k)])
gpus = sorted({int(m.group(1)) for k in keys if (m := re.search(r"gpu\.(\d+)\.", k))})
want = {"gpu": "util", "memoryAllocated": "mem_pct", "memoryAllocatedBytes": "mem_bytes", "powerWatts": "power_w",
        "temp": "temp_c", "memory": "mem_io_pct"}
out = {"source": source, "run_id": rid, "gpus": gpus, "t": [], "series": {}}
rows = [r_ for r_ in rows if r_.get("_timestamp") is not None]
rows.sort(key=lambda r_: r_["_timestamp"])
out["t"] = [round(float(r_["_timestamp"])) for r_ in rows]
for g in gpus:
    for k, name in want.items():
        vals = [r_.get(f"system.gpu.{g}.{k}") for r_ in rows]
        if any(v is not None for v in vals):
            out["series"][f"{g}.{name}"] = [None if v is None else round(float(v), 2) for v in vals]
for k in ("system.cpu", "system.memory_percent", "system.proc.memory.rssMB"):
    vals = [r_.get(k) for r_ in rows]
    if any(v is not None for v in vals):
        out["series"][k] = [None if v is None else round(float(v), 2) for v in vals]
(run / "gpu_usage.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", run / "gpu_usage.json", (run / "gpu_usage.json").stat().st_size, "bytes; samples", len(out["t"]), "gpus", gpus)
