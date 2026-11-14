#!/usr/bin/env python
"""Build rhea/rlvr-moe-vs-dense-report.html: Qwen3.8-27B rerun (dense, job 47941) against Qwen3.6-35B-A3B (MoE, job 40421).

Reuses the data objects embedded in qwen38-rlvr-allgpu-report.html and qwen36-rlvr-report.html (run build_rlvr_full.py for
both first) and adds per-GPU per-minute memory from the nvidia-smi logs in pulled/<dir>/gpulog.
"""
import datetime as dt
import glob
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent


def data_of(name):
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("const D = "):
            return json.loads(line[len("const D = "):].rstrip().rstrip(";"))
    raise SystemExit(f"no data object in {name}")


def memheat(d, sub, t0):
    """32 x minutes: mean memory in use (thousands of MiB) per GPU per minute of the training loop."""
    job, nodes, loop_end = d["job"]["id"], d["job"]["nodes"], d["steps"][-1]["el"]
    nb = math.ceil(loop_end / 60)
    out = [[[] for _ in range(nb)] for _ in range(len(nodes) * 4)]
    for ni, n in enumerate(nodes):
        (f,) = glob.glob(str(root / "pulled" / sub / "gpulog" / f"{n}-{job}.csv"))
        for line in open(f, encoding="utf-8"):
            p = [x.strip() for x in line.split(",")]
            try:
                t = (dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f") - t0).total_seconds()
                g, mem = ni * 4 + int(p[1]), float(p[3]) / 1000
            except (ValueError, IndexError):
                continue
            if 0 <= t <= loop_end:
                out[g][min(nb - 1, int(t // 60))].append(mem)
    return [[None if not v else round(sum(v) / len(v)) for v in row] for row in out]


a, b = data_of("qwen38-rlvr-allgpu-report.html"), data_of("qwen36-rlvr-report.html")
ri = json.loads((root / "pulled" / "rlvr38b" / "runinfo.json").read_text())
a["gpu"]["memheat"] = memheat(a, "rlvr38b", dt.datetime.strptime(ri["t0"], "%Y-%m-%d %H:%M:%S"))
b["gpu"]["memheat"] = memheat(b, "rlvr36", dt.datetime(2026, 11, 13, 15, 1, 57))
for d in (a, b):
    assert len(d["gpu"]["heat"]) == 32 and len(d["gpu"]["memheat"]) == 32 and len(d["gpu"]["heat"][0]) == len(d["gpu"]["memheat"][0])
    d.pop("prev", None)
    for s in d["steps"]:
        s.pop("util", None)

# from each model's config.json on Lustre (text_config / vision_config); disk size from du -sh of the model directory
VISION = "27 layers, 1,152 wide"
arch = [
    dict(cls="Qwen3_5ForConditionalGeneration", size="27B parameters, all used for every token", disk_gb=52, layers=64, linear=48, full=16,
         hidden=5120, heads=24, kv_heads=4, lin_k=16, lin_v=48, mlp=17408, vocab=248320, ctx=262144, vision=VISION),
    dict(cls="Qwen3_5MoeForConditionalGeneration", size="35B parameters, about 3B active per token", disk_gb=67, layers=40, linear=30, full=10,
         hidden=2048, heads=16, kv_heads=2, lin_k=16, lin_v=32, experts=256, per_tok=8, expert_dim=512, shared_dim=512,
         vocab=248320, ctx=262144, vision=VISION),
]
# [run index, job id, what, nodes, duration, outcome, ok]
jobs = [
    [1, 40338, "Model download to Lustre (67 GB) and layer inspection", 1, "2 min", "Completed", True],
    [1, 40368, "Smoke test: 3 small steps on 1 node", 1, "13 min", "Passed, but 88% of answers hit the 1,024-token cap; cap raised to 4,096", True],
    [1, 40421, "RLVR training, 60 steps, GPU logging on every node", 8, "2 h 39 min", "Completed", True],
    [1, 40422, "Resume job (safety net)", 8, "39 s", "Nothing to do, training had finished", True],
    [1, 40423, "Benchmark of base and tuned model", 4, "29 min", "Failed: a GPU ran out of memory near the end of the base pass", False],
    [1, 47560, "Benchmark rerun with half the batch size, resumed from saved samples", 4, "21 min", "Completed", True],
    [1, 47561, "W&B before/after table and private Hugging Face upload", 1, "1 min", "Completed", True],
    [0, 47941, "RLVR training, 120 steps, GPU logging on every node", 8, "1 h 48 min", "Completed", True],
    [0, 47942, "Resume job (safety net)", 8, "38 s", "Nothing to do, training had finished", True],
    [0, 47943, "Benchmark of the tuned model (base samples reused from the first Qwen3.8 run)", 4, "4 min", "Completed", True],
    [0, 47944, "W&B before/after table (no upload)", 1, "1 min", "Completed", True],
    [0, 48773, "Private Hugging Face upload to a new repo", 1, "1 min", "Completed", True],
]
html = (here / "rlvr_moe_dense_template.html").read_text(encoding="utf-8").replace(
    "__DATA__", json.dumps({"a": a, "b": b, "jobs": jobs, "arch": arch}, separators=(",", ":")))
out = root / "rlvr-moe-vs-dense-report.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("wrote", out, out.stat().st_size, "bytes")
