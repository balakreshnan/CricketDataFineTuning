#!/usr/bin/env python
"""Build rhea/rlvr-two-runs-report.html: both RLVR runs in one page.

Reuses the data objects already embedded in cricket-rlvr-report.html (27B) and qwen36-rlvr-report.html (Qwen3.6),
so run build_rlvr.py and build_rlvr36.py first.
"""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent


def data_of(name):
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("const D = "):
            return json.loads(line[len("const D = "):].rstrip().rstrip(";"))
    raise SystemExit(f"no data object in {name}")


a, b = data_of("cricket-rlvr-report.html"), data_of("qwen36-rlvr-report.html")
a["args"].setdefault("train_batch", 8)
# the 27B report stored memory in GiB; the Qwen3.6 one in thousands of MiB (a full GPU = 286.5). Use the latter for both.
g = a["gpu"]
g["mem_gb"] = [[None if x is None else round(x * 1.024, 1) for x in row] for row in g["mem_gb"]]
g["mem_peak_gb"] = [round(x * 1.024, 1) for x in g["mem_peak_gb"]]
g["mem_avg_gb"] = [round(x * 1.024, 1) for x in g["mem_avg_gb"]]
b["gpu"].pop("heat", None)                       # per-GPU heatmap lives in the single-run report
for d in (a, b):
    for s in d["steps"]:
        s.pop("util", None)
# [run index, job id, what, nodes, duration, outcome, ok]
jobs = [
    [0, 38952, "Smoke test: 3 small steps on 1 node", 1, "2 min", "Failed: W&B run-ID helper missing in this W&B version; fixed", False],
    [0, 38966, "Smoke test, second try", 1, "10 min", "Passed, but logged to the wrong W&B project; fixed", True],
    [0, 39000, "RLVR training, 100 steps", 8, "1 h 22 min", "Completed", True],
    [0, 39001, "Resume job (safety net)", 8, "40 s", "Nothing to do, training had finished", True],
    [0, 39066, "Benchmark of the tuned model (base samples reused from the earlier LoRA benchmark)", 4, "4 min", "Completed", True],
    [0, 39067, "W&B before/after table and private Hugging Face upload", 1, "1 min", "Completed", True],
    [0, 39287, "Export of W&B GPU readings for the report", 1, "1 min", "Completed", True],
    [1, 40338, "Model download to Lustre (67 GB) and layer inspection", 1, "2 min", "Completed", True],
    [1, 40368, "Smoke test: 3 small steps on 1 node", 1, "13 min", "Passed, but 88% of answers hit the 1,024-token cap; cap raised to 4,096", True],
    [1, 40421, "RLVR training, 60 steps", 8, "2 h 39 min", "Completed", True],
    [1, 40422, "Resume job (safety net)", 8, "39 s", "Nothing to do, training had finished", True],
    [1, 40423, "Benchmark of base and tuned model", 4, "29 min", "Failed: a GPU ran out of memory near the end of the base pass", False],
    [1, 47560, "Benchmark rerun with half the batch size, resumed from saved samples", 4, "21 min", "Completed", True],
    [1, 47561, "W&B before/after table and private Hugging Face upload", 1, "1 min", "Completed", True],
]
html = (here / "rlvr_both_template.html").read_text(encoding="utf-8").replace(
    "__DATA__", json.dumps({"a": a, "b": b, "jobs": jobs}, separators=(",", ":")))
out = root / "rlvr-two-runs-report.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("wrote", out, out.stat().st_size, "bytes")
