#!/usr/bin/env python
"""Build rhea/specdec-report.html from pulled/specdec/{summary.json,records_compact.json,gpulog/*.csv}.

Run on Rhea: specdec-qwen38-rlvr-1123 (job 120765). CSS is reused from rlvr_moe_dense_template.html.
"""
import datetime as dt
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
src = root / "pulled" / "specdec"
S = json.loads((src / "summary.json").read_text(encoding="utf-8"))
recs = json.loads((src / "records_compact.json").read_text(encoding="utf-8"))
JOB, T0, T1 = 120765, dt.datetime(2026, 11, 23, 10, 30, 27), dt.datetime(2026, 11, 23, 11, 10, 30)

by = defaultdict(list)
for r in recs:
    by[r["config"]].append(r)
base = {r["id"]: r for r in by["baseline"]}
mean = lambda v: sum(v) / len(v)  # noqa: E731


def fit(rs):
    """sec ~ a * target passes + b * draft passes, least squares through the origin."""
    stt = sum(r["target_calls"] ** 2 for r in rs)
    sdd = sum(r["draft_calls"] ** 2 for r in rs)
    std = sum(r["target_calls"] * r["draft_calls"] for r in rs)
    sty = sum(r["target_calls"] * r["sec"] for r in rs)
    sdy = sum(r["draft_calls"] * r["sec"] for r in rs)
    if sdd == 0:
        return sty / stt, 0.0
    det = stt * sdd - std * std
    return (sty * sdd - sdy * std) / det, (sdy * stt - sty * std) / det


configs = []
for m in S["configs"]:
    rs = by[m["config"]]
    # Within one configuration draft passes and target passes are nearly proportional, so the two costs cannot be
    # separated there. Fit once per draft over all its round sizes instead (assumes the same two costs for each).
    if m["draft"]:
        a, b = fit([r for r in recs if r["draft"] == m["draft"]])
    else:
        a, b = m["sec_per_answer"] / mean([r["target_calls"] for r in rs]), 0.0
    c = {k: m[k] for k in ("config", "draft", "gamma", "answers", "tokens_per_s", "speedup", "sec_per_answer", "accuracy", "mean_tokens",
                           "truncated_rate", "tokens_per_target_pass", "acceptance", "draft_passes_per_token", "draft_params")}
    c.update(ms_target=a * 1000, ms_draft=b * 1000, target_calls=mean([r["target_calls"] for r in rs]),
             draft_calls=mean([r["draft_calls"] for r in rs]), rounds=mean([r["rounds"] for r in rs]),
             readvance=mean([r["readvance"] for r in rs]),
             qtype={qt: {k: v[k] for k in ("tokens_per_s", "acceptance", "accuracy", "mean_tokens")} for qt, v in m["by_qtype"].items()},
             ratio=[round((r["n_tokens"] / r["sec"]) / (base[r["id"]]["n_tokens"] / base[r["id"]]["sec"]), 3) for r in rs if r["id"] in base],
             correct=sum(r["correct"] for r in rs))
    bq = S["configs"][0]["by_qtype"]
    for qt in c["qtype"]:
        c["qtype"][qt]["speedup"] = c["qtype"][qt]["tokens_per_s"] / bq[qt]["tokens_per_s"]
    configs.append(c)

# GPU logs: per GPU per minute
nodes = sorted(Path(f).name.split("-")[0] for f in glob.glob(str(src / "gpulog" / f"*-{JOB}.csv")))
NB = int((T1 - T0).total_seconds() // 60) + 1
bins = {k: [[[] for _ in range(NB)] for _ in range(len(nodes) * 4)] for k in ("util", "mem", "pow")}
nread = 0
for ni, n in enumerate(nodes):
    for line in open(src / "gpulog" / f"{n}-{JOB}.csv", encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        try:
            t = (dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f") - T0).total_seconds()
            g, vals = ni * 4 + int(p[1]), (float(p[2]), float(p[3]) / 1000, float(p[5]))
        except (ValueError, IndexError):
            continue
        if 0 <= t < NB * 60:
            nread += 1
            for k, v in zip(("util", "mem", "pow"), vals):
                bins[k][g][int(t // 60)].append(v)
gpu = {k: [[None if not v else round(mean(v), 1) for v in row] for row in bins[k]] for k in bins}
flat = lambda k: [x for row in gpu[k] for x in row if x is not None]  # noqa: E731

D = {
    "run": S["run"], "job": JOB, "questions": S["questions"], "gpus": len({r["id"] for r in recs}) and len(nodes) * 4, "nodes": nodes, "elapsed_s": (T1 - T0).total_seconds(),
    "start": "10:30:27", "end": "11:10:30", "date": "23 Nov 2026", "max_new_tokens": S["args"]["max_new_tokens"], "decoding": S["decoding"],
    "target_params": S["models"]["target"]["params"], "drafts": {k: v["params"] for k, v in S["models"]["drafts"].items()},
    "configs": configs,
    "gpu": {**gpu, "readings": nread, "util_avg": round(mean(flat("util")), 1), "mem_peak": round(max(flat("mem")), 1),
            "mem_avg": round(mean(flat("mem")), 1), "pow_avg": round(mean(flat("pow")))},
    "arch": {"Qwen3.8-27B": 64, "Qwen3.5-0.8B": 24, "Qwen3.5-2B": 24, "Qwen3.5-4B": 32, "Qwen3.5-9B": 32},
    "hidden": {"Qwen3.8-27B": 5120, "Qwen3.5-0.8B": 1024, "Qwen3.5-2B": 2048, "Qwen3.5-4B": 2560, "Qwen3.5-9B": 4096},
    "disk": {"Qwen3.5-0.8B": 1.7, "Qwen3.5-2B": 4.5, "Qwen3.5-4B": 9.3, "Qwen3.5-9B": 19.3},
    "wandb": (src / "wandb_run.txt").read_text().split()[0],
}
css = re.search(r"<style>(.*?)</style>", (here / "rlvr_moe_dense_template.html").read_text(encoding="utf-8"), re.S).group(1)
# the architecture section lives in its own fragment file: CSS, HTML and JS parts
frag = (here / "specdec_arch_fragment.html").read_text(encoding="utf-8")
f_css, rest = frag.split("<!--HTML-->")
f_html, f_js = rest.split("<!--JS-->")
tpl = (here / "specdec_template.html").read_text(encoding="utf-8")
for old, new in (("</style>", f_css.replace("<!--CSS-->", "") + "</style>"),
                 ("<h2>How speculative decoding works", f_html + "<h2>How speculative decoding works"),
                 ("build();" + chr(10) + "</script>", f_js + "build();" + chr(10) + "</script>"),
                 ("draw(); drawSel();", "draw(); drawSel(); archSel();")):
    assert tpl.count(old) == 1, old
    tpl = tpl.replace(old, new)
html = tpl.replace("__STYLE__", css).replace("__DATA__", json.dumps(D, separators=(",", ":")))
out = root / "specdec-report.html"
out.write_text(html, encoding="utf-8", newline="\n")
for c in configs:
    print(f"{c['config']:18s} {c['tokens_per_s']:5.1f} tok/s x{c['speedup']:.2f}  target pass {c['ms_target']:5.1f} ms  draft pass {c['ms_draft']:5.1f} ms  "
          f"T {c['target_calls']:.0f} D {c['draft_calls']:.0f} readv {c['readvance']:.2f} correct {c['correct']}")
print("gpu", D["gpu"]["readings"], "readings; util", D["gpu"]["util_avg"], "mem peak", D["gpu"]["mem_peak"], "minutes", NB)
print("wrote", out, out.stat().st_size, "bytes")
