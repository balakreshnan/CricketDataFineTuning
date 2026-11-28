#!/usr/bin/env python3
"""Build rl_length_sweep_report.html from data/sweep_raw.txt.

Usage:  python build_report.py
Reads   paper-report/data/sweep_raw.txt   (sections separated by lines starting with ###)
Writes  paper-report/rl_length_sweep_report.html  (single self-contained file, no external resources)

All narrative text and the hard-coded facts (earlier runs, published work, set-up) live in this
file; every number that describes the sweep is computed from the data file.
"""
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "sweep_raw.txt"
EXTRA = HERE / "data" / "sweep_extra.txt"
GPU_MEM_MIB = 286524
OUT = HERE / "rl_length_sweep_report.html"

COEFS = ["0", "0.1", "0.4", "1.0"]
SEEDS = [1, 2, 3]
TYPES = ["overall", "run_rate", "chase_rate", "milestone"]
GPUS_PER_NODE = 4

# ----------------------------------------------------------------------------------------------
# Hard-coded facts that are not in the data file
# ----------------------------------------------------------------------------------------------
# Earlier runs on the same task, from the project's run tracker. Scored with eval.py, not NeMo Gym.
EARLIER_RUNS = [
    {"method": "LoRA on distilled reasoning traces", "model": "Qwen3.8-27B", "steps": None,
     "gpus": 32, "hours": 4.5, "gpuh": 144, "acc0": 97.00, "acc1": 97.92, "tok0": 734, "tok1": 455,
     "scorer": "eval.py"},
    {"method": "RLVR, 100 steps", "model": "Qwen3.8-27B", "steps": 100,
     "gpus": 32, "hours": 1.33, "gpuh": 43, "acc0": 97.00, "acc1": 99.17, "tok0": 734, "tok1": 109,
     "scorer": "eval.py"},
    {"method": "RLVR, 120 steps", "model": "Qwen3.8-27B", "steps": 120,
     "gpus": 32, "hours": 1.76, "gpuh": 56, "acc0": 97.00, "acc1": 99.96, "tok0": 734, "tok1": 119,
     "scorer": "eval.py"},
    {"method": "RLVR, 60 steps", "model": "Qwen3.6-35B-A3B", "steps": 60,
     "gpus": 32, "hours": 2.65, "gpuh": 85, "acc0": 93.62, "acc1": 98.96, "tok0": 2177, "tok1": 151,
     "scorer": "eval.py"},
    # GPU-hours from the tracker; accuracy and tokens for this row are filled from the data file
    # (NeMo Gym harness) in build_data().
    {"method": "NeMo RL (GRPO), 100 steps, c = 0.4", "model": "Nemotron-3.5-Lightning-30B-A3B", "steps": 100,
     "gpus": 16, "hours": 1.62, "gpuh": 26, "hours_approx": True, "scorer": "NeMo Gym"},
]

SPECDEC = {"date": "2026-11-23", "alone_tps": 36, "draft_lo": 17, "draft_hi": 25, "best_ratio": 0.70}


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------
def read_sections(path):
    secs, name = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("###"):
            name = line[3:].strip()
            secs[name] = []
        elif name is not None:
            secs[name].append(line)
    return secs


def read_extra(path):
    """Sections '###NAME col,col,...' followed by CSV rows -> {NAME: [dict, ...]}."""
    secs, name, cols = {}, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("###"):
            name, hdr = line[3:].split(None, 1)
            cols = [c.strip() for c in hdr.split(",")]
            secs[name] = []
        elif name and line.strip():
            vals = line.split(",")
            if len(vals) == len(cols):
                secs[name].append(dict(zip(cols, vals)))
    return secs


def fnum(x):
    return float(x) if x not in ("", None) else None


RANGES = {"all": (2, 60), "early": (2, 10), "late": (51, 60)}
# index of each field in the per-step rows embedded in the page (run["st"])
ST = ["step", "job", "loss", "kl", "gen", "refit", "transfer", "lpprep", "logp", "train", "tprep", "ckpt",
      "other", "inflight", "active", "idle", "total", "warm"]


def build_extra(sweep, jobs):
    """Per-step phase timings and per-minute GPU telemetry (data/sweep_extra.txt)."""
    X = read_extra(EXTRA)
    steps = collections.defaultdict(dict)
    for d in X["STEPS"]:
        run, step, job = d["run"], int(d["step"]), int(d["job"])
        if step not in steps[run] or steps[run][step]["job"] <= job:
            steps[run][step] = dict(d, job=job)
    gpu = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in X["GPU"]:
        gpu[d["run"]][int(d["job"])].append(d)

    step_jobs = collections.defaultdict(lambda: collections.defaultdict(list))
    job_first_step = {}
    for d in X["STEPS"]:
        step_jobs[d["run"]][int(d["job"])].append(float(d["total_s"]))
        job_first_step[int(d["job"])] = min(job_first_step.get(int(d["job"]), 10 ** 9), int(d["step"]))

    # Estimated minute at which the first logged step of a job starts. For completed jobs the step
    # times are laid end to end so that the last step ends one minute before the last GPU sample.
    # Timed-out jobs use the median offset of the completed jobs.
    offs = {}
    for run, js in gpu.items():
        for job, rows in js.items():
            if job in step_jobs[run] and jobs[job]["state"] == "COMPLETED":
                offs[job] = max(0.0, len(rows) - 1 - sum(step_jobs[run][job]) / 60)
    med_off = sorted(offs.values())[len(offs) // 2]

    for r in sweep:
        run = r["run"]
        st = []
        first_of_job = {}
        for s in sorted(steps[run]):
            first_of_job.setdefault(steps[run][s]["job"], s)
        for s in sorted(steps[run]):
            d = steps[run][s]
            v = {k: fnum(d[k]) for k in d if k not in ("run", "job", "step")}
            parts = [v["generation"], v["prepare_for_generation_total"], v["logprob_inference_prep"],
                     v["policy_and_reference_logprobs"], v["policy_training"], v["training_prep"],
                     v["checkpointing"] or 0.0]
            other = max(0.0, v["total_s"] - sum(parts))
            st.append([s, d["job"], round(v["loss"], 4), round(v["kl_err"], 4), v["generation"],
                       v["prepare_for_generation_total"], v["prepare_for_generation_transfer_and_update_weights"],
                       v["logprob_inference_prep"], v["policy_and_reference_logprobs"], v["policy_training"],
                       v["training_prep"], v["checkpointing"], round(other, 2), v["max_inflight"],
                       v["mean_active_s"], v["mean_idle_s"], v["total_s"],
                       1 if first_of_job[d["job"]] == s else 0])
        r["st"] = st
        segs = []
        for job in sorted(gpu[run]):
            rows = gpu[run][job]
            logged = step_jobs[run].get(job, [])
            off = None
            if logged:
                off = offs.get(job, min(med_off, max(0.0, len(rows) - sum(logged) / 60)))
            segs.append({
                "job": job, "t0": rows[0]["minute"], "state": jobs[job]["state"],
                "off": None if off is None else round(off, 1),
                # steps logged by this job, including steps later repeated by a resumed job
                "s0": job_first_step.get(job), "secs": [round(x, 1) for x in logged],
                "rows": [[float(d["avg_util"]), int(float(d["avg_mem_mib"])), int(float(d["peak_mem_mib"])),
                          float(d["avg_power_w"]), float(d["avg_temp_c"]), float(d["max_temp_c"])] for d in rows],
            })
        r["gpu"] = segs
        allrows = [row for sg in segs for row in sg["rows"]]
        loaded = [row for row in allrows if row[1] >= 10000]
        r["gs"] = {
            "minutes": len(allrows),
            "util": mean(row[0] for row in allrows), "util_loaded": mean(row[0] for row in loaded),
            "mem_loaded": mean(row[1] for row in loaded), "peak": max(row[2] for row in allrows),
            "power": mean(row[3] for row in allrows), "power_loaded": mean(row[3] for row in loaded),
            "kwh": sum(row[3] for row in allrows) * 16 / 60 / 1000,
            "tmax": max(row[5] for row in allrows), "tavg": mean(row[4] for row in allrows),
        }
        # training jobs of this run: union of the jobs seen in the training log and the GPU log
        r["jobs"] = sorted(set(r["jobs"]) | set(gpu[run]))
        r["resumed"] = len(step_jobs[run]) > 1
        r["train_gpuh"] = sum(jobs[j]["hours"] * jobs[j]["nodes"] * GPUS_PER_NODE for j in r["jobs"])

    I = {k: i for i, k in enumerate(ST)}
    phase, gcoef, eng, health = {}, {}, {}, {}
    for c in COEFS:
        rs = [r for r in sweep if r["c"] == c]
        phase[c], eng[c] = {}, {}
        for name, (a, b) in RANGES.items():
            rows = [x for r in rs for x in r["st"] if a <= x[0] <= b]

            def m(k):
                return mean((x[I[k]] or 0.0) for x in rows)

            phase[c][name] = {"gen": m("gen"), "refit": m("refit"), "transfer": m("transfer"),
                              "logp": m("lpprep") + m("logp"), "train": m("train") + m("tprep"),
                              "misc": m("ckpt") + m("other"), "total": m("total")}
            eng[c][name] = {"active": m("active"), "idle": m("idle"), "inflight": m("inflight"),
                            "busy": m("active") / (m("active") + m("idle")) * 100}
        ck = [x[I["ckpt"]] for r in rs for x in r["st"] if x[I["ckpt"]] is not None]
        phase[c]["ckpt_when_logged"] = mean(ck)
        g = [r["gs"] for r in rs]
        gcoef[c] = {k: mean(x[k] for x in g) for k in g[0]}
        gcoef[c]["peak"] = max(x["peak"] for x in g)
        gcoef[c]["tmax"] = max(x["tmax"] for x in g)
        gcoef[c]["kwh_min"] = min(x["kwh"] for x in g)
        gcoef[c]["kwh_max"] = max(x["kwh"] for x in g)
        loss = sorted(x[I["loss"]] for r in rs for x in r["st"])
        kl = sorted(x[I["kl"]] for r in rs for x in r["st"])
        health[c] = {"loss_med": loss[len(loss) // 2], "loss_min": loss[0], "loss_max": loss[-1],
                     "kl_med": kl[len(kl) // 2], "kl_min": kl[0], "kl_max": kl[-1]}
    empty = [row[3] for r in sweep for sg in r["gpu"] for row in sg["rows"] if row[1] == 0 and row[0] == 0]
    warm = [x for r in sweep for x in r["st"] if x[I["warm"]]]
    nxt = [r["st"][x[0]] for r in sweep for x in r["st"] if x[I["warm"]] and x[0] < len(r["st"])]
    # run whose generation KL error is highest over the last ten steps
    def klm(r, a, b):
        return mean(x[I["kl"]] for x in r["st"] if a <= x[0] <= b)
    top = max(sweep, key=lambda r: klm(r, 51, 60))
    others = [klm(r, 51, 60) for r in sweep if r is not top]
    klrise = {"run": top["run"], "first": klm(top, 1, 10), "last": klm(top, 51, 60),
              "others_lo": min(others), "others_hi": max(others)}
    return {"klrise": klrise, "phase": phase, "gcoef": gcoef, "eng": eng, "health": health, "ST": ST, "memMiB": GPU_MEM_MIB, "emptyPower": mean(empty),
            "warm": {"first": mean(x[I["total"]] for x in warm), "next": mean(x[I["total"]] for x in nxt),
                     "n": len(warm)}}


def hours(hms):
    days = 0
    if "-" in hms:
        d, hms = hms.split("-", 1)
        days = int(d)
    parts = [int(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 24 + h + m / 60 + s / 3600


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def stat(xs):
    xs = list(xs)
    return {"mean": mean(xs), "min": min(xs), "max": max(xs)}


def summ(s):
    """Reduce an evaluation summary to what the page needs (percentages, not fractions)."""
    out = {}
    for t in TYPES:
        o = s[t]
        out[t] = {"acc": o["accuracy"] * 100, "tok": o["mean_tokens"], "trunc": o["truncated_rate"] * 100,
                  "pass4": o["pass@4"] * 100, "noans": o["no_answer_rate"] * 100}
    return out


def energy(e):
    if not e:
        return None
    return {"aps": e["answers_per_s"], "power": e["avg_power_w"], "mwh": e["wh_per_correct"] * 1000,
            "seconds": e["seconds"], "samples": e["power_samples"], "tps": e["tokens_per_s"],
            "wh": e["energy_wh"]}


def build_data():
    secs = read_sections(RAW)
    ev = json.loads(next(l for l in secs["EVAL"] if l.strip()))

    # --- training logs: per run and step keep the row from the later log file ------------------
    rows = collections.defaultdict(dict)
    run_jobs = collections.defaultdict(set)
    for line in secs["TRAIN"]:
        p = line.strip().split(",")
        if len(p) != 6:
            continue
        run, log, step = p[0], p[1], int(p[2])
        job = int(log.split("-")[1])
        run_jobs[run].add(job)
        if step not in rows[run] or rows[run][step][0] <= job:
            rows[run][step] = (job, float(p[3]), float(p[4]), float(p[5]))

    # --- Slurm jobs -----------------------------------------------------------------------------
    jobs = {}
    train_node_h = eval_node_h = 0.0
    n_train = n_eval = n_timeout = 0
    for line in secs["JOBS"]:
        p = line.strip().split("|")
        if len(p) != 5:
            continue
        jid, name, state, elapsed, nn = int(p[0]), p[1], p[2], p[3], int(p[4])
        h = hours(elapsed)
        jobs[jid] = {"name": name, "state": state, "hours": h, "nodes": nn}
        if name.endswith(".train"):
            train_node_h += h * nn
            n_train += 1
            n_timeout += state == "TIMEOUT"
        else:
            eval_node_h += h * nn
            n_eval += 1

    # --- sweep runs -----------------------------------------------------------------------------
    sweep = []
    for c in COEFS:
        for s in SEEDS:
            run = f"sweep-c{c}-s{s}"
            e = ev[f"{run}/gym"]
            steps = sorted(rows[run])
            tj = sorted(run_jobs[run])
            sweep.append({
                "run": run, "c": c, "seed": s,
                "sum": summ(e["tuned"]), "en": energy(e["energy"]["tuned"]),
                "idle": e["energy"]["idle_power_w"],
                "train": [[st, rows[run][st][1], round(rows[run][st][2], 1), rows[run][st][3]] for st in steps],
                "jobs": tj, "resumed": len(tj) > 1,
                "train_gpuh": sum(jobs[j]["hours"] * jobs[j]["nodes"] * GPUS_PER_NODE for j in tj),
            })

    extra = build_extra(sweep, jobs)

    coef = {}
    for c in COEFS:
        rs = [r for r in sweep if r["c"] == c]
        coef[c] = {
            "acc": stat(r["sum"]["overall"]["acc"] for r in rs),
            "tok": stat(r["sum"]["overall"]["tok"] for r in rs),
            "trunc": stat(r["sum"]["overall"]["trunc"] for r in rs),
            "aps": stat(r["en"]["aps"] for r in rs),
            "power": stat(r["en"]["power"] for r in rs),
            "mwh": stat(r["en"]["mwh"] for r in rs),
            "train_gpuh": stat(r["train_gpuh"] for r in rs),
            "step_first10": mean(row[3] for r in rs for row in r["train"] if row[0] <= 10),
            "step_last10": mean(row[3] for r in rs for row in r["train"] if row[0] > 50),
        }
        for t in TYPES[1:]:
            coef[c][t] = stat(r["sum"][t]["acc"] for r in rs)

    # --- base, original run, other models -------------------------------------------------------
    o1, o2 = ev["nemorl-cricket-1124/gym"], ev["nemorl-cricket-1124/gym-energy"]
    base_evals = [
        {"label": "Base Nemotron, first evaluation", "sum": summ(o1["base"]), "en": None},
        {"label": "Base Nemotron, second evaluation", "sum": summ(o2["base"]), "en": energy(o2["energy"]["base"])},
    ]
    orig_evals = [
        {"label": "100-step run (c = 0.4, seed 42), first evaluation", "sum": summ(o1["tuned"]), "en": None},
        {"label": "100-step run (c = 0.4, seed 42), re-evaluation", "sum": summ(o2["tuned"]),
         "en": energy(o2["energy"]["tuned"])},
    ]
    base = {
        "acc": mean(b["sum"]["overall"]["acc"] for b in base_evals),
        "tok": mean(b["sum"]["overall"]["tok"] for b in base_evals),
        "trunc": mean(b["sum"]["overall"]["trunc"] for b in base_evals),
        "types": {t: mean(b["sum"][t]["acc"] for b in base_evals) for t in TYPES},
        "en": base_evals[1]["en"],
    }

    def arch(name, kind, method, key, en_key=None):
        e, ee = ev[key], ev[en_key or key]
        return {"name": name, "kind": kind, "method": method,
                "base": summ(e["base"]), "tuned": summ(e["tuned"]),
                "base_en": energy(ee["energy"]["base"]), "tuned_en": energy(ee["energy"]["tuned"])}

    archs = [
        arch("Nemotron-3.5-Lightning-30B-A3B", "hybrid Mamba-Transformer, mixture of experts",
             "NeMo RL (GRPO), 100 steps, c = 0.4", "nemorl-cricket-1124/gym", "nemorl-cricket-1124/gym-energy"),
        arch("Qwen3.8-27B", "dense Transformer", "RLVR, project script", "qwen38-rlvr-1114/gym"),
        arch("Qwen3.6-35B-A3B", "Transformer, mixture of experts", "RLVR, project script", "qwen36-rlvr-1113/gym"),
    ]

    earlier = [dict(r) for r in EARLIER_RUNS]
    earlier[-1].update(acc0=base_evals[0]["sum"]["overall"]["acc"], acc1=orig_evals[0]["sum"]["overall"]["acc"],
                       tok0=base_evals[0]["sum"]["overall"]["tok"], tok1=orig_evals[0]["sum"]["overall"]["tok"])

    totals = {"train_node_h": train_node_h, "train_gpu_h": train_node_h * GPUS_PER_NODE,
              "eval_node_h": eval_node_h, "eval_gpu_h": eval_node_h,  # 1 node, 1 GPU serves
              "n_train_jobs": n_train, "n_eval_jobs": n_eval, "n_timeout": n_timeout,
              "n_resumed": sum(r["resumed"] for r in sweep)}
    idle = [e["energy"]["idle_power_w"] for e in ev.values() if "energy" in e]

    return {"coefs": COEFS, "sweep": sweep, "coef": coef, "base": base, "baseEvals": base_evals,
            "origEvals": orig_evals, "archs": archs, "earlier": earlier, "totals": totals,
            "idle": {"min": min(idle), "max": max(idle)}, **extra}


# ----------------------------------------------------------------------------------------------
# Text tokens (@@name@@ in the template)
# ----------------------------------------------------------------------------------------------
def n(x, d=0):
    return f"{x:,.{d}f}"


def tokens_for(D):
    c, b, t = D["coef"], D["base"], D["totals"]
    k = {}
    for cc, tag in zip(COEFS, ["c0", "c01", "c04", "c10"]):
        s = c[cc]
        k[f"{tag}_acc"] = n(s["acc"]["mean"], 2)
        k[f"{tag}_acc_lo"] = n(s["acc"]["min"], 2)
        k[f"{tag}_acc_hi"] = n(s["acc"]["max"], 2)
        k[f"{tag}_acc_spread"] = n(s["acc"]["max"] - s["acc"]["min"], 2)
        k[f"{tag}_tok"] = n(s["tok"]["mean"])
        k[f"{tag}_tok_lo"] = n(s["tok"]["min"])
        k[f"{tag}_tok_hi"] = n(s["tok"]["max"])
        k[f"{tag}_mwh"] = n(s["mwh"]["mean"], 1)
        k[f"{tag}_aps"] = n(s["aps"]["mean"], 1)
        k[f"{tag}_power"] = n(s["power"]["mean"])
        k[f"{tag}_gpuh"] = n(s["train_gpuh"]["mean"], 1)
        k[f"{tag}_step_first"] = n(s["step_first10"])
        k[f"{tag}_step_last"] = n(s["step_last10"])
        k[f"{tag}_trunc"] = n(s["trunc"]["mean"], 2)
    ps = [r["en"]["samples"] for r in D["sweep"]]
    k["ps_lo"], k["ps_hi"] = str(min(ps)), str(max(ps))
    k["base_acc"] = n(b["acc"], 1)
    k["base_acc2"] = n(b["acc"], 2)
    k["base_tok"] = n(round(b["tok"] / 5) * 5)
    k["base_trunc"] = n(b["trunc"], 1)
    k["base_mwh"] = n(b["en"]["mwh"], 1)
    k["base_aps"] = n(b["en"]["aps"], 1)
    k["base_power"] = n(b["en"]["power"])
    k["b1_acc"], k["b2_acc"] = (n(e["sum"]["overall"]["acc"], 2) for e in D["baseEvals"])
    k["b1_tok"], k["b2_tok"] = (n(e["sum"]["overall"]["tok"]) for e in D["baseEvals"])
    k["b1_trunc"], k["b2_trunc"] = (n(e["sum"]["overall"]["trunc"], 2) for e in D["baseEvals"])
    k["o1_acc"], k["o2_acc"] = (n(e["sum"]["overall"]["acc"], 2) for e in D["origEvals"])
    k["o1_tok"], k["o2_tok"] = (n(e["sum"]["overall"]["tok"]) for e in D["origEvals"])
    k["c0_shorter"] = n((1 - c["0"]["tok"]["mean"] / b["tok"]) * 100)
    k["train_node_h"] = n(t["train_node_h"], 1)
    k["train_gpu_h"] = n(t["train_gpu_h"])
    k["eval_gpu_h"] = n(t["eval_gpu_h"], 1)
    k["n_train_jobs"] = str(t["n_train_jobs"])
    k["n_eval_jobs"] = str(t["n_eval_jobs"])
    k["n_timeout"] = str(t["n_timeout"])
    k["n_resumed"] = str(t["n_resumed"])
    k["gpuh_per_run"] = n(t["train_gpu_h"] / len(D["sweep"]), 1)
    s2 = next(r for r in D["sweep"] if r["run"] == "sweep-c0.4-s2")["sum"]
    k["s2_mile"] = n(s2["milestone"]["acc"], 2)
    k["s2_rr"] = n(s2["run_rate"]["acc"], 2)
    k["s2_cr"] = n(s2["chase_rate"]["acc"], 2)
    k["s2_all"] = n(s2["overall"]["acc"], 2)
    q38, q36 = D["archs"][1], D["archs"][2]
    for tag, a in (("q38", q38), ("q36", q36)):
        k[f"{tag}_acc0"] = n(a["base"]["overall"]["acc"], 2)
        k[f"{tag}_acc1"] = n(a["tuned"]["overall"]["acc"], 2)
        k[f"{tag}_tok0"] = n(a["base"]["overall"]["tok"])
        k[f"{tag}_tok1"] = n(a["tuned"]["overall"]["tok"])
    k["q36_mile0"] = n(q36["base"]["milestone"]["acc"], 2)
    k["q36_mile1"] = n(q36["tuned"]["milestone"]["acc"], 2)
    cap = D["memMiB"]
    k["cap_gib"] = n(cap / 1024, 1)
    for cc, tag in zip(COEFS, ["c0", "c01", "c04", "c10"]):
        for rng in RANGES:
            ph, en = D["phase"][cc][rng], D["eng"][cc][rng]
            k[f"ph_{tag}_{rng}_gen"] = n(ph["gen"], 1)
            k[f"ph_{tag}_{rng}_refit"] = n(ph["refit"], 1)
            k[f"ph_{tag}_{rng}_train"] = n(ph["train"], 1)
            k[f"ph_{tag}_{rng}_logp"] = n(ph["logp"], 1)
            k[f"ph_{tag}_{rng}_total"] = n(ph["total"], 1)
            k[f"ph_{tag}_{rng}_genp"] = n(ph["gen"] / ph["total"] * 100)
            k[f"ph_{tag}_{rng}_refitp"] = n(ph["refit"] / ph["total"] * 100)
            k[f"ph_{tag}_{rng}_trainp"] = n(ph["train"] / ph["total"] * 100)
            k[f"eng_{tag}_{rng}_busy"] = n(en["busy"])
            k[f"eng_{tag}_{rng}_active"] = n(en["active"], 1)
            k[f"eng_{tag}_{rng}_idle"] = n(en["idle"], 1)
            k[f"eng_{tag}_{rng}_inflight"] = n(en["inflight"])
        g = D["gcoef"][cc]
        k[f"g_{tag}_util"] = n(g["util"], 1)
        k[f"g_{tag}_util_loaded"] = n(g["util_loaded"], 1)
        k[f"g_{tag}_mem_gib"] = n(g["mem_loaded"] / 1024, 1)
        k[f"g_{tag}_mem_pct"] = n(g["mem_loaded"] / cap * 100, 1)
        k[f"g_{tag}_peak_gib"] = n(g["peak"] / 1024, 1)
        k[f"g_{tag}_peak_pct"] = n(g["peak"] / cap * 100, 1)
        k[f"g_{tag}_power"] = n(g["power"])
        k[f"g_{tag}_kwh"] = n(g["kwh"], 1)
        k[f"g_{tag}_kwh_lo"] = n(g["kwh_min"], 1)
        k[f"g_{tag}_kwh_hi"] = n(g["kwh_max"], 1)
        k[f"g_{tag}_tmax"] = n(g["tmax"])
        k[f"g_{tag}_minutes"] = n(g["minutes"])
        h = D["health"][cc]
        k[f"loss_{tag}_med"] = f"{h['loss_med']:.3f}" if abs(h["loss_med"]) < 0.1 else n(h["loss_med"], 2)
        k[f"loss_{tag}_max"] = n(h["loss_max"], 2)
    allph = [D["phase"][cc][rng] for cc in COEFS for rng in RANGES]
    infl = [D["eng"][cc]["all"]["inflight"] for cc in COEFS]
    k["infl_lo"], k["infl_hi"] = n(min(infl)), n(max(infl))
    k["refit_lo"] = n(min(x["refit"] for x in allph), 1)
    k["refit_hi"] = n(max(x["refit"] for x in allph), 1)
    k["transfer"] = n(mean(x["transfer"] for x in allph), 1)
    early = [D["phase"][cc]["early"]["gen"] for cc in COEFS]
    k["gen_early_lo"], k["gen_early_hi"] = n(min(early)), n(max(early))
    k["kl_min"] = f"{min(D['health'][cc]['kl_min'] for cc in COEFS):.4f}"
    k["kl_max"] = f"{max(D['health'][cc]['kl_max'] for cc in COEFS):.4f}"
    k["kl_med_lo"] = f"{min(D['health'][cc]['kl_med'] for cc in COEFS):.4f}"
    k["kl_med_hi"] = f"{max(D['health'][cc]['kl_med'] for cc in COEFS):.4f}"
    kr = D["klrise"]
    k["klr_run"] = kr["run"]
    k["klr_first"], k["klr_last"] = f"{kr['first']:.4f}", f"{kr['last']:.4f}"
    k["klr_olo"], k["klr_ohi"] = f"{kr['others_lo']:.4f}", f"{kr['others_hi']:.4f}"
    k["warm_first"], k["warm_next"], k["warm_n"] = n(D["warm"]["first"]), n(D["warm"]["next"]), str(D["warm"]["n"])
    k["empty_power"] = n(D["emptyPower"])
    k["ckpt_s"] = n(mean(D["phase"][cc]["ckpt_when_logged"] for cc in COEFS), 1)
    k["sd_alone"] = str(SPECDEC["alone_tps"])
    k["sd_lo"], k["sd_hi"] = str(SPECDEC["draft_lo"]), str(SPECDEC["draft_hi"])
    k["sd_ratio"] = f"{SPECDEC['best_ratio']:.2f}"
    return k


# ----------------------------------------------------------------------------------------------
# Template
# ----------------------------------------------------------------------------------------------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL Length Sweep</title>
<style>
:root{
  --bg:#f5f6f8; --surface:#ffffff; --surface2:#f0f2f5; --ink:#1a1f2b; --ink2:#454e60; --muted:#667085;
  --grid:#e4e7ec; --axis:#98a1b0; --border:#d9dde4;
  --base:#2a78d6; --c0:#1baf7a; --c01:#8a5cd6; --c04:#eb6834; --c10:#d1418f; --tuned:#33415c;
  --warn-bg:#fff6e5; --warn-border:#e0b252;
  --p1:#2a78d6; --p2:#d99a00; --p3:#0f9aa8; --p4:#7b5bd0; --p5:#8a93a3;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#11141a; --surface:#1a1f28; --surface2:#232a35; --ink:#e9ecf2; --ink2:#bcc3cf; --muted:#8f98a9;
    --grid:#2b323f; --axis:#5f6879; --border:#333b49;
    --base:#5a9cf0; --c0:#35c994; --c01:#a988ee; --c04:#f58456; --c10:#e76cb0; --tuned:#c9d3e6;
    --warn-bg:#2b2413; --warn-border:#8a6d2a;
    --p1:#5a9cf0; --p2:#e6b530; --p3:#35b9c6; --p4:#a58cf0; --p5:#9aa3b3;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:16px;line-height:1.55;overflow-wrap:break-word}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.75rem;line-height:1.2;margin:0 0 8px}
h2{font-size:1.25rem;line-height:1.3;margin:0 0 4px}
h3{font-size:1rem;margin:16px 0 4px}
p{margin:8px 0}
.lede{color:var(--ink2);max-width:75ch}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 12px;max-width:85ch}
.note{color:var(--muted);font-size:.85rem;max-width:90ch}
code{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;font-size:.88em;
  background:var(--surface2);padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}
section{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;margin:20px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;
  border-top:4px solid var(--axis);min-width:0}
.tile .k{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:1.7rem;font-weight:650;line-height:1.2;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .d{font-size:.85rem;color:var(--ink2)}
.controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin:8px 0 10px}
.controls label{font-size:.9rem;color:var(--ink2);display:inline-flex;align-items:center;gap:6px}
select,button{font:inherit;color:var(--ink)}
select{background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:5px 8px;max-width:100%}
input[type=checkbox]{width:16px;height:16px;accent-color:var(--base);margin:0}
.seg{display:inline-flex;flex-wrap:wrap;gap:6px}
.seg button,.legend button{background:var(--surface);border:1px solid var(--axis);border-radius:6px;
  padding:5px 10px;cursor:pointer;font-size:.9rem;display:inline-flex;align-items:center;gap:6px}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.legend button[aria-pressed=false]{opacity:.45;text-decoration:line-through}
button:focus-visible,select:focus-visible,input:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--base);outline-offset:2px}
.sw{width:14px;height:14px;flex:none}
.chart{width:100%;position:relative}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font-family:inherit;fill:var(--ink2);font-size:12px}
svg .axl{fill:var(--muted);font-size:12px}
svg .dl{fill:var(--ink);font-size:12px;font-weight:600}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .ring{stroke:var(--surface);stroke-width:2}
svg .pt{cursor:pointer}
svg .pt:focus{outline:none}
svg .pt:focus-visible{stroke:var(--ink);stroke-width:2.5}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;transition:opacity .15s}
svg .ref{stroke:var(--base);stroke-width:1.5;stroke-dasharray:6 4;fill:none}
svg .whisk{stroke-width:2;fill:none}
.pv-gen{--col:var(--p1)} .pv-refit{--col:var(--p2)} .pv-logp{--col:var(--p3)} .pv-train{--col:var(--p4)} .pv-misc{--col:var(--p5)}
.player{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin:8px 0}
.player button,.btn{background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:5px 14px;cursor:pointer;font-size:.9rem;min-width:72px}
.player button[aria-pressed=true]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.player input[type=range]{flex:1 1 160px;min-width:120px;accent-color:var(--base)}
.player output{font-size:.9rem;color:var(--ink2);font-variant-numeric:tabular-nums;min-width:8.5em}
.stack{display:flex;height:26px;border-radius:4px;overflow:hidden;background:var(--surface2);min-width:0}
.stack i{display:block;height:100%;background:var(--col);width:0;box-shadow:inset -2px 0 0 var(--surface);transition:width .35s ease}
ul.pleg{list-style:none;padding:0;margin:8px 0;display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.88rem;max-width:none}
ul.pleg li{margin:0} ul.pleg b{font-variant-numeric:tabular-nums;font-weight:600}
.srow{display:grid;grid-template-columns:64px minmax(0,1fr);gap:2px 10px;align-items:center;margin:8px 0}
.srow .sl{font-weight:600;font-size:.9rem} .srow .st2{grid-column:2;font-size:.84rem;color:var(--ink2)}
.brow.wide{grid-template-columns:minmax(0,170px) minmax(0,1fr) minmax(0,150px)}
.gtiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:12px 0}
.gtile{border:1px solid var(--border);border-top:4px solid var(--col);border-radius:8px;padding:10px 12px;font-size:.86rem;min-width:0}
.gtile h3{margin:0 0 4px;font-size:.95rem} .gtile dl{margin:0;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 10px}
.gtile dt{color:var(--ink2)} .gtile dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.diagwrap{overflow-x:auto;max-width:100%;border:1px solid var(--border);border-radius:8px;padding:6px}
.diagwrap svg{display:block;width:100%;min-width:740px;height:auto}
#arch-svg .cont rect{fill:none;stroke:var(--axis);stroke-width:1.2;stroke-dasharray:5 4}
#arch-svg .node rect{fill:var(--surface2);stroke:var(--axis);stroke-width:1.2}
#arch-svg .node{cursor:pointer} #arch-svg .node:focus{outline:none}
#arch-svg .node:hover rect,#arch-svg .node:focus-visible rect,#arch-svg .node.sel rect{stroke:var(--base);stroke-width:2.5}
#arch-svg .nl{fill:var(--ink);font-size:12.5px;font-weight:600} #arch-svg .ns{fill:var(--muted);font-size:11px}
#arch-svg .cl{fill:var(--ink2);font-size:12px;font-weight:600}
#arch-svg .wire{fill:none;stroke:var(--axis);stroke-width:1.3}
#arch-svg .gv{fill:var(--p1);opacity:.28;stroke:none} #arch-svg .gp{fill:var(--p4);opacity:.28;stroke:none}
#arch-svg .pk1{fill:var(--p1)} #arch-svg .pk2{fill:var(--p4)} #arch-svg .pk3{fill:var(--p2)}
#arch-desc{border-left:4px solid var(--base);background:var(--surface2);border-radius:0 6px 6px 0;padding:8px 12px;margin:10px 0;min-height:3.4em;font-size:.92rem;max-width:95ch}
ol.flow{list-style:none;padding:0;margin:10px 0;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;max-width:none;counter-reset:fl}
ol.flow li{counter-increment:fl;border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin:0;font-size:.86rem;background:var(--surface);
  transition:border-color .25s,background-color .25s,transform .25s}
ol.flow li::before{content:counter(fl);display:inline-block;width:1.5em;height:1.5em;line-height:1.5em;text-align:center;border-radius:50%;
  background:var(--surface2);font-size:.78rem;font-weight:600;margin-right:6px}
ol.flow li b{font-weight:600} ol.flow li .ds{display:block;color:var(--muted);font-size:.8rem;margin:2px 0}
ol.flow li .tm{display:block;font-variant-numeric:tabular-nums;font-weight:650}
ol.flow li.on{border-color:var(--base);background:var(--surface2);transform:translateY(-3px);box-shadow:0 0 0 2px var(--base)}
.cv-c0{--col:var(--c0)} .cv-c01{--col:var(--c01)} .cv-c04{--col:var(--c04)} .cv-c10{--col:var(--c10)}
.cv-base{--col:var(--base)} .cv-orig{--col:var(--c04)} .cv-tuned{--col:var(--tuned)}
.fillc{fill:var(--col)} .strokec{stroke:var(--col)}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--ink);color:var(--surface);
  padding:8px 10px;border-radius:6px;font-size:.82rem;line-height:1.4;max-width:260px;display:none;
  box-shadow:0 2px 8px rgba(0,0,0,.25)}
#tip b{font-weight:650}
#tip .tr{display:flex;justify-content:space-between;gap:14px;font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto;max-width:100%;border:1px solid var(--border);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}
td.wrap{white-space:normal;min-width:150px}
thead th{background:var(--surface2);color:var(--ink2);font-weight:600;position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
th button{background:none;border:0;padding:0;font:inherit;font-weight:600;color:inherit;cursor:pointer;white-space:nowrap}
th button .ar{display:inline-block;width:1em;color:var(--muted)}
.chip{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--col);margin-right:6px;vertical-align:baseline}
.flag{background:var(--warn-bg);border:1px solid var(--warn-border);border-radius:6px;padding:1px 6px;font-size:.78rem;white-space:nowrap}
.pending{border:1px dashed var(--axis);border-radius:8px;padding:12px 14px;margin:10px 0;background:var(--surface2)}
.pending .st{display:inline-block;border:1px solid var(--warn-border);background:var(--warn-bg);border-radius:999px;
  padding:0 10px;font-size:.8rem;font-weight:600;margin-right:8px}
.arch{display:grid;gap:14px;margin-top:8px}
.arch .m{display:grid;grid-template-columns:minmax(0,230px) minmax(0,1fr);gap:4px 14px;align-items:center;
  padding:10px 0;border-top:1px solid var(--grid)}
.arch .m:first-child{border-top:0}
.arch .nm{font-weight:600}
.arch .kd{font-size:.82rem;color:var(--muted)}
.brow{display:grid;grid-template-columns:48px minmax(0,1fr) 92px;gap:8px;align-items:center;font-size:.85rem;margin:3px 0}
.brow .lab{color:var(--ink2)}
.brow .trk{height:16px;background:var(--surface2);border-radius:4px;overflow:hidden}
.brow .bar{height:100%;width:0;background:var(--col);border-radius:0 4px 4px 0;transition:width .7s cubic-bezier(.2,.7,.2,1)}
.brow .val{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.gbar{display:inline-block;height:10px;background:var(--tuned);border-radius:0 3px 3px 0;vertical-align:middle;margin-right:8px}
ol.find{padding-left:1.3em;max-width:90ch} ol.find li{margin:8px 0}
ul{padding-left:1.2em;max-width:90ch} ul li{margin:5px 0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:8px 28px}
details{margin:8px 0} summary{cursor:pointer;color:var(--ink2);font-size:.92rem}
footer{color:var(--muted);font-size:.82rem;margin-top:24px}
@media (max-width:600px){
  section{padding:14px} h1{font-size:1.4rem}
  .arch .m{grid-template-columns:minmax(0,1fr)}
  .brow.wide{grid-template-columns:minmax(0,1fr) auto} .brow.wide .trk{grid-column:1/-1;order:3}
  .tile .v{font-size:1.4rem}
}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{transition:none !important;animation:none !important;scroll-behavior:auto !important}
}
</style>
</head>
<body>
<main>
<header>
  <h1>Reinforcement learning with a length term: accuracy, answer length and energy</h1>
  <p class="lede">Twelve GRPO runs on one task (T20 cricket rate and milestone questions with exact numeric answers)
  with <code>nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16</code>, varying how strongly the reward penalises
  long answers: length coefficient c = 0, 0.1, 0.4 and 1.0, three seeds each, 60 training steps per run.
  All results below are on 600 held-out questions with four sampled answers each (2,400 answers per evaluation).</p>
  <div class="tiles" id="tiles">
    <div class="tile" style="border-top-color:var(--base)"><div class="k">Base model, no training</div>
      <div class="v">@@base_acc@@%</div><div class="d">about @@base_tok@@ output tokens per answer; @@base_trunc@@% of answers cut off (mean of two evaluations)</div></div>
    <div class="tile" style="border-top-color:var(--c0)"><div class="k">Best accuracy: c = 0</div>
      <div class="v">@@c0_acc@@%</div><div class="d">@@c0_tok@@ tokens; seeds @@c0_acc_lo@@&ndash;@@c0_acc_hi@@%</div></div>
    <div class="tile" style="border-top-color:var(--c01)"><div class="k">Best trade-off: c = 0.1</div>
      <div class="v">@@c01_acc@@%</div><div class="d">@@c01_tok@@ tokens; seeds @@c01_acc_lo@@&ndash;@@c01_acc_hi@@%</div></div>
    <div class="tile" style="border-top-color:var(--c10)"><div class="k">Fewest tokens: c = 1.0</div>
      <div class="v">@@c10_tok@@ tokens</div><div class="d">@@c10_acc@@% accuracy; seeds @@c10_acc_lo@@&ndash;@@c10_acc_hi@@%</div></div>
    <div class="tile"><div class="k">Compute used</div>
      <div class="v">@@train_gpu_h@@ GPU-h</div><div class="d">12 runs, @@n_train_jobs@@ training jobs (@@train_node_h@@ node-hours on 4 nodes &times; 4 GPUs); evaluation @@eval_gpu_h@@ GPU-hours</div></div>
  </div>
</header>

<section aria-labelledby="h-setup">
  <h2 id="h-setup">Set-up</h2>
  <div class="two">
    <div>
      <h3>Task and model</h3>
      <ul>
        <li>Task ("cricket"): T20 cricket rate and milestone questions with exact numeric answers; 458,702 training questions, 600 held out. Three question types of 200 held-out questions each: run rate, chase rate, milestone.</li>
        <li>Model: <code>nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16</code>, a hybrid Mamba-Transformer with mixture of experts.</li>
        <li>Hardware: Rhea cluster, NVIDIA Vera Rubin GPUs (4 per node, 286,524 MiB each), aarch64.</li>
      </ul>
    </div>
    <div>
      <h3>Training</h3>
      <ul>
        <li>GRPO in NVIDIA NeMo RL with a rank-32 LoRA. Each step: 64 questions &times; 8 sampled answers. 60 steps per run on 4 nodes (16 GPUs).</li>
        <li>Generation with vLLM 0.33: temperature 1.0, top-p 0.95, at most 4,096 new tokens.</li>
        <li>Reward from a NeMo Gym environment: a finished, exactly correct answer scores <code>1 &minus; c &times; output_tokens / 4096</code>; anything else scores 0. c is the length coefficient; c = 0 is a plain correct/incorrect reward.</li>
      </ul>
    </div>
    <div>
      <h3>Evaluation (same harness for every result unless marked)</h3>
      <ul>
        <li>NeMo Gym 0.7.0; 600 held-out questions &times; 4 sampled answers = 2,400 answers; temperature 1.0, top-p 0.95, at most 4,096 output tokens.</li>
        <li>vLLM 0.31 on one GPU, serving the adapter as a LoRA module, 64 concurrent requests.</li>
        <li>"Cut off" means the answer reached the 4,096-token limit before finishing.</li>
      </ul>
    </div>
    <div>
      <h3>Energy</h3>
      <ul>
        <li>Energy = mean power of the serving GPU (sampled every 15 s) &times; duration of the evaluation run.</li>
        <li>The duration includes NeMo Gym start-up. That matters more for short runs, so the throughput of the tuned models is understated.</li>
        <li>Idle power with the model loaded is about 490&ndash;520 W and is included in the energy figures.</li>
      </ul>
    </div>
  </div>
</section>


<section aria-labelledby="h-sys">
  <h2 id="h-sys">System architecture</h2>
  <p class="sub">How one training job is laid out on the cluster. Select or hover a component for a description. The moving dots show the direction of requests (blue),
  Ray tasks and weights (violet) and logged outputs (amber); they are illustrative, not measured traffic.</p>
  <div class="player"><button type="button" id="arch-play" aria-pressed="true">Pause</button><span class="note" style="margin:0">Animation of the diagram and of the step below</span></div>
  <div class="diagwrap"><svg id="arch-svg" viewBox="0 0 760 462" role="group" aria-label="Diagram of one training job: a Slurm job of four nodes with four GPUs each, a Ray head on node 1 and Ray workers on nodes 2 to 4, NeMo Gym servers and the driver script on node 1, and three logged outputs"></svg></div>
  <div id="arch-desc" aria-live="polite"></div>
  <h3>One training step</h3>
  <div class="controls">
    <label for="fl-c">Average seconds for coefficient</label>
    <select id="fl-c"><option value="0">c = 0</option><option value="0.1">c = 0.1</option><option value="0.4">c = 0.4</option><option value="1.0">c = 1.0</option></select>
    <div class="seg" id="fl-seg" role="group" aria-label="Steps averaged">
      <button type="button" data-r="all" aria-pressed="true">Steps 2&ndash;60</button>
      <button type="button" data-r="early" aria-pressed="false">Steps 2&ndash;10</button>
      <button type="button" data-r="late" aria-pressed="false">Steps 51&ndash;60</button>
    </div>
  </div>
  <ol class="flow" id="fl-list"></ol>
  <p class="note" id="fl-note"></p>
</section>

<section aria-labelledby="h-scatter">
  <h2 id="h-scatter">Accuracy versus answer length</h2>
  <p class="sub">Each point is one evaluation of 2,400 answers. Further left is shorter; higher is more accurate.
  Select a legend entry to hide or show it. Hover or focus a point for details.</p>
  <div class="controls">
    <label><input type="checkbox" id="sc-means"> Show per-coefficient means with min&ndash;max whiskers</label>
  </div>
  <div class="legend" id="sc-legend" role="group" aria-label="Series shown in the scatter plot"></div>
  <div class="chart" id="ch-scatter"></div>
  <p class="note">Base model: @@b1_acc@@% and @@b2_acc@@% accuracy, @@b1_tok@@ and @@b2_tok@@ mean output tokens,
  @@b1_trunc@@% and @@b2_trunc@@% of answers cut off (two evaluations). The earlier 100-step run (c = 0.4, seed 42):
  @@o1_acc@@% at @@o1_tok@@ tokens, and @@o2_acc@@% at @@o2_tok@@ tokens when the same adapter was re-evaluated two days later.
  The token axis is logarithmic.</p>
</section>

<section aria-labelledby="h-type">
  <h2 id="h-type">Per question type</h2>
  <p class="sub">Held-out accuracy of each run by coefficient. Within each coefficient the three dots are seeds 1, 2, 3 from left to right;
  the short bar is their mean. The dashed line is the base model (mean of two evaluations).</p>
  <div class="controls">
    <label for="ty-sel">Question type</label>
    <select id="ty-sel">
      <option value="overall">Overall (600 questions)</option>
      <option value="run_rate">Run rate (200 questions)</option>
      <option value="chase_rate">Chase rate (200 questions)</option>
      <option value="milestone">Milestone (200 questions)</option>
    </select>
  </div>
  <div class="chart" id="ch-type"></div>
  <p class="note">One seed at c = 0.4 (<code>sweep-c0.4-s2</code>) scored @@s2_mile@@% on milestone questions while scoring
  @@s2_rr@@% on run rate and @@s2_cr@@% on chase rate (@@s2_all@@% overall).</p>
</section>

<section aria-labelledby="h-train">
  <h2 id="h-train">Training curves</h2>
  <p class="sub">One line per run, 60 steps each; colour is the coefficient. Hover over the plot for run, step and value.</p>
  <div class="controls">
    <label for="train-sel">Metric</label>
    <select id="train-sel">
      <option value="1">Average reward per step</option>
      <option value="2">Mean answer length (tokens)</option>
      <option value="3">Step time (seconds)</option>
    </select>
    <span id="train-checks" role="group" aria-label="Coefficients shown" style="display:inline-flex;flex-wrap:wrap;gap:6px 14px"></span>
  </div>
  <div class="chart" id="ch-train"></div>
  <p class="note" id="train-note"></p>
  <p class="note">Mean step time, first 10 steps &rarr; last 10 steps: c = 0: @@c0_step_first@@ &rarr; @@c0_step_last@@ s;
  c = 0.1: @@c01_step_first@@ &rarr; @@c01_step_last@@ s; c = 0.4: @@c04_step_first@@ &rarr; @@c04_step_last@@ s;
  c = 1.0: @@c10_step_first@@ &rarr; @@c10_step_last@@ s.
  Mean training GPU-hours per run from Slurm elapsed time (16 GPUs; includes start-up, and for resumed runs the steps repeated after the checkpoint):
  c = 0: @@c0_gpuh@@; c = 0.1: @@c01_gpuh@@; c = 0.4: @@c04_gpuh@@; c = 1.0: @@c10_gpuh@@.
  @@n_resumed@@ of the 12 runs hit the job time limit and were resumed from a checkpoint; for those, each step shows the value from the later log.</p>
</section>


<section aria-labelledby="h-phase">
  <h2 id="h-phase">Where the time goes in a step</h2>
  <p class="sub">Seconds per phase of each training step, from the training log. Choose a run, then play or drag the slider: the bar shows the selected step,
  drawn to the same scale as the slowest step of that run, so it shrinks as the step gets faster.</p>
  <div class="controls">
    <label for="ph-run">Run</label><select id="ph-run"></select>
  </div>
  <div class="player">
    <button type="button" id="ph-play" aria-pressed="false">Play</button>
    <input type="range" id="ph-step" min="1" max="60" step="1" value="1" aria-label="Training step">
    <output id="ph-out" for="ph-step"></output>
  </div>
  <div class="stack" id="ph-bar" role="img" aria-label="Phase times of the selected step"></div>
  <ul class="pleg" id="ph-leg"></ul>
  <p class="note" id="ph-info" aria-live="off"></p>
  <div class="chart" id="ch-phase"></div>
  <h3>Share of step time by coefficient</h3>
  <div class="controls">
    <div class="seg" id="ph-seg" role="group" aria-label="Steps averaged">
      <button type="button" data-r="all" aria-pressed="true">Steps 2&ndash;60</button>
      <button type="button" data-r="early" aria-pressed="false">Steps 2&ndash;10</button>
      <button type="button" data-r="late" aria-pressed="false">Steps 51&ndash;60</button>
    </div>
  </div>
  <div id="ph-sum"></div>
  <p style="max-width:90ch"><strong>What the data shows.</strong> In steps 2&ndash;10 generation takes @@gen_early_lo@@&ndash;@@gen_early_hi@@ s per step at every coefficient.
  By steps 51&ndash;60 it is @@ph_c0_late_gen@@ s at c = 0, @@ph_c01_late_gen@@ s at c = 0.1, @@ph_c04_late_gen@@ s at c = 0.4 and @@ph_c10_late_gen@@ s at c = 1.0.
  Log-probabilities and the gradient update also get shorter with the answers. The refit does not: it stays between @@refit_lo@@ and @@refit_hi@@ s in every coefficient and period
  (the weight transfer itself is about @@transfer@@ s of that). So in the last ten steps at c = 1.0 the refit (@@ph_c10_late_refit@@ s, @@ph_c10_late_refitp@@% of the step) takes longer than
  generation (@@ph_c10_late_gen@@ s, @@ph_c10_late_genp@@%), while at c = 0 generation is still @@ph_c0_late_genp@@% of the step and the refit @@ph_c0_late_refitp@@%.
  Averaged over steps 2&ndash;60, generation / refit / policy training take @@ph_c0_all_genp@@% / @@ph_c0_all_refitp@@% / @@ph_c0_all_trainp@@% of a @@ph_c0_all_total@@ s step at c = 0 and
  @@ph_c10_all_genp@@% / @@ph_c10_all_refitp@@% / @@ph_c10_all_trainp@@% of a @@ph_c10_all_total@@ s step at c = 1.0.</p>
  <p class="note">Groups: generation = <code>generation</code>; refit = <code>prepare_for_generation_total</code>; log-probabilities = <code>logprob_inference_prep</code> + <code>policy_and_reference_logprobs</code>;
  policy training = <code>training_prep</code> + <code>policy_training</code>; checkpointing and other = <code>checkpointing</code> (logged every 5 steps, about @@ckpt_s@@ s when written) + step time not covered by a logged phase.
  The first step after a start or a resume is slower because of warm-up (mean @@warm_first@@ s against @@warm_next@@ s for the step after it, over @@warm_n@@ job starts).
  Step 1 is left out of the averages; the first step after a resume is not.</p>
</section>

<section aria-labelledby="h-eng">
  <h2 id="h-eng">vLLM engines: busy versus idle</h2>
  <p class="sub">During the generation phase of each step, the training log records how long the 16 vLLM engines were busy and idle on average, and the largest number of requests in flight on one engine.</p>
  <div class="controls">
    <label for="eng-sel">Metric</label>
    <select id="eng-sel">
      <option value="busy">Busy share of the generation phase (%)</option>
      <option value="active">Mean active time per engine (seconds)</option>
      <option value="idle">Mean idle time per engine (seconds)</option>
      <option value="inflight">Largest number of requests in flight on one engine</option>
    </select>
    <span id="eng-checks" role="group" aria-label="Coefficients shown" style="display:inline-flex;flex-wrap:wrap;gap:6px 14px"></span>
  </div>
  <div class="chart" id="ch-eng"></div>
  <p class="note" id="eng-note"></p>
  <p style="max-width:90ch"><strong>What the data shows.</strong> Busy share = active / (active + idle). In steps 2&ndash;10 the engines are busy
  @@eng_c0_early_busy@@% (c = 0), @@eng_c01_early_busy@@% (c = 0.1), @@eng_c04_early_busy@@% (c = 0.4) and @@eng_c10_early_busy@@% (c = 1.0) of the generation phase.
  In steps 51&ndash;60 that is @@eng_c0_late_busy@@%, @@eng_c01_late_busy@@%, @@eng_c04_late_busy@@% and @@eng_c10_late_busy@@%.
  At c = 1.0 the engines are then active for @@eng_c10_late_active@@ s and idle for @@eng_c10_late_idle@@ s per step. So as answers shorten, a growing part of the remaining generation phase is
  engines waiting, not decoding; generation capacity is not what limits the late steps. The log does not say what the engines wait for.
  The largest number of requests in flight on one engine averages @@infl_lo@@&ndash;@@infl_hi@@ across the four coefficients (512 requests over 16 engines is 32 each if spread evenly).</p>
</section>

<section aria-labelledby="h-health">
  <h2 id="h-health">Training health</h2>
  <p class="sub">Loss and generation KL error per step. The KL error is the mismatch between the log-probabilities of the vLLM sampler and of the trainer, a check that the weights pushed to vLLM match the trainer's.</p>
  <div class="controls">
    <label for="health-sel">Metric</label>
    <select id="health-sel">
      <option value="kl">Generation KL error</option>
      <option value="loss">Loss</option>
    </select>
    <span id="health-checks" role="group" aria-label="Coefficients shown" style="display:inline-flex;flex-wrap:wrap;gap:6px 14px"></span>
  </div>
  <div class="chart" id="ch-health"></div>
  <p class="note" id="health-note"></p>
  <p style="max-width:90ch"><strong>What the data shows.</strong> The generation KL error stayed between @@kl_min@@ and @@kl_max@@ over all 720 steps (median per coefficient @@kl_med_lo@@&ndash;@@kl_med_hi@@),
  including the steps after a resume. The highest values late in training are in <code>@@klr_run@@</code>: its mean over steps 51&ndash;60 is @@klr_last@@,
  against @@klr_first@@ over its first ten steps and @@klr_olo@@&ndash;@@klr_ohi@@ over steps 51&ndash;60 in the other eleven runs. The median loss is @@loss_c0_med@@ at c = 0, @@loss_c01_med@@ at c = 0.1, @@loss_c04_med@@ at c = 0.4 and @@loss_c10_med@@ at c = 1.0, with single-step maxima of
  @@loss_c0_max@@, @@loss_c01_max@@, @@loss_c04_max@@ and @@loss_c10_max@@. The loss scale differs by orders of magnitude between c = 0 and the runs with a length term; the log does not explain why,
  so the loss is shown as a per-run signal and not for comparison across coefficients.</p>
</section>

<section aria-labelledby="h-gpu">
  <h2 id="h-gpu">GPU utilisation, memory, power and temperature during training</h2>
  <p class="sub">Per-minute means over all 16 GPUs of a training job (<code>nvidia-smi</code> every 15 s on each of the 4 nodes). Each GPU has 286,524 MiB (@@cap_gib@@ GiB).
  A resumed run has two jobs that ran hours apart; they are drawn one after the other against minutes of job time.</p>
  <div class="gtiles" id="gp-tiles"></div>
  <div class="controls">
    <label for="gp-run">Run</label><select id="gp-run"></select>
    <label for="gp-met">Metric</label>
    <select id="gp-met">
      <option value="util">Utilisation, mean of 16 GPUs (%)</option>
      <option value="mem">Memory in use per GPU, mean of 16 (GiB)</option>
      <option value="peak">Highest single-GPU memory (GiB)</option>
      <option value="power">Power per GPU, mean of 16 (W)</option>
      <option value="temp">Temperature, mean of 16 GPUs (&deg;C)</option>
      <option value="tmax">Hottest GPU (&deg;C)</option>
    </select>
  </div>
  <div class="chart" id="ch-gpu"></div>
  <h3>Replay</h3>
  <div class="player">
    <button type="button" id="gp-play" aria-pressed="false">Play</button>
    <input type="range" id="gp-min" min="0" max="1" step="1" value="0" aria-label="Minute of job time">
    <output id="gp-out" for="gp-min"></output>
  </div>
  <div id="gp-meters"></div>
  <p class="note" id="gp-info"></p>
  <p style="max-width:90ch"><strong>What the data shows.</strong> Estimated energy per training run (sum over job minutes of mean power &times; 16 GPUs):
  @@g_c0_kwh@@ kWh at c = 0 (seeds @@g_c0_kwh_lo@@&ndash;@@g_c0_kwh_hi@@), @@g_c01_kwh@@ kWh at c = 0.1 (@@g_c01_kwh_lo@@&ndash;@@g_c01_kwh_hi@@),
  @@g_c04_kwh@@ kWh at c = 0.4 (@@g_c04_kwh_lo@@&ndash;@@g_c04_kwh_hi@@) and @@g_c10_kwh@@ kWh at c = 1.0 (@@g_c10_kwh_lo@@&ndash;@@g_c10_kwh_hi@@).
  The difference is almost entirely run time (@@g_c0_minutes@@ against @@g_c10_minutes@@ job minutes on average): mean power per GPU is @@g_c0_power@@ W at c = 0 and @@g_c10_power@@ W at c = 1.0,
  against about @@empty_power@@ W in the minutes before anything is loaded on the GPUs.
  Long-answer (c = 0) and short-answer (c = 1.0) runs differ little in utilisation: @@g_c0_util@@% against @@g_c10_util@@% averaged over all job minutes
  (@@g_c0_util_loaded@@% against @@g_c10_util_loaded@@% once the models are loaded). Mean memory in use per GPU once loaded is @@g_c0_mem_gib@@ GiB (@@g_c0_mem_pct@@% of a GPU) at c = 0
  and @@g_c10_mem_gib@@ GiB (@@g_c10_mem_pct@@%) at c = 1.0. The highest single-GPU value is the same in both: @@g_c0_peak_gib@@ GiB (@@g_c0_peak_pct@@%) and @@g_c10_peak_gib@@ GiB (@@g_c10_peak_pct@@%),
  so close to half of each GPU's memory was never used.</p>
  <p class="note">These are means over 16 GPUs and over a minute. Generation and training alternate within a minute and across GPUs, so the figures do not show how busy a single GPU is inside one phase.
  Energy includes job start-up and, for resumed runs, the second start-up and the steps repeated after the checkpoint.
  In the replay, the training step shown for a minute is an estimate: the logged step times are laid end to end from an estimated start of training about 9&ndash;10 minutes into the job, so it can be off by about a minute.
  <code>sweep-c0.1-s3</code> has a short second job that logged no training steps; its GPU minutes are counted in the energy.</p>
</section>

<section aria-labelledby="h-energy">
  <h2 id="h-energy">Throughput and energy</h2>
  <p class="sub">Measured during the evaluation run on one serving GPU. Bars are the mean of three seeds; dots are the individual seeds;
  the dashed line is the base model (one evaluation).</p>
  <div class="controls">
    <label for="en-sel">Metric</label>
    <select id="en-sel">
      <option value="mwh">Energy per correct answer (mWh)</option>
      <option value="aps">Answers per second</option>
      <option value="power">Average GPU power (W)</option>
    </select>
  </div>
  <div class="chart" id="ch-energy"></div>
  <p class="note">Energy is one evaluation run per adapter, with @@ps_lo@@ to @@ps_hi@@ power samples per tuned run (15 s apart), so individual values are coarse.
  Durations include NeMo Gym start-up, which understates answers per second for the short runs. Idle power with the model loaded (about 490&ndash;520 W) is included.</p>
</section>

<section aria-labelledby="h-arch">
  <h2 id="h-arch">Three architectures, one harness</h2>
  <p class="sub">Base model versus RL-tuned adapter, all evaluated with the same NeMo Gym harness. The Nemotron adapter was trained with NeMo RL (GRPO, 100 steps, c = 0.4);
  the two Qwen adapters were trained with the project's own RLVR script, not NeMo RL.</p>
  <div class="controls">
    <div class="seg" id="ar-seg" role="group" aria-label="Metric">
      <button type="button" data-m="acc" aria-pressed="true">Accuracy</button>
      <button type="button" data-m="tok" aria-pressed="false">Output tokens</button>
      <button type="button" data-m="trunc" aria-pressed="false">Cut-off rate</button>
      <button type="button" data-m="mwh" aria-pressed="false">Energy per correct answer</button>
    </div>
  </div>
  <div class="arch" id="ch-arch"></div>
  <p class="note" id="ar-note"></p>
</section>

<section aria-labelledby="h-cost">
  <h2 id="h-cost">Training cost</h2>
  <p class="sub">Earlier runs on the same task, from the project's run tracker. <span class="flag">different scorer</span>
  Rows marked <code>eval.py</code> were scored with a different script from the NeMo Gym harness used elsewhere in this report, so they are only roughly comparable with it and with the last two rows.</p>
  <div class="tablewrap"><table id="tb-cost">
    <thead><tr><th class="l">Method</th><th class="l">Model</th><th>GPUs &times; hours</th><th class="l">GPU-hours</th>
      <th>Accuracy before &rarr; after</th><th>Tokens before &rarr; after</th><th class="l">Scored with</th></tr></thead>
    <tbody></tbody></table></div>
  <p class="note">GPU-hours for the earlier runs are as recorded in the tracker (GPUs &times; wall-clock hours, rounded). The sweep row is the mean per run from Slurm elapsed time:
  @@train_gpu_h@@ GPU-hours over 12 runs = @@gpuh_per_run@@ GPU-hours per run; its accuracy and token ranges are across the four coefficients.</p>
</section>

<section aria-labelledby="h-table">
  <h2 id="h-table">All twelve sweep runs</h2>
  <p class="sub">Select a column header to sort; select it again to reverse the order.</p>
  <div class="tablewrap"><table id="tb-runs">
    <thead><tr></tr></thead><tbody></tbody></table></div>
  <p class="note">Accuracy, cut-off and per-type columns are percentages of 2,400 (overall) or 800 (per type) sampled answers. Power is the mean of the serving GPU during the evaluation.</p>
</section>

<section aria-labelledby="h-find">
  <h2 id="h-find">Findings</h2>
  <ol class="find">
    <li><strong>A plain correct/incorrect reward (c = 0) was enough to raise accuracy and shorten answers.</strong>
      Accuracy rose from about 86% to @@c0_acc@@% (mean of 3 seeds, range @@c0_acc_lo@@&ndash;@@c0_acc_hi@@), and answers were about @@c0_shorter@@% shorter
      (about @@base_tok@@ &rarr; @@c0_tok@@ tokens) with no length reward. Cut-offs at the token limit went from about 10% to about 0 (@@c0_trunc@@%).</li>
    <li><strong>The length term buys tokens, not accuracy.</strong>
      c = 0.1: @@c01_acc@@% (@@c01_acc_lo@@&ndash;@@c01_acc_hi@@) at @@c01_tok@@ tokens (@@c01_tok_lo@@&ndash;@@c01_tok_hi@@);
      c = 0.4: @@c04_acc@@% (@@c04_acc_lo@@&ndash;@@c04_acc_hi@@) at @@c04_tok@@ tokens (@@c04_tok_lo@@&ndash;@@c04_tok_hi@@);
      c = 1.0: @@c10_acc@@% (@@c10_acc_lo@@&ndash;@@c10_acc_hi@@) at @@c10_tok@@ tokens (@@c10_tok_lo@@&ndash;@@c10_tok_hi@@).
      No setting with a length term had a higher mean accuracy than c = 0.</li>
    <li><strong>Seed-to-seed spread is larger with a length term.</strong>
      The min&ndash;max range of accuracy across seeds was @@c0_acc_spread@@ points at c = 0, @@c01_acc_spread@@ at c = 0.1, @@c04_acc_spread@@ at c = 0.4 and @@c10_acc_spread@@ at c = 1.0;
      it grows from 0 to 0.4 and is smaller again at 1.0, though with three seeds the last two cannot be ranked.
      One seed at c = 0.4 (<code>sweep-c0.4-s2</code>) fell to @@s2_mile@@% on milestone questions while staying above 99% on the other two types.</li>
    <li><strong>Energy per correct answer falls with answer length.</strong>
      Base model: @@base_mwh@@ mWh per correct answer. Means of three seeds: @@c0_mwh@@ mWh at c = 0, @@c01_mwh@@ at c = 0.1, @@c04_mwh@@ at c = 0.4, @@c10_mwh@@ at c = 1.0.</li>
    <li><strong>Training gets faster as answers shorten.</strong>
      Step time drops during every run as answers shorten, and drops sooner and further with a larger coefficient (see the step-time curves and the figures under them).</li>
    <li><strong>Late in training, fixed costs dominate the step.</strong>
      Generation falls from @@gen_early_lo@@&ndash;@@gen_early_hi@@ s per step (steps 2&ndash;10) to @@ph_c10_late_gen@@ s at c = 1.0 (steps 51&ndash;60), while the refit of the vLLM engines stays at
      @@refit_lo@@&ndash;@@refit_hi@@ s and becomes the largest phase (@@ph_c10_late_refitp@@% of the step). In the same steps the engines are busy only @@eng_c10_late_busy@@% of the generation phase.
      Training energy per run falls from @@g_c0_kwh@@ kWh (c = 0) to @@g_c10_kwh@@ kWh (c = 1.0), because the run is shorter, not because the GPUs draw less power.</li>
    <li><strong>The same recipe worked on three architectures.</strong>
      Same harness: Nemotron (hybrid Mamba-Transformer, mixture of experts) @@b1_acc@@% &rarr; @@o1_acc@@%, @@b1_tok@@ &rarr; @@o1_tok@@ tokens;
      Qwen3.8-27B (dense) @@q38_acc0@@% &rarr; @@q38_acc1@@%, @@q38_tok0@@ &rarr; @@q38_tok1@@ tokens;
      Qwen3.6-35B-A3B (mixture of experts) @@q36_acc0@@% &rarr; @@q36_acc1@@%, @@q36_tok0@@ &rarr; @@q36_tok1@@ tokens,
      with its milestone accuracy going down from @@q36_mile0@@% to @@q36_mile1@@%.</li>
  </ol>
  <h3>Related negative result</h3>
  <p style="max-width:90ch">Speculative decoding (tried on 2026-11-23 on the RLVR-tuned Qwen3.8-27B) was slower than the target model alone:
  @@sd_alone@@ tokens/s alone versus @@sd_lo@@&ndash;@@sd_hi@@ tokens/s with a draft model (best @@sd_ratio@@&times;).</p>
</section>

<section aria-labelledby="h-notm">
  <h2 id="h-notm">What was not measured</h2>
  <p style="max-width:90ch">No network, CPU, disk or host-memory telemetry was collected for these jobs. The only system data are the <code>nvidia-smi</code> GPU metrics
  (utilisation, memory, power, temperature), the per-phase timings in the training log and the Ray driver's memory lines. This report therefore has no network metrics,
  and it cannot say how much of the refit or of the engines' idle time is network transfer.</p>
</section>

<section aria-labelledby="h-cav">
  <h2 id="h-cav">Caveats</h2>
  <ul>
    <li>Three seeds per coefficient. Means and min&ndash;max ranges are shown; no significance test is claimed.</li>
    <li>The accuracy ranges of c = 0.4 (@@c04_acc_lo@@&ndash;@@c04_acc_hi@@%) and c = 1.0 (@@c10_acc_lo@@&ndash;@@c10_acc_hi@@%) overlap, so the order of those two settings is not established.</li>
    <li>One task. A second task (GSM8K) has been submitted and has no results yet.</li>
    <li>Energy is one evaluation run per adapter with few power samples (@@ps_lo@@ to @@ps_hi@@ for the tuned sweep runs). Durations include harness start-up.</li>
    <li>Training reward includes the length term, so it is not comparable across coefficients. Use held-out accuracy for comparisons.</li>
    <li>The sweep runs were 60 steps; the earlier 100-step run used a different seed (42) and step count, so it is shown for context, not as a fourth seed of c = 0.4.</li>
    <li>The two Qwen models were trained with a different RL implementation (the project's RLVR script). The training-cost table mixes two scoring scripts and is marked accordingly.</li>
  </ul>
</section>

<section aria-labelledby="h-pos">
  <h2 id="h-pos">Positioning versus published work</h2>
  <p style="max-width:90ch">Based on a search of paper abstracts on 2026-11-26; the papers themselves were not read in full, so this section is provisional.</p>
  <p style="max-width:90ch"><strong>Already published:</strong> rewarding correct answers and penalising length in reinforcement learning. Examples found:
  Short-RL (arXiv 2505.12284), AALC (arXiv 2506.20160), Powered Length Penalty (arXiv 2506.10446), Concise Reasoning via RL (arXiv 2504.05185), LEASH (ACL 2026), DDCA (arXiv 2602.02099).
  The method used here should not be presented as new.</p>
  <p style="max-width:90ch"><strong>What these runs appear to add is empirical:</strong></p>
  <ul>
    <li>a measured accuracy&ndash;tokens&ndash;energy curve with seed spread on a hybrid Mamba mixture-of-experts model;</li>
    <li>the c = 0 control, which separates the effect of the correctness reward from the effect of the length term;</li>
    <li>results on three architectures with the same evaluation harness;</li>
    <li>measured GPU power on Vera Rubin;</li>
    <li>a training-cost comparison with LoRA on distilled traces.</li>
  </ul>
</section>

<section aria-labelledby="h-pend">
  <h2 id="h-pend">What is still pending</h2>
  <div class="pending"><span class="st">pending</span><strong>Second task: GSM8K.</strong> Submitted on 2026-11-28; no results yet.
  Until it finishes, every statement in this report is about one task.</div>
  <ul>
    <li>More than three seeds would be needed to rank c = 0.4 against c = 1.0.</li>
    <li>Repeated energy measurements per adapter, with start-up time excluded, would be needed to tighten the throughput and energy figures.</li>
  </ul>
</section>

<footer>Generated by <code>build_report.py</code> from <code>data/sweep_raw.txt</code>. Sweep numbers are computed from that file; set-up facts, earlier runs and the list of published work are recorded in the builder.</footer>
</main>
<div id="tip" role="tooltip"></div>

<script id="data" type="application/json">@@DATA@@</script>
<script>
(function(){
"use strict";
var D = JSON.parse(document.getElementById('data').textContent);
var NS = 'http://www.w3.org/2000/svg';
var CK = {'0':'c0','0.1':'c01','0.4':'c04','1.0':'c10'};
var SHAPE = {'0':'circle','0.1':'square','0.4':'triangle','1.0':'diamond','base':'plus','orig':'triangle'};
var TYPE_LABEL = {overall:'Overall',run_rate:'Run rate',chase_rate:'Chase rate',milestone:'Milestone'};

function fmt(x,d){ return (x==null||!isFinite(x)) ? 'n/a' : x.toLocaleString('en-US',{minimumFractionDigits:d||0,maximumFractionDigits:d||0}); }
function el(tag,attrs,parent,text){
  var e=document.createElement(tag); if(attrs) for(var k in attrs) e.setAttribute(k,attrs[k]);
  if(text!=null) e.textContent=text; if(parent) parent.appendChild(e); return e;
}
function S(tag,attrs,parent,text){
  var e=document.createElementNS(NS,tag); if(attrs) for(var k in attrs) e.setAttribute(k,attrs[k]);
  if(text!=null) e.textContent=text; if(parent) parent.appendChild(e); return e;
}
function lin(d0,d1,r0,r1){ var f=function(v){return r0+(v-d0)/(d1-d0)*(r1-r0);}; f.inv=function(p){return d0+(p-r0)/(r1-r0)*(d1-d0);}; return f; }
function logs(d0,d1,r0,r1){ var a=Math.log(d0),b=Math.log(d1); return function(v){return r0+(Math.log(v)-a)/(b-a)*(r1-r0);}; }
function niceMax(v){ var p=Math.pow(10,Math.floor(Math.log10(v))); var m=v/p; var s=[1,1.2,1.5,2,2.5,3,4,5,6,8,10]; for(var i=0;i<s.length;i++) if(m<=s[i]) return s[i]*p; return 10*p; }
function ticks(max,nn){ var step=max/nn, out=[]; for(var i=0;i<=nn;i++) out.push(+(i*step).toPrecision(6)); return out; }
function shapePath(shape,x,y,r){
  if(shape==='circle') return 'M'+(x-r)+','+y+'a'+r+','+r+' 0 1,0 '+(2*r)+',0a'+r+','+r+' 0 1,0 '+(-2*r)+',0z';
  if(shape==='square'){ var s=r*0.9; return 'M'+(x-s)+','+(y-s)+'h'+(2*s)+'v'+(2*s)+'h'+(-2*s)+'z'; }
  if(shape==='triangle'){ var t=r*1.15; return 'M'+x+','+(y-t)+'L'+(x+t)+','+(y+t*0.8)+'L'+(x-t)+','+(y+t*0.8)+'z'; }
  if(shape==='diamond'){ var q=r*1.25; return 'M'+x+','+(y-q)+'L'+(x+q)+','+y+'L'+x+','+(y+q)+'L'+(x-q)+','+y+'z'; }
  var a=r*1.15,b=r*0.42; /* plus */
  return 'M'+(x-b)+','+(y-a)+'h'+(2*b)+'v'+(a-b)+'h'+(a-b)+'v'+(2*b)+'h'+(-(a-b))+'v'+(a-b)+'h'+(-2*b)+'v'+(-(a-b))+'h'+(-(a-b))+'v'+(-2*b)+'h'+(a-b)+'z';
}
function swatch(key,hollow){
  var s=S('svg',{viewBox:'0 0 16 16','class':'sw cv-'+(CK[key]||key),'aria-hidden':'true'});
  var p=S('path',{d:shapePath(SHAPE[key],8,8,5.5)},s);
  if(hollow){ p.setAttribute('fill','none'); p.setAttribute('class','strokec'); p.setAttribute('stroke-width','2'); }
  else p.setAttribute('class','fillc');
  return s;
}

/* ---------- tooltip ---------- */
var tip=document.getElementById('tip');
function tipRows(title,rows){
  var h='<b>'+title+'</b>';
  rows.forEach(function(r){ h+='<div class="tr"><span>'+r[0]+'</span><span>'+r[1]+'</span></div>'; });
  return h;
}
function showTip(html,x,y){
  tip.innerHTML=html; tip.style.display='block';
  var w=tip.offsetWidth,h=tip.offsetHeight,vw=document.documentElement.clientWidth,vh=window.innerHeight;
  var left=x+14; if(left+w>vw-6) left=x-w-14; if(left<6) left=6;
  var top=y+14; if(top+h>vh-6) top=y-h-14; if(top<6) top=6;
  tip.style.left=left+'px'; tip.style.top=top+'px';
}
function hideTip(){ tip.style.display='none'; }
function bindTip(node,htmlFn){
  node.addEventListener('pointerenter',function(e){ showTip(htmlFn(),e.clientX,e.clientY); });
  node.addEventListener('pointermove',function(e){ showTip(htmlFn(),e.clientX,e.clientY); });
  node.addEventListener('pointerleave',hideTip);
  node.addEventListener('focus',function(){ var r=node.getBoundingClientRect(); showTip(htmlFn(),r.left+r.width/2,r.top+r.height/2); });
  node.addEventListener('blur',hideTip);
}
window.addEventListener('scroll',hideTip,{passive:true});

/* ---------- chart frame ---------- */
function frame(host,label,hRatio,hMin,hMax){
  host.textContent='';
  var w=Math.max(280,Math.round(host.clientWidth||700));
  var h=Math.round(Math.max(hMin||280,Math.min(hMax||430,w*(hRatio||0.55))));
  var svg=S('svg',{viewBox:'0 0 '+w+' '+h,role:'img','aria-label':label,preserveAspectRatio:'xMidYMid meet'},host);
  return {svg:svg,w:w,h:h,narrow:w<520};
}
function yAxis(f,m,sc,tk,fm,label){
  tk.forEach(function(t){
    var y=sc(t);
    S('line',{x1:m.l,x2:f.w-m.r,y1:y,y2:y,'class':'grid'},f.svg);
    S('text',{x:m.l-8,y:y+4,'text-anchor':'end'},f.svg,fm(t));
  });
  S('text',{x:14,y:(m.t+f.h-m.b)/2,'text-anchor':'middle',transform:'rotate(-90 14 '+((m.t+f.h-m.b)/2)+')','class':'axl'},f.svg,label);
}
function xAxis(f,m,sc,tk,fm,label){
  S('line',{x1:m.l,x2:f.w-m.r,y1:f.h-m.b,y2:f.h-m.b,'class':'axis'},f.svg);
  tk.forEach(function(t){
    var x=sc(t);
    S('line',{x1:x,x2:x,y1:f.h-m.b,y2:f.h-m.b+5,'class':'axis'},f.svg);
    S('text',{x:x,y:f.h-m.b+19,'text-anchor':'middle'},f.svg,fm(t));
  });
  S('text',{x:(m.l+f.w-m.r)/2,y:f.h-6,'text-anchor':'middle','class':'axl'},f.svg,label);
}
function evalRows(s,en){
  var o=s.overall, r=[['Accuracy',fmt(o.acc,2)+'%'],['Mean output tokens',fmt(o.tok,0)],['Cut off at limit',fmt(o.trunc,2)+'%'],
    ['Run rate / chase / milestone',fmt(s.run_rate.acc,1)+' / '+fmt(s.chase_rate.acc,1)+' / '+fmt(s.milestone.acc,1)+'%']];
  if(en){ r.push(['Answers per second',fmt(en.aps,1)]); r.push(['Energy per correct',fmt(en.mwh,1)+' mWh']); }
  return r;
}

/* ---------- 2. scatter ---------- */
var scHidden={}, scMeans=false;
var SC_KEYS=[['base','Base model (2 evaluations)'],['0','c = 0'],['0.1','c = 0.1'],['0.4','c = 0.4'],['1.0','c = 1.0'],['orig','100-step run, c = 0.4 (2 evaluations)']];
(function(){
  var lg=document.getElementById('sc-legend');
  SC_KEYS.forEach(function(k){
    var b=el('button',{type:'button','aria-pressed':'true'},lg);
    b.appendChild(swatch(k[0],k[0]==='orig')); b.appendChild(document.createTextNode(k[1]));
    b.addEventListener('click',function(){
      scHidden[k[0]]=!scHidden[k[0]]; b.setAttribute('aria-pressed',scHidden[k[0]]?'false':'true'); drawScatter();
    });
  });
  document.getElementById('sc-means').addEventListener('change',function(e){ scMeans=e.target.checked; drawScatter(); });
})();
function drawScatter(){
  var f=frame(document.getElementById('ch-scatter'),'Scatter plot of held-out accuracy against mean output tokens for each run',0.56,300,460);
  var m={l:58,r:18,t:14,b:46};
  var x=logs(50,2600,m.l,f.w-m.r), y=lin(85,100.6,f.h-m.b,m.t);
  yAxis(f,m,y,[86,88,90,92,94,96,98,100],function(t){return t+'%';},'Held-out accuracy (%)');
  xAxis(f,m,x,f.narrow?[50,100,200,500,1000,2000]:[50,70,100,200,300,500,700,1000,2000],function(t){return fmt(t,0);},'Mean output tokens per answer (log scale)');
  var r=f.narrow?5:6;
  function point(key,cls,s,en,title,extra,hollow,dim){
    var o=s.overall;
    var p=S('path',{d:shapePath(SHAPE[key],x(o.tok),y(o.acc),r),'class':'pt ring cv-'+cls+(hollow?'':' fillc'),tabindex:'0',
      'aria-label':title+': '+fmt(o.acc,2)+'% accuracy, '+fmt(o.tok,0)+' tokens'},f.svg);
    if(hollow){ p.setAttribute('fill','var(--surface)'); p.setAttribute('class','pt strokec cv-'+cls); p.setAttribute('stroke-width','2.2'); }
    if(dim) p.setAttribute('opacity','0.45');
    bindTip(p,function(){ return tipRows(title,(extra||[]).concat(evalRows(s,en))); });
  }
  if(!scHidden.base){
    D.baseEvals.forEach(function(b){ point('base','base',b.sum,b.en,b.label,[]); });
    S('text',{x:x(D.base.tok),y:y(D.base.acc)+22,'text-anchor':'middle','class':'dl'},f.svg,'base model');
  }
  if(!scHidden.orig){
    D.origEvals.forEach(function(b){ point('orig','orig',b.sum,b.en,b.label,[['Training steps','100']],true); });
  }
  D.coefs.forEach(function(c){
    if(scHidden[c]) return;
    D.sweep.filter(function(r){return r.c===c;}).forEach(function(r){
      point(c,CK[c],r.sum,r.en,r.run,[['Coefficient / seed','c = '+c+' / '+r.seed]],false,scMeans);
    });
  });
  if(scMeans) D.coefs.forEach(function(c){
    if(scHidden[c]) return;
    var s=D.coef[c], cx=x(s.tok.mean), cy=y(s.acc.mean), g=S('g',{'class':'cv-'+CK[c]},f.svg), cap=4;
    S('path',{'class':'whisk strokec',d:'M'+x(s.tok.min)+','+cy+'H'+x(s.tok.max)+'M'+x(s.tok.min)+','+(cy-cap)+'v'+(2*cap)+'M'+x(s.tok.max)+','+(cy-cap)+'v'+(2*cap)+
      'M'+cx+','+y(s.acc.min)+'V'+y(s.acc.max)+'M'+(cx-cap)+','+y(s.acc.min)+'h'+(2*cap)+'M'+(cx-cap)+','+y(s.acc.max)+'h'+(2*cap)},g);
    var p=S('path',{d:shapePath(SHAPE[c],cx,cy,r+3),'class':'pt ring fillc',tabindex:'0','aria-label':'Mean of 3 seeds, c = '+c+': '+fmt(s.acc.mean,2)+'% accuracy, '+fmt(s.tok.mean,0)+' tokens'},g);
    bindTip(p,function(){ return tipRows('c = '+c+', mean of 3 seeds',[['Accuracy',fmt(s.acc.mean,2)+'%'],['Accuracy min–max',fmt(s.acc.min,2)+'–'+fmt(s.acc.max,2)+'%'],
      ['Mean output tokens',fmt(s.tok.mean,0)],['Tokens min–max',fmt(s.tok.min,0)+'–'+fmt(s.tok.max,0)],['Energy per correct',fmt(s.mwh.mean,1)+' mWh']]); });
    S('text',{x:cx,y:y(s.acc.min)+17,'text-anchor':'middle','class':'dl'},f.svg,'c = '+c);
  });
}

/* ---------- 3. per type dot plot ---------- */
var tySel=document.getElementById('ty-sel');
tySel.addEventListener('change',drawType);
function drawType(){
  var t=tySel.value;
  var f=frame(document.getElementById('ch-type'),'Dot plot of held-out accuracy per run by length coefficient, question type '+TYPE_LABEL[t],0.5,280,400);
  var m={l:58,r:18,t:14,b:46};
  var y=lin(85,100.6,f.h-m.b,m.t), bw=(f.w-m.l-m.r)/4;
  yAxis(f,m,y,[86,88,90,92,94,96,98,100],function(v){return v+'%';},'Held-out accuracy (%)');
  S('line',{x1:m.l,x2:f.w-m.r,y1:f.h-m.b,y2:f.h-m.b,'class':'axis'},f.svg);
  S('text',{x:(m.l+f.w-m.r)/2,y:f.h-6,'text-anchor':'middle','class':'axl'},f.svg,'Length coefficient c (dimensionless)');
  var by=y(D.base.types[t]);
  S('line',{x1:m.l,x2:f.w-m.r,y1:by,y2:by,'class':'ref'},f.svg);
  S('text',{x:f.w-m.r-4,y:by+(D.base.types[t]>97?16:-6),'text-anchor':'end','class':'dl'},f.svg,'base model '+fmt(D.base.types[t],2)+'%');
  var off=Math.min(26,bw/5), r=f.narrow?5:6;
  D.coefs.forEach(function(c,i){
    var cx=m.l+bw*(i+0.5), runs=D.sweep.filter(function(r){return r.c===c;});
    var vals=runs.map(function(r){return r.sum[t].acc;}), mean=vals.reduce(function(a,b){return a+b;},0)/vals.length;
    S('text',{x:cx,y:f.h-m.b+19,'text-anchor':'middle'},f.svg,'c = '+c);
    var g=S('g',{'class':'cv-'+CK[c]},f.svg);
    S('line',{x1:cx-off*1.7,x2:cx+off*1.7,y1:y(mean),y2:y(mean),'class':'strokec','stroke-width':3,'stroke-linecap':'round',opacity:0.55},g);
    runs.forEach(function(run,j){
      var p=S('path',{d:shapePath(SHAPE[c],cx+(j-1)*off,y(run.sum[t].acc),r),'class':'pt ring fillc',tabindex:'0',
        'aria-label':run.run+', '+TYPE_LABEL[t]+': '+fmt(run.sum[t].acc,2)+'%'},g);
      bindTip(p,function(){ var o=run.sum[t]; return tipRows(run.run,[['Question type',TYPE_LABEL[t]],['Accuracy',fmt(o.acc,2)+'%'],
        ['Mean output tokens',fmt(o.tok,0)],['Cut off at limit',fmt(o.trunc,2)+'%'],['Mean of 3 seeds',fmt(mean,2)+'%']]); });
    });
    S('text',{x:cx,y:Math.min(y(Math.min.apply(null,vals))+20,f.h-m.b-4),'text-anchor':'middle','class':'dl'},f.svg,fmt(mean,2)+'%');
  });
}

/* ---------- shared: line-chart panel (metric select + coefficient checkboxes) ---------- */
var redraws=[];
var RM=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var IX={}; D.ST.forEach(function(k,i){ IX[k]=i; });
function $(id){ return document.getElementById(id); }
function symlog(v){ return (v<0?-1:1)*Math.log10(1+Math.abs(v)/0.01); }
function segGroup(id,cb){
  var g=$(id);
  g.addEventListener('click',function(e){
    var b=e.target.closest('button'); if(!b) return;
    g.querySelectorAll('button').forEach(function(x){ x.setAttribute('aria-pressed',x===b?'true':'false'); });
    cb(b.getAttribute('data-r'));
  });
}
function curvePanel(P){
  var sel=$(P.id+'-sel'), host=$(P.id+'-checks'), show={};
  D.coefs.forEach(function(c){
    show[c]=true;
    var l=el('label',null,host), cb=el('input',{type:'checkbox'},l); cb.checked=true;
    l.appendChild(swatch(c)); l.appendChild(document.createTextNode('c = '+c));
    cb.addEventListener('change',function(){ show[c]=cb.checked; draw(); });
  });
  function draw(){
    var M=P.metrics[sel.value];
    $(P.id+'-note').textContent=M.note;
    var f=frame($('ch-'+P.id),'Line chart of '+M.name+' against training step, one line per run',0.52,300,440);
    var m={l:66,r:18,t:14,b:46};
    var all=D.sweep.map(function(r){ return {r:r,pts:M.get(r)}; });
    var runs=all.filter(function(q){ return show[q.r.c]; });
    var x=lin(1,60,m.l,f.w-m.r), y, tk, fm;
    if(M.scale==='log'){ y=logs(M.dom[0],M.dom[1],f.h-m.b,m.t); tk=M.ticks; fm=function(v){return fmt(v,0);}; }
    else if(M.scale==='sym'){ var l0=lin(symlog(M.dom[0]),symlog(M.dom[1]),f.h-m.b,m.t); y=function(v){return l0(symlog(v));}; tk=M.ticks; fm=function(v){return String(v);}; }
    else { var mx=M.dom?M.dom[1]:0;
      if(!M.dom){ all.forEach(function(q){ q.pts.forEach(function(p){ if(p[1]>mx) mx=p[1]; }); }); mx=niceMax(mx); }
      y=lin(0,mx,f.h-m.b,m.t); tk=ticks(mx,5); fm=function(v){return fmt(v,M.td||0);}; }
    yAxis(f,m,y,tk,fm,M.label);
    xAxis(f,m,x,[1,10,20,30,40,50,60],function(v){return v;},'Training step');
    var paths=[];
    runs.forEach(function(q){
      var d=''; q.pts.forEach(function(p,i){ d+=(i?'L':'M')+x(p[0]).toFixed(1)+','+y(p[1]).toFixed(1); });
      paths.push(S('path',{d:d,'class':'line strokec cv-'+CK[q.r.c]},f.svg));
    });
    if(!runs.length){ S('text',{x:f.w/2,y:f.h/2,'text-anchor':'middle'},f.svg,'No coefficient selected'); return; }
    var cross=S('line',{y1:m.t,y2:f.h-m.b,'class':'axis',visibility:'hidden'},f.svg);
    var dot=S('circle',{r:5,'class':'ring',visibility:'hidden'},f.svg);
    var hit=S('rect',{x:m.l,y:m.t,width:f.w-m.l-m.r,height:f.h-m.t-m.b,fill:'transparent'},f.svg);
    function move(e){
      var bb=f.svg.getBoundingClientRect(), sx=f.w/bb.width, px=(e.clientX-bb.left)*sx, py=(e.clientY-bb.top)*sx;
      var step=Math.max(1,Math.min(60,Math.round(x.inv(px)))), best=-1, bd=1e9;
      runs.forEach(function(q,i){ var p=q.pts[step-1]; if(!p) return; var dd=Math.abs(y(p[1])-py); if(dd<bd){bd=dd;best=i;} });
      if(best<0) return;
      var q=runs[best], p=q.pts[step-1];
      paths.forEach(function(pa,i){ pa.setAttribute('opacity',i===best?1:0.25); pa.setAttribute('stroke-width',i===best?3:2); });
      cross.setAttribute('x1',x(step)); cross.setAttribute('x2',x(step)); cross.setAttribute('visibility','visible');
      dot.setAttribute('cx',x(step)); dot.setAttribute('cy',y(p[1])); dot.setAttribute('class','ring fillc cv-'+CK[q.r.c]); dot.setAttribute('visibility','visible');
      var rows=[['Coefficient / seed','c = '+q.r.c+' / '+q.r.seed],['Step',step],[M.name,fmt(p[1],M.d)+M.unit]];
      if(q.r.st[step-1][IX.warm]) rows.push(['Note',step===1?'first step (warm-up)':'first step after resume']);
      showTip(tipRows(q.r.run,rows),e.clientX,e.clientY);
    }
    function leave(){ paths.forEach(function(pa){ pa.removeAttribute('opacity'); pa.setAttribute('stroke-width',2); });
      cross.setAttribute('visibility','hidden'); dot.setAttribute('visibility','hidden'); hideTip(); }
    hit.addEventListener('pointermove',move); hit.addEventListener('pointerdown',move); hit.addEventListener('pointerleave',leave);
  }
  sel.addEventListener('change',draw); redraws.push(draw);
}
function trainCol(k){ return function(r){ return r.train.map(function(p){ return [p[0],p[k]]; }); }; }
function stCol(k){ return function(r){ return r.st.map(function(p){ return [p[0],p[IX[k]]]; }); }; }

/* ---------- 4. training curves ---------- */
curvePanel({id:'train',metrics:{
  '1':{label:'Average reward per step (0–1)',name:'Average reward',d:3,unit:'',dom:[0,1],td:1,get:trainCol(1),
       note:'Reward is not comparable across coefficients: it includes the length term, so the same answers score lower at a larger c. Compare coefficients on held-out accuracy instead.'},
  '2':{label:'Mean answer length (output tokens, log scale)',name:'Mean answer tokens',d:0,unit:' tokens',scale:'log',dom:[40,3000],ticks:[50,100,200,500,1000,2000],get:trainCol(2),
       note:'Mean output tokens of the 512 answers sampled in each training step (64 questions × 8 answers).'},
  '3':{label:'Step time (seconds)',name:'Step time',d:0,unit:' s',get:trainCol(3),
       note:'Wall-clock seconds per training step on 16 GPUs. The first step of a job, and the first step after a resume, is slower because of warm-up.'}
}});

/* ---------- C. vLLM engines ---------- */
curvePanel({id:'eng',metrics:{
  busy:{label:'Busy share of the generation phase (%)',name:'Busy share',d:0,unit:'%',dom:[0,100],
        get:function(r){ return r.st.map(function(p){ return [p[0],p[IX.active]/(p[IX.active]+p[IX.idle])*100]; }); },
        note:'Mean active time divided by mean active plus mean idle time of the 16 vLLM engines during the generation phase of the step.'},
  active:{label:'Mean active time per engine (seconds)',name:'Mean active time',d:1,unit:' s',get:stCol('active'),note:'Average time the 16 vLLM engines spent busy during the generation phase of the step.'},
  idle:{label:'Mean idle time per engine (seconds)',name:'Mean idle time',d:1,unit:' s',get:stCol('idle'),note:'Average time the 16 vLLM engines spent idle during the generation phase of the step.'},
  inflight:{label:'Requests in flight on one engine (count)',name:'Largest number in flight',d:0,unit:'',get:stCol('inflight'),note:'Largest number of requests in flight on a single vLLM engine during the step. It varies widely from step to step.'}
}});

/* ---------- D. training health ---------- */
curvePanel({id:'health',metrics:{
  kl:{label:'Generation KL error (as logged)',name:'Generation KL error',d:4,unit:'',td:4,get:stCol('kl'),note:'Mismatch between the log-probabilities of the vLLM sampler and of the trainer for the sampled answers. Small and flat means the weight sync is working.'},
  loss:{label:'Loss (symmetric log scale)',name:'Loss',d:4,unit:'',scale:'sym',dom:[-0.1,200],ticks:[-0.1,0,0.1,1,10,100],get:stCol('loss'),
        note:'Loss per step on a symmetric log scale (linear near zero), because it spans from slightly negative values to above 100. Not comparable across coefficients.'}
}});

/* ---------- A. where the time goes ---------- */
var PH=[['gen','Generation'],['refit','Refit (merge LoRA, push weights to vLLM)'],['logp','Log-probabilities'],['train','Policy training'],['misc','Checkpointing and other']];
function phaseOf(x){
  var p={gen:x[IX.gen],refit:x[IX.refit],logp:x[IX.lpprep]+x[IX.logp],train:x[IX.train]+x[IX.tprep],misc:(x[IX.ckpt]||0)+x[IX.other]};
  p.total=p.gen+p.refit+p.logp+p.train+p.misc; return p;
}
(function(){
  var runSel=$('ph-run'), play=$('ph-play'), rng=$('ph-step'), out=$('ph-out'), bar=$('ph-bar'), leg=$('ph-leg'), info=$('ph-info');
  D.sweep.forEach(function(r,i){ el('option',{value:i},runSel,r.run+' (c = '+r.c+', seed '+r.seed+')'); });
  runSel.value='9';
  var segs=[], vals=[], step=1, timer=null, cur=null, xs=null, mxT=1;
  PH.forEach(function(p){
    segs.push(el('i',{'class':'pv-'+p[0]},bar));
    var li=el('li',null,leg); el('span',{'class':'chip pv-'+p[0]},li); li.appendChild(document.createTextNode(p[1]+': ')); vals.push(el('b',null,li));
  });
  function run(){ return D.sweep[+runSel.value]; }
  function setStep(sv){
    step=sv; rng.value=sv;
    var r=run(), x=r.st[sv-1], p=phaseOf(x);
    out.textContent='Step '+sv+' of 60';
    PH.forEach(function(q,i){ segs[i].style.width=(p[q[0]]/mxT*100)+'%'; vals[i].textContent=fmt(p[q[0]],1)+' s ('+fmt(p[q[0]]/p.total*100,0)+'%)'; });
    info.textContent='Step '+sv+': '+fmt(x[IX.total],1)+' s in total; mean answer length '+fmt(r.train[sv-1][2],0)+' tokens'+
      (x[IX.warm]?(sv===1?'. First step of the job: slower because of warm-up.':'. First step after the resume from a checkpoint: slower because of warm-up.'):'.');
    if(cur&&xs){ cur.setAttribute('x1',xs(sv)); cur.setAttribute('x2',xs(sv)); }
  }
  function stop(){ if(timer) clearInterval(timer); timer=null; play.textContent='Play'; play.setAttribute('aria-pressed','false'); }
  function draw(){
    var r=run(), ph=r.st.map(phaseOf);
    mxT=0; ph.forEach(function(p){ if(p.total>mxT) mxT=p.total; });
    var f=frame($('ch-phase'),'Stacked area chart of seconds per phase against training step for '+r.run,0.42,260,360);
    var m={l:58,r:18,t:14,b:46}, top=niceMax(mxT);
    var x=lin(1,60,m.l,f.w-m.r), y=lin(0,top,f.h-m.b,m.t);
    yAxis(f,m,y,ticks(top,5),function(v){return fmt(v,0);},'Seconds per step');
    xAxis(f,m,x,[1,10,20,30,40,50,60],function(v){return v;},'Training step');
    var base=ph.map(function(){return 0;});
    PH.forEach(function(q){
      var up='', dn='';
      ph.forEach(function(p,i){ var t=base[i]+p[q[0]]; up+=(i?'L':'M')+x(i+1).toFixed(1)+','+y(t).toFixed(1); dn='L'+x(i+1).toFixed(1)+','+y(base[i]).toFixed(1)+dn; base[i]=t; });
      S('path',{d:up+dn+'z','class':'fillc pv-'+q[0],stroke:'var(--surface)','stroke-width':1},f.svg);
    });
    cur=S('line',{y1:m.t,y2:f.h-m.b,stroke:'var(--ink)','stroke-width':1.5},f.svg); xs=x;
    var hit=S('rect',{x:m.l,y:m.t,width:f.w-m.l-m.r,height:f.h-m.t-m.b,fill:'transparent',style:'cursor:pointer'},f.svg);
    function at(e){ var bb=f.svg.getBoundingClientRect(); return Math.max(1,Math.min(60,Math.round(x.inv((e.clientX-bb.left)*f.w/bb.width)))); }
    hit.addEventListener('pointermove',function(e){
      var sv=at(e), p=ph[sv-1];
      showTip(tipRows(r.run+', step '+sv,PH.map(function(q){ return [q[1],fmt(p[q[0]],1)+' s']; }).concat([['Step total',fmt(r.st[sv-1][IX.total],1)+' s']])),e.clientX,e.clientY);
    });
    hit.addEventListener('pointerleave',hideTip);
    hit.addEventListener('pointerdown',function(e){ stop(); setStep(at(e)); });
    setStep(step);
  }
  play.addEventListener('click',function(){
    if(timer){ stop(); return; }
    if(step>=60) setStep(1);
    play.textContent='Pause'; play.setAttribute('aria-pressed','true');
    timer=setInterval(function(){ if(step>=60){ stop(); return; } setStep(step+1); },RM?700:230);
  });
  rng.addEventListener('input',function(){ stop(); setStep(+rng.value); });
  runSel.addEventListener('change',function(){ stop(); draw(); });
  redraws.push(draw);

  /* per-coefficient share */
  var host=$('ph-sum'), rows=[], R='all';
  D.coefs.forEach(function(c){
    var row=el('div',{'class':'srow'},host); var sl=el('div',{'class':'sl'},row); el('span',{'class':'chip cv-'+CK[c]},sl); sl.appendChild(document.createTextNode('c = '+c));
    var st=el('div',{'class':'stack',role:'img'},row), is=PH.map(function(q){ return el('i',{'class':'pv-'+q[0]},st); });
    rows.push({c:c,is:is,st:st,tx:el('div',{'class':'st2'},row)});
  });
  function upd(){
    rows.forEach(function(w){
      var p=D.phase[w.c][R], sum=p.gen+p.refit+p.logp+p.train+p.misc;
      function pc(k){ return fmt(p[k]/sum*100,0)+'%'; }
      PH.forEach(function(q,i){ w.is[i].style.width=(p[q[0]]/sum*100)+'%'; w.is[i].title=q[1]+': '+fmt(p[q[0]],1)+' s ('+pc(q[0])+')'; });
      var t=fmt(p.total,1)+' s per step: generation '+pc('gen')+' ('+fmt(p.gen,1)+' s), refit '+pc('refit')+' ('+fmt(p.refit,1)+' s), log-probabilities '+pc('logp')+
        ', policy training '+pc('train')+', checkpointing and other '+pc('misc');
      w.tx.textContent=t; w.st.setAttribute('aria-label','c = '+w.c+': '+t);
    });
  }
  segGroup('ph-seg',function(v){ R=v; upd(); });
  upd();
})();

/* ---------- B. GPU telemetry ---------- */
(function(){
  var CAP=D.memMiB/1024;
  var GM={ util:{i:0,label:'GPU utilisation, mean of 16 GPUs (%)',name:'Utilisation',d:1,unit:'%',max:100},
           mem:{i:1,label:'Memory in use per GPU, mean of 16 (GiB)',name:'Memory in use (mean)',d:1,unit:' GiB',max:CAP,k:1/1024,cap:true},
           peak:{i:2,label:'Highest single-GPU memory (GiB)',name:'Highest single GPU',d:1,unit:' GiB',max:CAP,k:1/1024,cap:true},
           power:{i:3,label:'Power per GPU, mean of 16 (W)',name:'Power per GPU',d:0,unit:' W',max:800},
           temp:{i:4,label:'Temperature, mean of 16 GPUs (°C)',name:'Temperature (mean)',d:1,unit:' °C',max:40},
           tmax:{i:5,label:'Temperature of the hottest GPU (°C)',name:'Hottest GPU',d:0,unit:' °C',max:40} };
  function gv(row,k){ var M=GM[k]; return row[M.i]*(M.k||1); }
  var tiles=$('gp-tiles');
  D.coefs.forEach(function(c){
    var g=D.gcoef[c], t=el('div',{'class':'gtile cv-'+CK[c]},tiles); el('h3',null,t,'c = '+c+' (mean of 3 runs)');
    var dl=el('dl',null,t);
    [['Mean utilisation',fmt(g.util,1)+'%'],['Utilisation once loaded',fmt(g.util_loaded,1)+'%'],
     ['Peak single-GPU memory',fmt(g.peak/1024,1)+' GiB ('+fmt(g.peak/D.memMiB*100,1)+'%)'],
     ['Mean memory once loaded',fmt(g.mem_loaded/1024,1)+' GiB ('+fmt(g.mem_loaded/D.memMiB*100,1)+'%)'],
     ['Mean power per GPU',fmt(g.power,0)+' W'],['Energy per run',fmt(g.kwh,1)+' kWh'],
     ['Job minutes per run',fmt(g.minutes,0)],['Highest temperature',fmt(g.tmax,0)+' °C']].forEach(function(q){ el('dt',null,dl,q[0]); el('dd',null,dl,q[1]); });
  });
  var runSel=$('gp-run'), metSel=$('gp-met'), play=$('gp-play'), rng=$('gp-min'), out=$('gp-out'), info=$('gp-info'), mh=$('gp-meters');
  D.sweep.forEach(function(r,i){ el('option',{value:i},runSel,r.run+' (c = '+r.c+', seed '+r.seed+')'); });
  runSel.value='0';
  var MET=[['util','Utilisation'],['mem','Memory in use (mean)'],['peak','Highest single GPU'],['power','Power per GPU'],['step','Step time (estimated step)']];
  var meters=MET.map(function(q,i){
    var br=el('div',{'class':'brow wide '+(i===4?'cv-tuned':'cv-base')},mh); el('span',{'class':'lab'},br,q[1]);
    var trk=el('div',{'class':'trk'},br), b=el('div',{'class':'bar'},trk); b.style.transitionDuration='.25s';
    return {bar:b,val:el('span',{'class':'val'},br)};
  });
  var GAP=8, flat=[], pos=0, timer=null, cur=null, xs=null, maxStep=1;
  function run(){ return D.sweep[+runSel.value]; }
  function estStep(sg,mi){
    if(sg.off==null) return null;
    var t=(mi+0.5-sg.off)*60, i; if(t<0) return null;
    for(i=0;i<sg.secs.length;i++){ if(t<sg.secs[i]) return {step:sg.s0+i,secs:sg.secs[i]}; t-=sg.secs[i]; }
    return null;
  }
  function setPos(pv){
    pos=Math.max(0,Math.min(flat.length-1,pv)); rng.value=pos;
    var q=flat[pos], row=q.sg.rows[q.m], es=estStep(q.sg,q.m);
    out.textContent='Minute '+(q.m+1)+' of '+q.sg.rows.length;
    ['util','mem','peak','power'].forEach(function(k,i){
      var v=gv(row,k); meters[i].bar.style.width=Math.min(100,v/GM[k].max*100)+'%';
      meters[i].val.textContent=fmt(v,GM[k].d)+GM[k].unit+(GM[k].cap?' ('+fmt(v/CAP*100,0)+'%)':'');
    });
    meters[4].bar.style.width=(es?Math.min(100,es.secs/maxStep*100):0)+'%';
    meters[4].val.textContent=es?fmt(es.secs,0)+' s (step '+es.step+')':'no step';
    info.textContent='Job '+q.sg.job+', minute '+(q.m+1)+' (job started '+q.sg.t0+' PST). '+
      (es?'Estimated training step '+es.step+', which took '+fmt(es.secs,1)+' s.':(q.sg.off==null?'This job logged no training steps.':'Start-up or shut-down: no training step in progress.'))+
      ' Temperature: mean '+fmt(row[4],1)+' °C, hottest GPU '+fmt(row[5],0)+' °C.';
    if(cur&&xs){ cur.setAttribute('x1',xs(q.x)); cur.setAttribute('x2',xs(q.x)); }
  }
  function stop(){ if(timer) clearInterval(timer); timer=null; play.textContent='Play'; play.setAttribute('aria-pressed','false'); }
  function draw(){
    var r=run(), k=metSel.value, M=GM[k], xo=0;
    flat=[]; maxStep=1;
    r.gpu.forEach(function(sg){ sg.x0=xo; sg.rows.forEach(function(row,mi){ flat.push({sg:sg,m:mi,x:xo+mi}); }); xo+=sg.rows.length+GAP;
      sg.secs.forEach(function(v){ if(v>maxStep) maxStep=v; }); });
    var total=xo-GAP; rng.max=flat.length-1; if(pos>flat.length-1) pos=0;
    var f=frame($('ch-gpu'),'Line chart of '+M.name+' per minute of job time for '+r.run,0.45,280,400);
    var m={l:58,r:18,t:30,b:46};
    var mx=M.cap?280:(k==='util'?100:0);
    if(!mx){ D.sweep.forEach(function(rr){ rr.gpu.forEach(function(sg){ sg.rows.forEach(function(row){ var v=gv(row,k); if(v>mx) mx=v; }); }); }); mx=niceMax(mx); }
    var x=lin(0,Math.max(total-1,1),m.l,f.w-m.r), y=lin(0,mx,f.h-m.b,m.t);
    yAxis(f,m,y,ticks(mx,M.cap?7:5),function(v){return fmt(v,0);},M.label);
    S('line',{x1:m.l,x2:f.w-m.r,y1:f.h-m.b,y2:f.h-m.b,'class':'axis'},f.svg);
    S('text',{x:(m.l+f.w-m.r)/2,y:f.h-6,'text-anchor':'middle','class':'axl'},f.svg,'Minutes of job time (each job starts at 0)');
    if(M.cap){ S('line',{x1:m.l,x2:f.w-m.r,y1:y(CAP),y2:y(CAP),'class':'ref'},f.svg);
      S('text',{x:f.w-m.r-4,y:y(CAP)+15,'text-anchor':'end','class':'dl'},f.svg,'GPU capacity '+fmt(CAP,1)+' GiB'); }
    var tstep=f.narrow?30:15;
    r.gpu.forEach(function(sg,si){
      var n=sg.rows.length, d='';
      for(var t=0;t<n;t+=tstep){ if(n-t<tstep*0.4&&t>0) break;
        S('line',{x1:x(sg.x0+t),x2:x(sg.x0+t),y1:f.h-m.b,y2:f.h-m.b+5,'class':'axis'},f.svg);
        S('text',{x:x(sg.x0+t),y:f.h-m.b+19,'text-anchor':'middle'},f.svg,t); }
      sg.rows.forEach(function(row,mi){ d+=(mi?'L':'M')+x(sg.x0+mi).toFixed(1)+','+y(gv(row,k)).toFixed(1); });
      S('path',{d:d,'class':'line strokec cv-'+CK[r.c]},f.svg);
      var short=x(sg.x0+n-1)-x(sg.x0)<150;
      S('text',{x:x(sg.x0),y:m.t-12,'class':'dl'},f.svg,(short?'':'job ')+sg.job+(short?'':(si>0?(sg.off==null?' (no steps logged)':' (resumed)'):(sg.state==='TIMEOUT'?' (hit time limit)':''))));
      if(si>0) S('line',{x1:x(sg.x0-GAP/2),x2:x(sg.x0-GAP/2),y1:m.t-8,y2:f.h-m.b,'class':'grid','stroke-dasharray':'3 3'},f.svg);
    });
    cur=S('line',{y1:m.t,y2:f.h-m.b,stroke:'var(--ink)','stroke-width':1.5},f.svg); xs=x;
    var hit=S('rect',{x:m.l,y:m.t,width:f.w-m.l-m.r,height:f.h-m.t-m.b,fill:'transparent',style:'cursor:pointer'},f.svg);
    function at(e){ var bb=f.svg.getBoundingClientRect(), xv=x.inv((e.clientX-bb.left)*f.w/bb.width), best=0, bd=1e9;
      flat.forEach(function(q,i){ var dd=Math.abs(q.x-xv); if(dd<bd){bd=dd;best=i;} }); return best; }
    hit.addEventListener('pointermove',function(e){
      var i=at(e), q=flat[i], row=q.sg.rows[q.m], es=estStep(q.sg,q.m);
      cur.setAttribute('x1',x(q.x)); cur.setAttribute('x2',x(q.x));
      showTip(tipRows('Job '+q.sg.job+', minute '+(q.m+1),[['Utilisation',fmt(row[0],1)+'%'],['Memory in use (mean)',fmt(row[1]/1024,1)+' GiB'],['Highest single GPU',fmt(row[2]/1024,1)+' GiB'],
        ['Power per GPU',fmt(row[3],0)+' W'],['Power, 16 GPUs',fmt(row[3]*16/1000,2)+' kW'],['Temperature mean / hottest',fmt(row[4],1)+' / '+fmt(row[5],0)+' °C'],
        ['Estimated step',es?es.step+' ('+fmt(es.secs,0)+' s)':'none']]),e.clientX,e.clientY);
    });
    hit.addEventListener('pointerleave',function(){ hideTip(); setPos(pos); });
    hit.addEventListener('pointerdown',function(e){ stop(); setPos(at(e)); });
    setPos(pos);
  }
  play.addEventListener('click',function(){
    if(timer){ stop(); return; }
    if(pos>=flat.length-1) setPos(0);
    play.textContent='Pause'; play.setAttribute('aria-pressed','true');
    timer=setInterval(function(){ if(pos>=flat.length-1){ stop(); return; } setPos(pos+1); },RM?600:170);
  });
  rng.addEventListener('input',function(){ stop(); setPos(+rng.value); });
  runSel.addEventListener('change',function(){ stop(); pos=0; draw(); });
  metSel.addEventListener('change',draw);
  redraws.push(draw);
})();

/* ---------- E. system architecture ---------- */
(function(){
  var svg=$('arch-svg'), desc=$('arch-desc'), playBtn=$('arch-play');
  var INFO={
    slurm:['Slurm job','One training job: 4 nodes with 4 NVIDIA Vera Rubin GPUs each (16 GPUs, 286,524 MiB per GPU, aarch64). A sweep run is one such job, or two when the first hit the time limit and was resumed from a checkpoint.'],
    head:['Node 1: Ray head','Node 1 runs the Ray head, the driver script and the NeMo Gym servers, and its 4 GPUs do the same work as the GPUs of the other nodes.'],
    worker:['Nodes 2–4: Ray workers','Each of nodes 2 to 4 runs a Ray worker that hosts the vLLM engines and policy workers for its 4 GPUs.'],
    gpu:['One GPU: vLLM engine + policy worker','Every one of the 16 GPUs is shared by a vLLM generation engine (samples the answers) and a NeMo Automodel policy worker (computes log-probabilities and the LoRA gradient update). The two take turns within a step.'],
    driver:['Driver script','Runs the GRPO loop: sample 64 questions, collect 8 answers each, compute rewards and advantages, update the rank-32 LoRA on 16 GPUs, then merge and push the new weights into the vLLM engines.'],
    ray:['Ray head','Schedules the generation engines and policy workers on the 16 GPUs and carries the calls between the driver and the workers on nodes 2 to 4.'],
    gym:['NeMo Gym servers','Three servers on the head node: the agent, the resources server and the model proxy. Together they turn a question into a rollout and a reward.'],
    agent:['NeMo Gym agent','Runs each rollout: sends the question to the model through the model proxy and passes the answer to the resources server.'],
    res:['NeMo Gym resources server','Checks the final number of each answer. A finished, exactly correct answer scores 1 − c × output_tokens / 4096; anything else scores 0.'],
    proxy:['NeMo Gym model proxy','Forwards generation requests to the 16 vLLM engines and returns the sampled answers.'],
    wandb:['W&B metrics','Per-step metrics logged by the driver: reward, answer length, loss, generation KL error and the per-phase timings used in this report.'],
    ckpt:['LoRA checkpoints','The adapter (95 MB) is saved every 5 steps. A resumed job restarts from the last checkpoint and repeats the steps after it.'],
    glog:['GPU logs','nvidia-smi sampled every 15 s on each of the 4 nodes: utilisation, memory, power and temperature per GPU. No network, CPU, disk or host-memory telemetry was collected.']
  };
  var selG=null;
  function showInfo(k,g){ desc.textContent=''; el('strong',null,desc,INFO[k][0]+'. '); desc.appendChild(document.createTextNode(INFO[k][1]));
    if(selG) selG.classList.remove('sel'); selG=g; if(g) g.classList.add('sel'); }
  function bind(g,k){
    g.setAttribute('tabindex','0'); g.setAttribute('role','button'); g.setAttribute('aria-label',INFO[k][0]);
    ['pointerenter','focus','click'].forEach(function(ev){ g.addEventListener(ev,function(e){ e.stopPropagation(); showInfo(k,g); }); });
    g.addEventListener('keydown',function(e){ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); showInfo(k,g); } });
  }
  function cont(k,x,y,w,h,label){ var g=S('g',{'class':'node cont'},svg); S('rect',{x:x,y:y,width:w,height:h,rx:8},g);
    var t=S('text',{x:x+10,y:y+17,'class':'cl'},g,label); bind(g,k); return g; }
  function box(k,x,y,w,h,label,sub){ var g=S('g',{'class':'node'},svg); S('rect',{x:x,y:y,width:w,height:h,rx:6},g);
    S('text',{x:x+w/2,y:y+(sub?h/2-3:h/2+4),'text-anchor':'middle','class':'nl'},g,label);
    if(sub) S('text',{x:x+w/2,y:y+h/2+13,'text-anchor':'middle','class':'ns'},g,sub); bind(g,k); return g; }
  function gpus(x0,y,h){ for(var i=0;i<4;i++){ var x=x0+i*82, g=S('g',{'class':'node'},svg);
    S('rect',{x:x,y:y,width:74,height:h,rx:5},g); S('rect',{x:x+1,y:y+1,width:72,height:h/2-1,'class':'gv'},g); S('rect',{x:x+1,y:y+h/2,width:72,height:h/2-1,'class':'gp'},g);
    S('text',{x:x+37,y:y+h/4+4,'text-anchor':'middle','class':'ns',style:'fill:var(--ink)'},g,'vLLM engine');
    S('text',{x:x+37,y:y+h*3/4+4,'text-anchor':'middle','class':'ns',style:'fill:var(--ink)'},g,'policy worker'); bind(g,'gpu'); } }
  var wires=[];
  function wire(d,cls,dur,rev){ S('path',{d:d,'class':'wire'},svg); wires.push([d,cls,dur,rev]); }
  cont('slurm',6,6,748,344,'Slurm job: 4 nodes × 4 Vera Rubin GPUs (16 GPUs)');
  cont('head',18,36,354,304,'Node 1: Ray head');
  [0,1,2].forEach(function(i){ cont('worker',392,36+i*103,350,96,'Node '+(i+2)+': Ray worker'); });
  /* wires first so that boxes cover their ends */
  wire('M303,170 V124','pk1',2.2); wire('M303,170 V124','pk1',2.2,true);
  [0,1,2].forEach(function(i){ var yy=36+i*103;
    wire('M348,197 H384 V'+(yy+52)+' H404','pk1',3+i*0.5); wire('M348,197 H384 V'+(yy+52)+' H404','pk1',3+i*0.5,true);
    wire('M358,288 H378 V'+(yy+78)+' H404','pk2',3.6+i*0.5); });
  wire('M90,250 V224','pk1',1.8); wire('M70,326 V384','pk3',2.6); wire('M150,326 V366 H380 V384','pk3',3.4); wire('M640,340 V384','pk3',2.6);
  gpus(30,62,62);
  [0,1,2].forEach(function(i){ gpus(404,60+i*103,62); });
  cont('gym',30,138,330,94,'NeMo Gym servers');
  box('agent',40,166,100,56,'Agent'); box('res',148,166,112,56,'Resources server','checks the number'); box('proxy',268,166,84,56,'Model proxy');
  box('driver',30,250,160,76,'Driver script','GRPO loop'); box('ray',200,250,160,76,'Ray head');
  box('wandb',18,384,230,66,'W&B metrics','per-step metrics'); box('ckpt',265,384,230,66,'LoRA checkpoints','95 MB adapter every 5 steps'); box('glog',512,384,230,66,'GPU logs','nvidia-smi every 15 s');
  wires.forEach(function(w,i){
    var c=S('circle',{r:4,'class':w[1]},svg), a=S('animateMotion',{dur:w[2]+'s',repeatCount:'indefinite',path:w[0],begin:(i%4)*0.4+'s'},c);
    if(w[3]){ a.setAttribute('keyPoints','1;0'); a.setAttribute('keyTimes','0;1'); a.setAttribute('calcMode','linear'); a.setAttribute('begin',(w[2]/2)+'s'); }
  });
  showInfo('slurm',null);

  /* one training step */
  var list=$('fl-list'), cSel=$('fl-c'), note=$('fl-note'), R='all', items=[], on=-1, ft=null;
  var STG=[['Sample 64 questions','from the 458,702 training questions',null],
           ['vLLM generates 8 answers each','512 answers on 16 engines','gen'],
           ['NeMo Gym verifies','the resources server checks the final number',null],
           ['Rewards','1 − c × tokens / 4096 if correct, otherwise 0',null],
           ['GRPO advantages','log-probabilities of the answers; leave-one-out baseline','logp'],
           ['LoRA update on 16 GPUs','rank-32 adapter, gradient update','train'],
           ['Merge and push weights to vLLM','the refit; then the next step starts','refit']];
  STG.forEach(function(q){ var li=el('li',null,list); el('b',null,li,q[0]); el('span',{'class':'ds'},li,q[1]); items.push({li:li,tm:el('span',{'class':'tm'},li),k:q[2]}); });
  function updFlow(){
    var p=D.phase[cSel.value][R];
    items.forEach(function(it){ it.tm.textContent=it.k?fmt(p[it.k],1)+' s ('+fmt(p[it.k]/p.total*100,0)+'% of the step)':'not timed separately'; });
    note.textContent='Mean seconds over the three seeds of c = '+cSel.value+'; mean step time '+fmt(p.total,1)+' s. Checkpointing and time not covered by a logged phase add '+fmt(p.misc,1)+
      ' s. Of the refit, '+fmt(p.transfer,1)+' s is the weight transfer itself. Sampling the questions, verification and the reward are not timed separately in the training log.';
  }
  function tick(){
    if(on>=0) items[on].li.classList.remove('on');
    on=(on+1)%items.length; items[on].li.classList.add('on');
    var p=D.phase[cSel.value][R], k=items[on].k;
    ft=setTimeout(tick,700+(k?3200*p[k]/p.total:0));
  }
  var playing=false;
  function setPlaying(v){
    playing=v; playBtn.textContent=v?'Pause':'Play'; playBtn.setAttribute('aria-pressed',v?'true':'false');
    if(v){ if(svg.unpauseAnimations) svg.unpauseAnimations(); if(!ft) tick(); }
    else { if(svg.pauseAnimations) svg.pauseAnimations(); clearTimeout(ft); ft=null; }
  }
  playBtn.addEventListener('click',function(){ setPlaying(!playing); });
  cSel.value='1.0'; cSel.addEventListener('change',updFlow);
  segGroup('fl-seg',function(v){ R=v; updFlow(); });
  updFlow(); setPlaying(!RM);
  if(RM){ items[0].li.classList.add('on'); on=0; }
})();

/* ---------- 5. throughput and energy ---------- */
var enSel=document.getElementById('en-sel');
var EN={ mwh:{label:'Energy per correct answer (mWh)',d:1,unit:' mWh'}, aps:{label:'Answers per second',d:1,unit:' /s'}, power:{label:'Average GPU power (W)',d:0,unit:' W'} };
enSel.addEventListener('change',drawEnergy);
function drawEnergy(){
  var k=enSel.value, M=EN[k];
  var f=frame(document.getElementById('ch-energy'),'Bar chart of '+M.label+' by length coefficient with the base model as a reference line',0.5,280,400);
  var m={l:58,r:18,t:20,b:46};
  var mx=D.base.en[k]; D.sweep.forEach(function(r){ if(r.en[k]>mx) mx=r.en[k]; }); mx=niceMax(mx*1.05);
  var y=lin(0,mx,f.h-m.b,m.t), bw=(f.w-m.l-m.r)/4, w=Math.min(64,bw*0.5);
  yAxis(f,m,y,ticks(mx,5),function(v){return fmt(v,mx<20?1:0);},M.label);
  S('text',{x:(m.l+f.w-m.r)/2,y:f.h-6,'text-anchor':'middle','class':'axl'},f.svg,'Length coefficient c (dimensionless)');
  if(k==='power'){
    S('rect',{x:m.l,y:y(520),width:f.w-m.l-m.r,height:y(490)-y(520),fill:'var(--axis)',opacity:0.3},f.svg);
    S('text',{x:m.l+6,y:y(490)+14,'class':'axl'},f.svg,'idle with model loaded, about 490–520 W');
  }
  D.coefs.forEach(function(c,i){
    var s=D.coef[c][k], cx=m.l+bw*(i+0.5), y0=y(0), y1=y(s.mean), g=S('g',{'class':'cv-'+CK[c]},f.svg), rr=Math.min(4,(y0-y1)/2);
    var bar=S('path',{'class':'fillc pt',tabindex:'0','aria-label':'c = '+c+', mean of 3 seeds: '+fmt(s.mean,M.d)+M.unit,
      d:'M'+(cx-w/2)+','+y0+'V'+(y1+rr)+'q0,'+(-rr)+' '+rr+','+(-rr)+'H'+(cx+w/2-rr)+'q'+rr+',0 '+rr+','+rr+'V'+y0+'z'},g);
    bindTip(bar,function(){ return tipRows('c = '+c+', mean of 3 seeds',[[M.label,fmt(s.mean,M.d)],['Min–max',fmt(s.min,M.d)+'–'+fmt(s.max,M.d)],
      ['Base model',fmt(D.base.en[k],M.d)],['Relative to base',fmt(s.mean/D.base.en[k],2)+'×']]); });
    S('text',{x:cx,y:f.h-m.b+19,'text-anchor':'middle'},f.svg,'c = '+c);
    var runs=D.sweep.filter(function(r){return r.c===c;}), top=y1;
    runs.forEach(function(run,j){
      var py=y(run.en[k]); if(py<top) top=py;
      var p=S('circle',{cx:cx+(j-1)*Math.min(14,w/4),cy:py,r:4.5,fill:'var(--ink)','class':'ring pt',tabindex:'0','aria-label':run.run+': '+fmt(run.en[k],M.d)+M.unit},f.svg);
      bindTip(p,function(){ return tipRows(run.run,[[M.label,fmt(run.en[k],M.d)],['Evaluation duration',fmt(run.en.seconds,0)+' s'],['Power samples',run.en.samples],
        ['Output tokens per second',fmt(run.en.tps,0)]]); });
    });
    S('text',{x:cx,y:top-9,'text-anchor':'middle','class':'dl'},f.svg,fmt(s.mean,M.d));
  });
  S('line',{x1:m.l,x2:f.w-m.r,y1:f.h-m.b,y2:f.h-m.b,'class':'axis'},f.svg);
  var by=y(D.base.en[k]);
  S('line',{x1:m.l,x2:f.w-m.r,y1:by,y2:by,'class':'ref'},f.svg);
  S('text',{x:k==='aps'?m.l+6:f.w-m.r-4,y:k==='aps'?by-6:by+16,'text-anchor':k==='aps'?'start':'end','class':'dl'},f.svg,'base model '+fmt(D.base.en[k],M.d)+M.unit);
}

/* ---------- 6. architectures ---------- */
var AR={ acc:{get:function(s,e){return s.overall.acc;},d:2,unit:'%',max:100,note:'Held-out accuracy over 2,400 sampled answers. Bars start at 0%.'},
         tok:{get:function(s,e){return s.overall.tok;},d:0,unit:' tokens',note:'Mean output tokens per answer.'},
         trunc:{get:function(s,e){return s.overall.trunc;},d:2,unit:'%',note:'Share of answers that reached the 4,096-token limit before finishing. All three tuned adapters had none.'},
         mwh:{get:function(s,e){return e.mwh;},d:1,unit:' mWh',note:'Energy per correct answer, serving GPU, idle power included. For Nemotron the energy figures come from the re-evaluation two days after the first evaluation; the other three metrics use the first evaluation.'} };
var arM='acc', arBuilt=false;
function drawArch(animate){
  var host=document.getElementById('ch-arch'), M=AR[arM];
  if(!arBuilt){
    D.archs.forEach(function(a){
      var row=el('div',{'class':'m'},host), left=el('div',null,row);
      el('div',{'class':'nm'},left,a.name); el('div',{'class':'kd'},left,a.kind); el('div',{'class':'kd'},left,a.method);
      var right=el('div',null,row);
      ['base','tuned'].forEach(function(tag){
        var br=el('div',{'class':'brow cv-'+tag},right); el('span',{'class':'lab'},br,tag==='base'?'Base':'Tuned');
        var trk=el('div',{'class':'trk'},br); el('div',{'class':'bar'},trk); el('span',{'class':'val'},br);
      });
    });
    arBuilt=true;
  }
  var mx=M.max||0;
  if(!M.max) D.archs.forEach(function(a){ mx=Math.max(mx,M.get(a.base,a.base_en),M.get(a.tuned,a.tuned_en)); });
  var rows=host.querySelectorAll('.brow');
  D.archs.forEach(function(a,i){
    ['base','tuned'].forEach(function(tag,j){
      var v=M.get(a[tag],a[tag+'_en']), br=rows[i*2+j];
      br.querySelector('.bar').style.width=(mx>0?Math.max(v/mx*100,v>0?0.5:0):0)+'%';
      br.querySelector('.val').textContent=fmt(v,M.d)+M.unit;
      br.setAttribute('aria-label',a.name+', '+tag+': '+fmt(v,M.d)+M.unit);
    });
  });
  document.getElementById('ar-note').textContent=M.note;
}
document.getElementById('ar-seg').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(!b) return;
  arM=b.getAttribute('data-m');
  this.querySelectorAll('button').forEach(function(x){ x.setAttribute('aria-pressed',x===b?'true':'false'); });
  drawArch();
});

/* ---------- 7. training cost ---------- */
(function(){
  var tb=document.querySelector('#tb-cost tbody'), mx=0;
  D.earlier.forEach(function(r){ if(r.gpuh>mx) mx=r.gpuh; });
  function row(method,model,gh,gpuh,gpuhText,acc,tok,scorer){
    var tr=el('tr',null,tb);
    el('td',{'class':'l wrap'},tr,method); el('td',{'class':'l'},tr,model); el('td',null,tr,gh);
    var td=el('td',{'class':'l'},tr); var b=el('span',{'class':'gbar'},td); b.style.width=Math.round(gpuh/mx*90)+'px'; td.appendChild(document.createTextNode(gpuhText));
    el('td',null,tr,acc); el('td',null,tr,tok);
    var s=el('td',{'class':'l'},tr); if(scorer==='eval.py'){ el('span',{'class':'flag'},s,'eval.py'); } else s.textContent=scorer;
  }
  D.earlier.forEach(function(r){
    row(r.method,r.model,r.gpus+' × '+(r.hours_approx?'~':'')+fmt(r.hours,2)+' h',r.gpuh,'≈ '+r.gpuh,
      fmt(r.acc0,2)+'% → '+fmt(r.acc1,2)+'%',fmt(r.tok0,0)+' → '+fmt(r.tok1,0),r.scorer);
  });
  var per=D.totals.train_gpu_h/D.sweep.length, a=D.coefs.map(function(c){return D.coef[c].acc.mean;}), t=D.coefs.map(function(c){return D.coef[c].tok.mean;});
  row('NeMo RL (GRPO), 60 steps, this sweep (mean of 12 runs)','Nemotron-3.5-Lightning-30B-A3B','16 × '+fmt(per/16,2)+' h',per,fmt(per,1),
    fmt(D.base.acc,2)+'% → '+fmt(Math.min.apply(null,a),2)+'–'+fmt(Math.max.apply(null,a),2)+'%',
    fmt(D.base.tok,0)+' → '+fmt(Math.min.apply(null,t),0)+'–'+fmt(Math.max.apply(null,t),0),'NeMo Gym');
})();

/* ---------- 8. sortable table ---------- */
(function(){
  var cols=[
    ['Coefficient',function(r){return +r.c;},function(r){return r.c;}],
    ['Seed',function(r){return r.seed;},function(r){return r.seed;}],
    ['Accuracy %',function(r){return r.sum.overall.acc;},2],
    ['Tokens',function(r){return r.sum.overall.tok;},0],
    ['Cut-off %',function(r){return r.sum.overall.trunc;},2],
    ['Run rate %',function(r){return r.sum.run_rate.acc;},2],
    ['Chase rate %',function(r){return r.sum.chase_rate.acc;},2],
    ['Milestone %',function(r){return r.sum.milestone.acc;},2],
    ['Answers/s',function(r){return r.en.aps;},1],
    ['GPU power W',function(r){return r.en.power;},0],
    ['mWh/correct',function(r){return r.en.mwh;},1]
  ];
  var head=document.querySelector('#tb-runs thead tr'), body=document.querySelector('#tb-runs tbody'), sortCol=-1, dir=1, ths=[];
  cols.forEach(function(c,i){
    var th=el('th',{scope:'col','aria-sort':'none'},head), b=el('button',{type:'button'},th);
    b.appendChild(document.createTextNode(c[0])); el('span',{'class':'ar','aria-hidden':'true'},b,'');
    b.addEventListener('click',function(){ if(sortCol===i) dir=-dir; else { sortCol=i; dir=i<2?1:-1; } render(); });
    ths.push(th);
  });
  function render(){
    var rows=D.sweep.slice();
    if(sortCol>=0) rows.sort(function(a,b){ var d=cols[sortCol][1](a)-cols[sortCol][1](b); return d?d*dir:(+a.c-+b.c)||(a.seed-b.seed); });
    ths.forEach(function(th,i){ th.setAttribute('aria-sort',i===sortCol?(dir>0?'ascending':'descending'):'none');
      th.querySelector('.ar').textContent=i===sortCol?(dir>0?'▲':'▼'):''; });
    body.textContent='';
    rows.forEach(function(r){
      var tr=el('tr',null,body);
      cols.forEach(function(c,i){
        var v=typeof c[2]==='function'?c[2](r):fmt(c[1](r),c[2]);
        var td=el('td',null,tr);
        if(i===0){ el('span',{'class':'chip cv-'+CK[r.c]},td); td.appendChild(document.createTextNode('c = '+v)); } else td.textContent=v;
      });
    });
  }
  render();
})();

/* ---------- draw + resize ---------- */
function drawAll(){ drawScatter(); drawType(); drawEnergy(); redraws.forEach(function(fn){ fn(); }); }
drawAll(); drawArch();
if('IntersectionObserver' in window && !RM){
  /* animate the architecture bars from zero when the section first scrolls into view */
  var archHost=$('ch-arch'), archBars=archHost.querySelectorAll('.bar');
  archBars.forEach(function(b){ b.style.transition='none'; b.style.width='0'; });
  var io=new IntersectionObserver(function(es){
    if(es[0].isIntersecting){ archBars.forEach(function(b){ b.style.transition=''; }); drawArch(); io.disconnect(); }
  },{threshold:0.25});
  io.observe(archHost);
}
var lastW=window.innerWidth, timer=null;
window.addEventListener('resize',function(){
  if(window.innerWidth===lastW) return; lastW=window.innerWidth;
  clearTimeout(timer); timer=setTimeout(drawAll,120);
});
})();
</script>
</body>
</html>
"""


def main():
    D = build_data()
    keys = tokens_for(D)
    payload = json.dumps(D, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")

    def sub(m):
        k = m.group(1)
        if k == "DATA":
            return payload
        if k not in keys:
            raise KeyError(f"template token @@{k}@@ has no value")
        return keys[k]

    html = re.sub(r"@@(\w+)@@", sub, TEMPLATE)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)

    print(f"wrote {OUT} ({len(html.encode('utf-8')):,} bytes)")
    print("coef  acc% (min-max)            tokens (min-max)      answers/s  power W  mWh/correct  train GPU-h/run")
    for c in COEFS:
        s = D["coef"][c]
        print(f"{c:>4}  {s['acc']['mean']:6.2f} ({s['acc']['min']:.2f}-{s['acc']['max']:.2f})   "
              f"{s['tok']['mean']:7.1f} ({s['tok']['min']:.0f}-{s['tok']['max']:.0f})   "
              f"{s['aps']['mean']:8.2f}  {s['power']['mean']:7.1f}  {s['mwh']['mean']:10.2f}  {s['train_gpuh']['mean']:8.1f}")
    b = D["base"]
    print(f"base  {b['acc']:6.2f}  tokens {b['tok']:.1f}  cut-off {b['trunc']:.2f}%  "
          f"answers/s {b['en']['aps']:.2f}  power {b['en']['power']:.1f} W  mWh/correct {b['en']['mwh']:.2f}")
    print("phase seconds per step, steps 2-60 (share of step) | steps 51-60")
    for c in COEFS:
        a, l = D["phase"][c]["all"], D["phase"][c]["late"]
        print(f"{c:>4}  total {a['total']:.1f}s  gen {a['gen']:.1f} ({a['gen']/a['total']*100:.0f}%)  refit {a['refit']:.1f} ({a['refit']/a['total']*100:.0f}%)  "
              f"logp {a['logp']:.1f}  train {a['train']:.1f} ({a['train']/a['total']*100:.0f}%)  misc {a['misc']:.1f} | "
              f"late total {l['total']:.1f}s  gen {l['gen']:.1f} ({l['gen']/l['total']*100:.0f}%)  refit {l['refit']:.1f} ({l['refit']/l['total']*100:.0f}%)  train {l['train']:.1f}")
    print("GPU telemetry per run, mean of seeds: util% (loaded) | mem GiB loaded | peak GiB (% of capacity) | W per GPU | kWh per run (min-max) | max temp | engine busy% early->late")
    for c in COEFS:
        g, e = D["gcoef"][c], D["eng"][c]
        print(f"{c:>4}  {g['util']:.1f} ({g['util_loaded']:.1f}) | {g['mem_loaded']/1024:.1f} | {g['peak']/1024:.1f} ({g['peak']/D['memMiB']*100:.1f}%) | "
              f"{g['power']:.0f} | {g['kwh']:.2f} ({g['kwh_min']:.2f}-{g['kwh_max']:.2f}) | {g['tmax']:.0f} C | {e['early']['busy']:.0f}->{e['late']['busy']:.0f}")
    t = D["totals"]
    print(f"training: {t['n_train_jobs']} jobs ({t['n_timeout']} TIMEOUT), {t['train_node_h']:.2f} node-h, {t['train_gpu_h']:.2f} GPU-h")
    print(f"evaluation: {t['n_eval_jobs']} jobs, {t['eval_node_h']:.2f} node-h = {t['eval_gpu_h']:.2f} GPU-h (1 GPU serving)")


if __name__ == "__main__":
    main()
