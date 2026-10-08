"""Collect training/bench/eval metrics for the dashboard: prints one JSON blob. Run on the login node with LOG_DIR, PROJ set."""
import json, re, glob, os, statistics, collections
L = os.environ["LOG_DIR"] + "/nemotron-cricket"
P = os.environ["PROJ"]
num = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
def parse_train(f):
    rows = []
    for line in open(f, errors="ignore"):
        m = re.search(r"\[(\S+ \S+)\] iteration +(\d+)/ +(\d+)", line)
        if not m: continue
        r = {"t": m.group(1), "step": int(m.group(2))}
        for k, key in [("elapsed time per iteration \(ms\)", "ms"), ("throughput per GPU \(TFLOP/s/GPU\)", "tflops"),
                       ("learning rate", "lr"), ("lm loss", "lm"), ("seq_load_balancing_loss", "lb"),
                       ("mtp_1 loss", "mtp1"), ("mtp_2 loss", "mtp2"), ("grad norm", "gn")]:
            mm = re.search(k + r": +" + num, line)
            if mm: r[key] = float(mm.group(1))
        rows.append(r)
    return rows
out = {}
tf = glob.glob(f"{L}/*6206759.out")[0]
out["train"] = parse_train(tf)
out["val"] = [{"step": int(a), "loss": float(b)} for a, b in re.findall(r"validation loss at iteration +(\d+).*?lm loss value: +" + num, open(tf, errors="ignore").read())]
out["bench"] = {}
for name, jid in [("A: TP2·EP8·selective", "6206161"), ("B: TP1·EP8·none", "6206659"), ("C: TP1·EP8·selective", "6206681")]:
    f = glob.glob(f"{L}/*{jid}.out")[0]; txt = open(f, errors="ignore").read()
    r = parse_train(f); mem = re.findall(r"mem-max-allocated-gigabytes: +" + num, txt)
    out["bench"][name] = {"steps": r, "oom": "out of memory" in txt, "mem_max_gb": max(map(float, mem)) if mem else None}
out["eval"] = {}
for run in ["base", "sft-full-e1"]:
    d = json.load(open(f"{P}/runs/{run}/eval/benchmark600.json"))
    toks = collections.defaultdict(list); hist = []
    for line in open(f"{P}/runs/{run}/eval/benchmark600_samples.jsonl"):
        s = json.loads(line); toks[s["qtype"]].append(s["tokens"]); hist.append(s["tokens"])
    d["tokens_by_type"] = {k: statistics.mean(v) for k, v in toks.items()}
    d["tokens_hist"] = hist
    d.pop("model", None)
    out["eval"][run] = d
ex = {}
for line in open(f"{P}/runs/sft-full-e1/eval/benchmark600_samples.jsonl"):
    s = json.loads(line)
    if s["correct"] and s["qtype"] not in ex and s["tokens"] < 700: ex[s["qtype"]] = s
out["examples"] = ex
out["prep"] = json.load(open(f"{P}/data/sft/prep_stats.json")); out["prep"].pop("snapshot", None)
print(json.dumps(out))
