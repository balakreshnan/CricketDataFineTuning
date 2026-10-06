#!/bin/bash
# Hardware topology of a batch-xdr node + statistics of the one-to-one vLLM distillation run (for cricket/README.md).
R=/lustre/fsw/general_sa/bbalakreshna/clustercodes
echo "### GPUs"; nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
echo "### GPU topology"; nvidia-smi topo -m 2>&1 | head -20
echo "### NVLink"; nvidia-smi nvlink --status -i 0 2>&1 | head -8; nvidia-smi -q -i 0 2>/dev/null | grep -iE -A4 "fabric|clique" | head -16
echo "### InfiniBand"; for d in /sys/class/infiniband/*; do n=$(basename $d); for p in $d/ports/*; do echo "$n port $(basename $p): $(cat $p/rate 2>/dev/null) state=$(cat $p/state 2>/dev/null) link=$(cat $p/link_layer 2>/dev/null)"; done; done | sort | head -20
echo "### CPU / memory"; lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"; free -g | head -2

echo "### vLLM shard throughput (final line per shard)"
for j in 697019 697021 697178 697022 700651; do
  d=$R/cricket/logs/general_sa-cricket.datagen-full-$j; [ -d $d ] || d=$R/cricket/logs/general_sa-cricket.retry-full-$j; [ -d $d ] || continue
  echo "== job $j ($(ls $d | wc -l) shards)"
  grep -h "generated tok/s" $d/shard*.log | awk '{print $NF" "$0}' | sed -E 's/.*\] ([0-9,]+)\/([0-9,]+) questions, ([0-9.]+) min, ([0-9,]+) generated.*/\1 \2 \3 \4/' | \
    awk '{gsub(",","",$1);gsub(",","",$2);gsub(",","",$4); last[NR]=$0} END{}' ; \
  for f in $d/shard*.log; do grep "generated tok/s" $f | tail -1; done | sed -E 's/.*\] ([0-9,]+)\/([0-9,]+) questions, ([0-9.]+) min, ([0-9,]+) generated.*/\1 \2 \3 \4/' | tr -d , | \
    awk '{q+=$1; m+=$3; t+=$4; n++; if($4>mx)mx=$4; if(mn==""||$4<mn)mn=$4} END{printf "shards %d, questions %d, mean minutes %.1f, tok/s per GPU mean %.0f min %.0f max %.0f\n", n, q, m/n, t/n, mn, mx}'
  grep -h "engine ready" $d/shard*.log | sed -E 's/.*ready in ([0-9]+)s/\1/' | awk '{s+=$1;n++} END{if(n) printf "engine startup mean %.0fs over %d shards\n", s/n, n}'
done

echo "### sample statistics"
source $R/venv/qwen38-ft/bin/activate 2>/dev/null || true
python3 - <<'EOF'
import json, glob, collections
G = "/lustre/fsw/general_sa/bbalakreshna/clustercodes/cricket/data/gen/train-full/samples"
tot = collections.Counter(); per = collections.defaultdict(collections.Counter); files = collections.Counter()
for f in sorted(glob.glob(G + "/*_rank*.jsonl")):
    tag = f.split("/")[-1].split("_rank")[0]
    for line in open(f, encoding="utf-8"):
        r = json.loads(line); files[tag] += 1
        for c in (tot, per[r["qtype"]]):
            c["samples"] += 1; c["tokens"] += r["n_tokens"]; c["correct"] += r["correct"]; c["finished"] += r["finished"]
print("samples by tag:", dict(files))
print("TOTAL", dict(tot), "| mean tokens %.0f" % (tot["tokens"] / tot["samples"]))
for qt, c in sorted(per.items()):
    print(f"{qt:12} samples {c['samples']:>9,}  tokens {c['tokens']:>13,}  acc {c['correct']/c['samples']:.4f}  finished {c['finished']/c['samples']:.4f}")
EOF
echo "### storage"; du -sh $R/cricket/data/gen/train-full/samples $R/cricket/data/gen/train-full/final $R/cricket/data/questions-full $R/cricket/cache 2>/dev/null
