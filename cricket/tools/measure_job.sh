#!/bin/bash
# Live GPU + InfiniBand measurement of a running multi-node job (run on the login node):
#   bash measure_job.sh <jobid> [seconds=60]
# Attaches to every node of the job (srun --overlap) and samples, over the window:
#   nvidia-smi: GPU utilization, memory used, power, SM clock (every 2 s, averaged per GPU)
#   InfiniBand: port_xmit_data / port_rcv_data deltas per XDR port -> GB/s per node
J=$1; T=${2:-60}
N=$(squeue -j "$J" -h -o %D)
[ -z "$N" ] && { echo "job $J not running"; exit 1; }
srun --jobid="$J" --overlap --nodes="$N" --ntasks="$N" --ntasks-per-node=1 bash -c '
T='"$T"'
h=$(hostname -s)
ib0=$(for d in /sys/class/infiniband/*/ports/1; do [ "$(cat $d/link_layer)" = InfiniBand ] && echo "$(cat $d/counters/port_xmit_data) $(cat $d/counters/port_rcv_data)"; done | awk "{t+=\$1; r+=\$2} END{print t, r}")
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader,nounits -l 2 > /tmp/gpu_$$.csv &
P=$!; sleep $T; kill $P 2>/dev/null
ib1=$(for d in /sys/class/infiniband/*/ports/1; do [ "$(cat $d/link_layer)" = InfiniBand ] && echo "$(cat $d/counters/port_xmit_data) $(cat $d/counters/port_rcv_data)"; done | awk "{t+=\$1; r+=\$2} END{print t, r}")
# IB data counters count 4-byte words
echo "$ib0 $ib1" | awk -v h=$h -v T=$T "{printf \"%s IB tx %.2f GB/s rx %.2f GB/s (8 XDR ports)\n\", h, (\$3-\$1)*4/T/1e9, (\$4-\$2)*4/T/1e9}"
awk -F", " -v h=$h "{u[\$1]+=\$2; m[\$1]+=\$3; p[\$1]+=\$4; c[\$1]+=\$5; n[\$1]++} END{for (g in u) printf \"%s GPU%s util %.0f%%  mem %.1f GiB  power %.0f W  sm %.0f MHz\n\", h, g, u[g]/n[g], m[g]/n[g]/1024, p[g]/n[g], c[g]/n[g]}" /tmp/gpu_$$.csv | sort
rm -f /tmp/gpu_$$.csv
' 2>&1 | grep -v "^srun"
