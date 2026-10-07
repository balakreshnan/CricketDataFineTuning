#!/bin/bash
# Per-node network sample for a running job: every RDMA port (InfiniBand and Ethernet/RoCE) and NVLink, over T seconds.
#   srun --jobid=<job> --overlap --nodes=1 --ntasks=1 bash net_sample.sh [T=20]
T=${1:-20}
ports() { for d in /sys/class/infiniband/*/ports/1; do
  n=$(basename "$(dirname "$(dirname "$d")")")
  echo "$n $(cat "$d/link_layer") $(cat "$d/counters/port_xmit_data") $(cat "$d/counters/port_rcv_data")"; done; }
nvl() { for g in 0 1 2 3; do nvidia-smi nvlink -gt d -i $g 2>/dev/null | awk -v g=$g '/Data Tx/{tx+=$(NF-1)} /Data Rx/{rx+=$(NF-1)} END{print g, tx, rx}'; done; }
p0=$(ports); v0=$(nvl); sleep "$T"; p1=$(ports); v1=$(nvl)
echo "== $(hostname -s): RDMA ports over ${T}s (counters are 4-byte words)"
paste <(echo "$p0") <(echo "$p1") | awk -v T="$T" '{tx=($7-$3)*4/T/1e9; rx=($8-$4)*4/T/1e9; printf "%-8s %-10s tx %7.3f GB/s  rx %7.3f GB/s\n", $1, $2, tx, rx}'
echo "== NVLink per GPU over ${T}s (all links)"
paste <(echo "$v0") <(echo "$v1") | awk -v T="$T" '{printf "GPU%s  tx %7.2f GB/s  rx %7.2f GB/s\n", $1, ($5-$2)*1024/T/1e9, ($6-$3)*1024/T/1e9}'
