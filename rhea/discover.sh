#!/bin/bash
# Run on the Rhea login node: what can we run, where, and on what hardware?  (no job needed, read-only)
echo "== whoami / host"; whoami; hostname; uname -m; echo "HOME=$HOME"
echo "== slurm accounts"; sacctmgr -nP show assoc user=$USER format=account,partition,qos,maxjobs,grptres 2>/dev/null | head -20
echo "== partitions (name, avail, timelimit, nodes, state, gres, cpus, mem)"
sinfo -o "%P|%a|%l|%D|%t|%G|%c|%m" | sort | head -60
echo "== idle/mix nodes by partition"; sinfo -t idle,mix -o "%P %D %N" | head -30
echo "== GPU type per partition (gres)"; sinfo -N -o "%P %N %G" | awk '{print $1,$3}' | sort | uniq -c | head -20
echo "== queue pressure"; squeue -h -t pending -o "%P" | sort | uniq -c; squeue --me
echo "== lustre mounts"; df -h 2>/dev/null | grep -i -E "lustre|lfs|fsw" 
ls -d /lustre/fsw/*/$USER 2>/dev/null
echo "== container tooling"; which enroot srun 2>/dev/null; srun --help 2>&1 | grep -c container-image
echo "== python"; which python3; python3 --version; which uv
echo "== quotas"; (quota -s 2>/dev/null | head -5) || true
