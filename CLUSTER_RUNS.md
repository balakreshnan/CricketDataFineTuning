# Cluster runs tracker

Last updated 2026-11-14. Rhea entries come from this week's job records. Hecate, EOS and Ptyche entries come from the
notes and READMEs written when those runs finished (October 2026) and were not re-checked on the clusters today.

## Clusters

| Cluster | GPUs per node | Login | Code folder | Slurm notes | Helper scripts |
|---|---|---|---|---|---|
| Hecate | 4x Vera Rubin (about 280 GiB) | `bbalakreshna-mfa@login-hecate.nvidia.com` | repo root: `bin/`, `finetune/`, `cricket/` | account `general_sa`; `batch-xdr`, `batch-spx` (5 h), `backfill-xdr` (8 h, preemptible) | `bin/hecate-bridge`, `bin/hx`, `bin/hsync` |
| EOS | 8x H100 (80 GB) | `login-eos02.eos.clusters.nvidia.com` | `eos/` | `batch` and `backfill` (4 h), `interactive` (2 h, up to 2 nodes) | `eos/bin/eos-bridge`, `ex`, `esync` |
| Ptyche | 4x GB200 (aarch64 Grace) | `bbalakreshna-mfa@login-ptyche` | `ptyche/` | `batch` (5 h), `tcpo` (5 h), `backfill` (8 h, preemptible) | `ptyche/bin/ptyche-bridge`, `px`, `psync` |
| Rhea | 4x Vera Rubin (286 GB, aarch64 Vera) | `bbalakreshna-mfa@login-rhea.nvidia.com` | `rhea/` | `batch` (5 h), `backfill` (8 h, preemptible) | `rhea/bin/rhea-bridge`, `rx`, `rhsync`, `set-secrets` |

On every cluster: account `general_sa`, whole-node allocation, large files on `/lustre/fsw/general_sa/bbalakreshna`.

## Projects and results

| # | Date | Cluster | Project | Model | Method | Compute | Result | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-05 | Hecate | FinNews tweets (sentiment + topic) | Qwen3.8-27B | LoRA, 6 runs from 1 to 10 epochs | 16 GPUs, 14 to 47 min per run | Overall accuracy 69.95% to 92.45% (3-epoch run, the one on HF); best 92.53% at 10 epochs | Done |
| 2 | 2026-10-05 | Hecate | Cricket reasoning, 4.5K-row dataset | Qwen3.8-27B | Distill reasoning traces, then LoRA (2 epochs) | SFT job 696652, 20 min | Accuracy flat (97.25% to 97.04%); reasoning 36% shorter (748 to 482 tokens) | Done |
| 3 | 2026-10-05/06 | Hecate | Cricket reasoning, full 500K-row dataset build | Qwen3.8-27B | vLLM distillation, about 2M samples | 8 nodes, about 15 node-hours | 500,000 verified rows (458,702 train, 41,298 test) | Done |
| 4 | 2026-10-07 | Hecate | Cricket LoRA on the 500K dataset | Qwen3.8-27B | LoRA r32, 1 epoch, lr 2e-4 | 32 GPUs, 4.0 h training in 2 allocations (10.1 h with queue) | Accuracy 97.29% to 97.88%; tokens 748 to 458 (-39%); truncation 0.38% to 0 | Done |
| 5 | 2026-10-08 | EOS | Cricket reasoning, full SFT | Nemotron-3.5-Lightning-30B-A3B | Full fine-tune (Megatron-Bridge), 1 epoch | 2 nodes (16 H100), 1 h 37 min | Accuracy 89.6% to 97.0%; tokens 1,700 to 519; truncation 4.9% to 0 | Done |
| 6 | 2026-10-08/09 | Ptyche | Olympics reasoning dataset | Nemotron-3.5-Lightning-30B-A3B | Dynamo + SGLang distillation, 1.08M samples | GB200 nodes | 269,723 of 269,731 source rows covered | Done |
| 7 | 2026-10-09 | Ptyche | Olympics full SFT | Nemotron-3.5-Lightning-30B-A3B | Full fine-tune, 789 steps | 4 nodes (16 GB200), 52 min | Row benchmark 92.00% to 92.35%; table benchmark 93.98% to 93.05%; tokens 2,209 to 1,951 | Done |
| 8 | 2026-11-11 | Rhea | Cricket LoRA on the 500K dataset | Qwen3.8-27B | LoRA r32, 1 epoch, lr 1e-4 | 32 GPUs, 4.5 h training in 1 allocation (4.7 h total) | Accuracy 97.00% to 97.92%; tokens 734 to 455 (-38%); truncation 0.83% to 0 | Done |
| 9 | 2026-11-12 | Rhea | Speculative decoding with Dynamo + SGLang | Qwen3.8-27B | Probe of public Dynamo SGLang 1.5.1 image | 1 node, 5 short jobs | No server started: image has no GPU kernels for Vera Rubin. No speed numbers | Stopped, cleaned up |
| 10 | 2026-11-13 | Rhea | Cricket RLVR (reward = exact answer, short-answer bonus) | Qwen3.8-27B | LoRA r32 from base, group-baseline policy gradient, 100 steps, 102,400 rollouts | 32 GPUs, 1.33 h training (jobs 39000, 39066, 39067) | Accuracy 97.00% to 99.17%; tokens 734 to 109 (-85%); truncation 0.83% to 0 | Done |
| 11 | 2026-11-13 | Rhea | Cricket RLVR on a mixture-of-experts model | Qwen3.6-35B-A3B | LoRA r32 (attention + shared-expert MLP, 38.3M params), 60 steps, 61,440 rollouts, 4,096-token cap | 32 GPUs, 2.65 h training (jobs 40421, 47560, 47561; first benchmark 40423 ran out of GPU memory) | Accuracy 93.62% to 98.96%; tokens 2,177 to 151 (-93%); truncation 3.25% to 0; milestone 99.00% to 97.88% | Done |
| 12 | 2026-11-14 | Rhea | Cricket RLVR rerun with GPU usage logged on all 32 GPUs | Qwen3.8-27B | Same recipe as row 10 but 120 steps (script default), 122,880 rollouts; run `cricket-rlvr-1114` | 32 GPUs, 1.76 h training (jobs 47941, 47943, 47944); avg utilisation 73% (65-78% per GPU), peak memory 260 of 286.5 GB | Accuracy 97.00% to 99.96%; tokens 734 to 119 (-84%); truncation 0.83% to 0; adapter in private HF repo `Balab2021/Qwen3.8-27B-Cricket-Reasoning-RLVR-120steps` (job 48773) | Done |
| 13 | 2026-11-23 | Rhea | Speculative decoding benchmark (`rhea/specdec/`), run `specdec-qwen38-rlvr-1123` | Target Qwen3.8-27B + RLVR adapter; drafts Qwen3.5-0.8B/2B/4B/9B (base) | Own decoding loop with cache snapshots (transformers refuses assisted generation for these stateful models); cuDNN attention kernel off; 200 questions, 2/4/8 draft tokens per round | 8 GPUs, 40 min (job 120765); set-up jobs 120135-120720 | Slower than target alone: 36 tok/s alone vs 17-25 tok/s with a draft (best 0.70x, Qwen3.5-2B, 2 tokens per round); acceptance 17-57%; accuracy 100% everywhere | Done |

## Rhea job log

| Job | Date | What | Nodes | Elapsed | Outcome |
|---|---|---|---|---|---|
| 23541 | 11-11 | Hardware probe | 1 | 0:37 | Completed |
| 23547 | 11-11 | Setup: container, venv, model and dataset download | 1 | 3:49 | Completed |
| 23579 | 11-11 | Smoke test | 1 | 1:26 | Failed: `warmup_ratio` removed in transformers 5.22 |
| 23600 | 11-11 | Smoke test | 1 | 1:26 | Failed: Python 3.14 could not pickle the collate function |
| 23603 | 11-11 | Smoke test, 20 steps | 1 | 5:34 | Completed |
| 23649 | 11-11 | Full LoRA training, 1,788 steps | 8 | 4:37:41 | Completed, dev loss 0.3126 |
| 23699 | 11-11 | Resume job | 8 | 0:53 | Completed, nothing left to do |
| 25544 | 11-11 | Upload adapter to Hugging Face | 1 | 1:03 | Completed |
| 25585 | 11-11 | Benchmark (base model half) | 4 | 41:25 | Marked failed on a sync timeout; all 2,400 base samples saved |
| 25847 | 11-11 | Benchmark (fine-tuned half) | 4 | 24:48 | Completed |
| 31925 | 11-12 | Speculative decoding probe | 1 | 0:38 | Failed: nvcr.io rejected the stored NGC key (since renewed) |
| 31926 | 11-12 | Speculative decoding probe | 1 | about 4 min | Servers failed: Triton cannot target `sm_107a` |
| 31955 | 11-12 | Probe with Triton patch (`sm_100a`) | 1 | about 2 min | Servers failed: assembler rejected `.target sm_107a` |
| 31982 | 11-12 | Probe with patch, second version | 1 | 2:20 | Servers failed: GPU would not load `sm_100a` kernels |
| 32031 | 11-12 | Probe with patch (`sm_100`) | 1 | 2:06 | Servers failed: prebuilt `GemmaRMSNorm` kernel has no Rubin build |

## Hugging Face repos (under `Balab2021`)

| Repo | Type | From | Visibility per notes |
|---|---|---|---|
| `Qwen3.8-27B-FinNews-LoRA` | Adapter | Project 1 | Private |
| `CricketData-T20-Reasoning-Qwen3.8` | Dataset | Project 2 | Public |
| `Qwen3.8-27B-Cricket-Reasoning-LoRA` | Adapter | Project 2 | Public |
| `CricketData-T20-Reasoning-Qwen3.8-Full` | Dataset | Project 3 | Created private |
| `Qwen3.8-27B-Cricket-Reasoning-Full-LoRA` | Adapter | Project 4 | Public per `cricket/README.md` |
| `Nemotron-3.5-Lightning-30B-A3B-Cricket-Reasoning` | Full model | Project 5 | Public |
| `Olympics-Reasoning-Nemotron-3.5-Lightning` | Dataset | Project 6 | Public |
| `Nemotron-3.5-Lightning-30B-A3B-Olympics-Reasoning` | Full model | Project 7 | Private |
| `Qwen3.8-27B-Cricket-Reasoning-Rhea-LoRA` | Adapter | Project 8 | Private |
| `Qwen3.8-27B-Cricket-Reasoning-RLVR` | Adapter | Project 10 | Private |
| `Qwen3.6-35B-A3B-Cricket-Reasoning-RLVR` | Adapter | Project 11 | Private |
| `Qwen3.8-27B-Cricket-Reasoning-RLVR-120steps` | Adapter | Project 12 | Private |
| `Cricket-SpecDec-Rhea` | Dataset (benchmark results) | Project 13 | Private |

## Reports and runbooks

| What | Where |
|---|---|
| Rhea run report (interactive) | `rhea/cricket-lora-report.html` |
| Hecate vs Rhea comparison (interactive) | `rhea/hecate-vs-rhea-report.html` |
| Rhea RLVR run and GPU usage report (interactive) | `rhea/cricket-rlvr-report.html` |
| Rhea Qwen3.6 RLVR before/after and 32-GPU usage report (interactive) | `rhea/qwen36-rlvr-report.html` |
| Both RLVR runs end to end (interactive) | `rhea/rlvr-two-runs-report.html` |
| Rhea Qwen3.8 RLVR rerun, all 32 GPUs measured (interactive) | `rhea/qwen38-rlvr-allgpu-report.html` |
| Dense vs mixture-of-experts RLVR comparison: architectures, benchmark, minute-by-minute GPU usage (interactive) | `rhea/rlvr-moe-vs-dense-report.html` |
| Speculative decoding benchmark: drafts vs target alone, time split, per-question spread, GPU minutes (interactive) | `rhea/specdec-report.html` |
| Rhea runbook | `rhea/README.md` |
| Hecate cricket runbook | `cricket/README.md` |
| FinNews experiment table | `pulled/experiments_summary.md` |
| EOS runbook | `eos/nemotron-cricket/README.md` |
| Ptyche runbook | `ptyche/olympics/README.md` |

## Still on Rhea's Lustre

| Item | Size |
|---|---|
| `containers/pytorch-main-py3-devel-arm64.sqsh` | 15 GB |
| `models/Qwen3.8-27B` | 52 GB |
| `datasets/CricketData-T20-Reasoning-Qwen3.8-Full` | 2.1 GB |
| `cricket-rhea/runs/cricket-lora-full-1111` (adapter, 3 checkpoints, benchmark records) | not measured |
| `cricket-rhea/runs/cricket-rlvr-1113` (RLVR adapter 870 MB, checkpoints, benchmark records) | not measured |
| `cricket-rhea/runs/cricket-rlvr-1114` (rerun adapter, checkpoints, benchmark records, GPU logs) | not measured |
| `cricket-rhea/runs/qwen36-rlvr-1113` (RLVR adapter 153 MB, checkpoints, benchmark records, GPU logs) and `models/Qwen3.6-35B-A3B` (67 GB) | not measured |
| `cache/enroot-cache` (PyTorch image layers) | 20 GB |
