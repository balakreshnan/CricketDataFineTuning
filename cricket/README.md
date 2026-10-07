# Cricket T20I reasoning: data distillation, fine-tuning and evaluation on Hecate

End-to-end record of the pipeline that turns the ball-by-ball cricket dataset
[`Valarmathy/CricketData`](https://huggingface.co/datasets/Valarmathy/CricketData) into a **verified reasoning
dataset** distilled from `Qwen/Qwen3.8-27B` (thinking mode), fine-tunes a **LoRA reasoning adapter** on it, and
evaluates base vs fine-tuned on held-out matches. Everything runs as SLURM batch jobs on the Hecate (Vera Rubin)
cluster. This document is the runbook: it lists the compute, images, software versions, configuration, file-system
layout and every file involved, so the run can be repeated or modified.

Reference run: **2026-10-05**, SFT job `696652`, user `bbalakreshna`, account `general_sa`, partition `batch-xdr`.

The source data itself (columns, matches, years, teams, quirks, validation and smoke tests) is documented in
[`DATASET.md`](DATASET.md); `tools/validate_source.py` re-runs those checks.

**Part I (§1–§11)** documents the reference pipeline: 1,500 sampled source rows → 4,500 reasoning rows → LoRA on 16 GPUs.
**Part II (§12–§14)** documents the scale-up: **every one of the 425,119 source rows** → **500,000 reasoning rows**
distilled with **vLLM on 32 Rubin GPUs** (1.58 B generated tokens), and a **1-epoch fine-tune on 32 GPUs** within the
5-hour partition limit, including hardware, memory, network and GPU-efficiency measurements.

## Published artifacts (public on Hugging Face)

| artifact | link | contents | revision at publication |
|---|---|---|---|
| **Reasoning dataset** | [Balab2021/CricketData-T20-Reasoning-Qwen3.8](https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8) | `data/train.jsonl` (4,500 verified reasoning rows, 3 per source delivery), `data/test.jsonl` (600 held-out questions with gold answers), `stats.json`, dataset card; license CC0-1.0 (as the source data) | `efa295283fc9b2a012b9fa06a77bf393e35de212` |
| **Full one-to-one reasoning dataset** (Part II) | [Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full](https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full) | 500,000 verified rows covering all 425,119 source rows: `data/train-0000{0..4}.jsonl` (458,702), `data/test-00000.jsonl` (41,298, held-out matches), `data/eval-00000.jsonl` (the fixed 600-question benchmark), `stats.json`, card; 2.08 GB | `ad669766ef046c8ccc5330d077c77118b692ec00` |
| **Fine-tuned reasoning LoRA, full dataset** (Part II) | [Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA](https://huggingface.co/Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA) | pushed automatically by the `train-full` chain when training and evals finish (§13) | - |
| **Fine-tuned reasoning LoRA** | [Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA](https://huggingface.co/Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA) | `adapter_model.safetensors` + `adapter_config.json` (LoRA r32/α64 on `Qwen/Qwen3.8-27B`), tokenizer files and chat template, model card with the results, `eval_report/` (report.md, base/ft eval summaries, comparison.json, training metrics, plots) | `40af7e4f6122ac69372d7fa36a0a00f0202ad761` |

Both repos were created private by the pipeline and made **public** by the owner after review (verified 2026-10-05:
`private: false`, not gated). Re-running `push-dataset` / `train` uploads new commits to the same repos without
changing their visibility (`create_repo(..., private=True, exist_ok=True)` leaves existing repos as they are); to
keep a reference result stable, push experiments to a branch (`--hub_revision`, §8) or pin the revisions above.

Source data: [Valarmathy/CricketData](https://huggingface.co/datasets/Valarmathy/CricketData) (CC0-1.0).
Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Tracking: W&B project [`balabala76/qwen38-cricket-reasoning`](https://wandb.ai/balabala76/qwen38-cricket-reasoning).

### Using them

```python
from datasets import load_dataset
ds = load_dataset("Balab2021/CricketData-T20-Reasoning-Qwen3.8",
                  revision="efa295283fc9b2a012b9fa06a77bf393e35de212")   # splits: train, test
print(ds["train"][0]["messages"])          # system, user, assistant ("<think>...</think>\n\n... Final answer: N")
```

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA",
                                  revision="40af7e4f6122ac69372d7fa36a0a00f0202ad761")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
# prompt with tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)
# and sample with temperature 1.0, top_p 0.95, top_k 20 (as in training/eval); the answer is the
# 'Final answer: <number>' line after </think>.
```

For fast inference on Rubin, merge the adapter (`merge.py`) and serve the merged directory with vLLM from the
`dl/dgx/vllm:rubin-py3-devel` image, as the eval stage does (§5).

---

## 1. Results (reference run)

Test set: 600 questions from 177 matches never used for training; each answered 4 times in thinking mode with
identical vLLM settings for both models; answers checked against exact values.

| question type | accuracy base → fine-tuned | mean output tokens | p95 tokens | hit 6,144-token cap |
|---|---|---|---|---|
| run_rate | 98.50% → 99.00% | 404 → 280 (−31%) | 679 → 421 | 0.1% → 0% |
| chase_rate | 93.87% → 92.25% | 929 → 610 (−34%) | 1,920 → 1,096 | 0.4% → 0% |
| milestone | 99.38% → 99.88% | 910 → 557 (−39%) | 2,489 → 1,046 | 0.5% → 0.1% |
| **overall** | **97.25% → 97.04%** | **748 → 482 (−36%)** | | **0.3% → 0%** |

pass@4 = 100% for both models on every type.

- Accuracy is unchanged within noise (36 questions became more reliable, 39 less). The chase_rate dip is
  −1.3 standard errors (not significant); every wrong answer of either model is at a mid-over position (e.g. 9.2
  overs) where overs notation must be converted to balls.
- The fine-tuned model reasons **~36% shorter** at the same accuracy and no longer runs away to the token cap -
  the effect of training on the *shortest* verified trace out of 4 samples.
- The base model already scores ~97% on these questions, so there is little headroom for accuracy; harder question
  types are the lever for accuracy gains (see §11).

Dataset generation statistics: 18,000 samples on the 4,500 training questions, 97.2% correct per sample,
pass@4 100%, kept-trace mean 452 tokens; 1,500/1,500 source rows produced all 3 rows.

---

## 2. Pipeline overview

```
Valarmathy/CricketData (425,119 ball-by-ball rows, 1,842 T20Is)
   │  prepare.py            drop 68 rain-revised-target matches; split by match (train 1,597 / test 177)
   ▼                        sample 1,500 train + 200 test deliveries -> 3 questions each, exact answers
questions/{train,test}.jsonl   (4,500 / 600 questions)
   │  gen_vllm.py x16 GPUs  Qwen3.8-27B thinking mode, k=4 samples/question, verify vs exact answer
   ▼  finish_gen.py         keep the shortest verified-correct trace per question; W&B datagen run
gen/train/final/{train,test}.jsonl, stats.json, samples.md      ── human review & approval ──►
   │  push_dataset.py       HF dataset repo (created private, only after approval; now public)
   │  train.py (4 nodes)    LoRA SFT on prompt + the model's own verified output (loss on output only)
   ▼  merge.py              fold LoRA into a full bf16 checkpoint for vLLM
checkpoints/<run>/{adapter, merged/}
   │  gen_vllm.py           eval-base (base model) and eval-ft (merged model) on the 600 test questions, k=4
   ▼  finish_eval.py        metrics, report, plots, W&B (same run as SFT), push adapter + card + eval report
runs/<run>/{report.md, comparison.json, eval/*, plots/*}
```

### Question design (3 per source delivery, answers computed from the row)

| qtype | question | exact answer |
|---|---|---|
| `run_rate` | current run rate; overs given in cricket notation (14.3 = 14 overs 3 balls) | `round(runs*6/balls_bowled, 2)` |
| `chase_rate` | innings 2: run rate needed over the rest of the 20 overs; innings 1 (full 20-over innings only): rate scored over the rest of the innings, given the final total | `round(runs_needed*6/balls_left, 2)` |
| `milestone` | balls the striker must face to reach the next multiple of 50 at their own current strike rate | `ceil(needed_runs * balls_faced / runs)` |

Eligible deliveries: legal ball, 12–108 balls bowled, team runs ≥ 1, batter runs ≥ 1 and ≥ 4 balls faced; innings-1
rows only from innings that lasted the full 20 overs; innings-2 rows only while runs are still needed. Answers are
graded with tolerance 0.011 for rates (2-dp rounding) and exactly for balls; the model must end with
`Final answer: <number>` after its `</think>`.

The prompt (system + user) is in `cricket_qa.py` (`SYSTEM_PROMPT`, `make_questions`). The milestone wording names the
batter explicitly ("…keeps scoring at their own current strike rate (their runs per ball faced so far)…"); an earlier
ambiguous wording ("their current strike rate") made the model deliberate batter-vs-team until it hit the token cap
(9% correct, 90% truncated) - see §10.

---

## 3. Compute used

### Hardware (per node, `batch-xdr`)

| item | value |
|---|---|
| GPUs | 4 × NVIDIA Rubin ("NVIDIA Graphics Device"), compute capability **10.7 (sm_107)**, **286,524 MiB** HBM each |
| GPU driver | 620.43 |
| CPU | 2 × NVIDIA Vera ("Olympus"), **aarch64**, 352 cores per node |
| RAM | 1.35 TB (SLURM RealMemory 1,350,000 MB) |
| OS / kernel | Linux 7.0.0-2015-nvidia-bos-64k (host); Ubuntu 24.04.5 in both containers |
| Interconnect | XDR InfiniBand (partition `batch-xdr`); GPUs are not SLURM GRES - nodes are allocated whole (`--exclusive`) |
| Storage | Lustre `/lustre/fsw` (project space `/lustre/fsw/general_sa/bbalakreshna`) |

### Jobs of the reference run

| stage | job | nodes × GPUs | wall time | node-h | notes |
|---|---|---|---|---|---|
| prepare | 696500 | 1 × 4 | 0:34 | 0.009 | CPU work; downloads CSV, builds questions |
| import vLLM image (one-time) | 696529 | 1 × 4 | 1:51 | 0.031 | `enroot import` + Triton smoke test |
| pilot (optional) | 696531 | 1 × 4 | 4:24 | 0.073 | 96 questions × 4 samples |
| **datagen** | **696561** | **4 × 16** | **4:48** | **0.320** | 18,000 samples, ~5,300–6,600 generated tok/s per GPU |
| push dataset | 696651 | 1 × 4 | 0:39 | 0.011 | after approval |
| **SFT** | **696652** | **4 × 16** | **21:42** | **1.447** | 20.2 min of training (138 steps) + model load |
| merge | 696653 | 1 × 4 | 1:20 | 0.022 | 30 s merge + save 51 GB |
| eval-base | 696654 | 1 × 4 | 3:23 | 0.056 | 2,400 samples, ~6,000 tok/s per GPU |
| eval-ft + finish | 696655 | 1 × 4 | 4:31 | 0.075 | 2,400 samples (~7,100 tok/s/GPU) + report + push |
| **total, repeatable pipeline** | | | | **≈ 2.0 node-h ≈ 8 GPU-h** | ~45 min wall clock if the queue is free |

Development overhead in the reference session (not needed to repeat): first prepare 696355 (0:37), HF-`generate()`
pilot 696356 (43:08, too slow, then NCCL timeout), NGC vLLM import 696499 (2:05), two failed vLLM pilots 696501/696514
(missing `__main__` guard; NGC image cannot target sm_107), Triton smoke test 696518 (1:36), env report 696799 -
about 0.9 node-h (~3.5 GPU-h).

---

## 4. Container images and software

Both images are squashfs files on Lustre, used through pyxis/enroot (`--container-image`, `--container-mounts`
of Lustre, `--container-mount-home`, `--no-container-remap-root`, `--container-workdir=/home/bbalakreshna`).

| | PyTorch image (prepare, SFT, merge, finish, push) | vLLM image (generation, evals) |
|---|---|---|
| source | `gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel` | `gitlab-master.nvidia.com/dl/dgx/vllm:rubin-py3-devel` |
| squashfs | `clustercodes/containers/pytorch-main-py3-devel.sqsh` (15 GB) | `clustercodes/containers/vllm-rubin-py3-devel.sqsh` (11 GB) |
| Python | 3.12.3 | 3.12.3 |
| CUDA (nvcc) | 13.4 (V13.4.92) | 13.5 (V13.5.25) |
| torch | 2.15.0a0+875d815502.nvinternal.main | 2.15.0a0+b2c75dd062.nvinternal.rubin |
| triton | 3.8.0 | 3.6.0 (Rubin-capable build) |
| NCCL | 2.32.3 | - (one engine per GPU, no NCCL) |
| key packages | from venv `clustercodes/venv/qwen38-ft` (`--system-site-packages` on top of the image's torch): transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, wandb 0.30.0, flash-linear-attention (fla) 0.5.2, huggingface_hub 1.33.0, matplotlib | vllm 0.30.0+d666afb4.dev, transformers 5.12.1 (in the image; nothing installed) |

- The venv is built by `finetune/setup_env.sh` (FinNews project): `pip install -r finetune/requirements.txt` with
  the image's `torch/triton/numpy` pinned via a constraints file so pip cannot replace the NGC builds.
- `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` is exported in the PyTorch jobs: Triton's bundled ptxas rejects sm_107.
- **Do not use `nvcr.io/nvidia/vllm:26.08-py3`** on Rubin: its Triton fails with
  `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32` (verified by `tools/triton_smoke.py`).

Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (architecture `qwen3_5`,
`Qwen3_5ForConditionalGeneration`: 64 layers, 3 of every 4 Gated-DeltaNet linear attention, vision tower unused),
loaded from the shared HF cache `/lustre/fsw/general_sa/bbalakreshna/hf_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/<rev>` (52 GB).

---

## 5. Configuration

All cluster/path settings live in `cricket/config.env` (sourced by `submit.sh`, exported into every job). Script
flags can be overridden with `EXTRA_ARGS="..."` and sbatch flags appended after the stage name.

### Data preparation (`prepare.py`)

| parameter | value |
|---|---|
| source file | `Valarmathy/CricketData` → `raw/ball_by_ball_it20.csv` (69 MiB, CC0) |
| matches dropped | 68 (target ≠ first-innings total + 1, i.e. rain-revised) → 1,774 kept |
| split | by match, seed 0, 10% test → 1,597 train / 177 test matches |
| source rows | `--n_train 1500`, `--n_test 200` (round-robin, at most one row per match per round) |
| questions | 3 per row → 4,500 train / 600 test (1,500 / 200 per type) |

### Distillation (`gen_vllm.py`, stage `generate`)

| parameter | value |
|---|---|
| engine | vLLM offline `LLM.generate`, `tensor_parallel_size=1`, one engine per GPU (16 shards on 4 nodes) |
| `dtype` / `max_model_len` / `gpu_memory_utilization` | bfloat16 / 8192 / 0.90, prefix caching on, `limit_mm_per_prompt={image:0, video:0}` |
| chat template | `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` |
| sampling | **k = 4**, temperature 1.0, top_p 0.95, top_k 20 (Qwen3.8 model card, thinking mode), `max_tokens 6144`, seed = shard index |
| selection | per question: finished (`finish_reason == stop`) **and** correct; keep the **shortest** such trace |
| chunking / resume | 256 questions per `generate()` call, appended to `samples/train_rank{shard}.jsonl`; already-sampled ids are skipped on resubmit; work stops at the job's deadline (100 min) |

### Fine-tuning (`train.py`, stage `train`)

| parameter | value |
|---|---|
| method | LoRA SFT, plain DDP (full bf16 replica per GPU, ~51 GB), `torch.distributed.run` 4 nodes × 4 GPUs |
| data | 4,500 rows → 4,410 train / 90 dev (2%, seed 42); 0 dropped over `max_len 6144`; mean sequence 674 tokens |
| input / target | prompt = exact thinking-mode prompt used at generation; target = the model's own verified output + `<|im_end|>`; loss on target only |
| LoRA | r 32, alpha 64, dropout 0.05; targets (language model only): `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `mlp.{gate,up,down}_proj` → 217,579,520 trainable params (0.79%) |
| optimisation | 2 epochs, lr 1e-4 cosine, warmup 5% of steps, AdamW (HF default), bf16, gradient checkpointing (non-reentrant) |
| batch | per-GPU 2 × grad-accum 2 × 16 GPUs = **global 64** → 138 steps |
| logging / saving | log every 2 steps; dev loss + checkpoint every 25 steps, keep 3; deadline stop at 90 min |
| outcome | 20.2 min, final train loss 0.3229, dev loss 0.3145 (step 25) → 0.3114 (step 138) |

### Merge (`merge.py`) and evaluation (`gen_vllm.py --tag base|ft`, `finish_eval.py`)

| parameter | value |
|---|---|
| merge | load base bf16 on 1 GPU, `PeftModel.from_pretrained(...).merge_and_unload()`, `save_pretrained(max_shard_size=5GB)`, copy tokenizer/processor/chat-template files → `checkpoints/<run>/merged` (51 GB) |
| eval set | `gen/train/final/test.jsonl`: 600 questions, 177 unseen matches |
| eval sampling | identical to distillation: k = 4, temperature 1.0, top_p 0.95, top_k 20, max_tokens 6144, same seeds for base and fine-tuned |
| metrics | accuracy (mean over samples), pass@4, mean / correct-only output tokens, truncated rate, no-answer rate; per question type and overall; per-question improved/regressed counts |

### Tracking and publishing

| | value |
|---|---|
| W&B project | `qwen38-cricket-reasoning` (entity `balabala76`) - datagen runs (`datagen-<dir>-<job>`) and one run per SFT named after the run dir (training curves, then eval summary, base-vs-ft table and plots appended by `finish_eval.py`) |
| reference W&B runs | datagen: `https://wandb.ai/balabala76/qwen38-cricket-reasoning/runs/g89he13w`; SFT+eval: run id `general_sa-cricket.sft-696652` |
| HF dataset (**public**) | [`Balab2021/CricketData-T20-Reasoning-Qwen3.8`](https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8) (`data/train.jsonl`, `data/test.jsonl`, `stats.json`, card) |
| HF model (**public**) | [`Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA`](https://huggingface.co/Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA) (adapter, model card with results, `eval_report/`) |
| credentials | `SECRETS_FILE` = `/lustre/fsw/general_sa/bbalakreshna/qwen38-gsm8k/secrets.env` (`HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`), sourced inside the jobs |

---

## 6. File system layout

Rule: jobs are launched from `/home/bbalakreshna`; everything they read or write lives on Lustre.

```
/lustre/fsw/general_sa/bbalakreshna/                       ($LUSTRE_DIR)
├── hf_cache/hub/models--Qwen--Qwen3.8-27B/...             base weights (52 GB, shared with other projects)
├── qwen38-gsm8k/secrets.env                               HF / W&B credentials (mode 600)
└── clustercodes/
    ├── code/                                              this repository, synced by bin/hsync
    │   └── cricket/                                       pipeline code (§7)
    ├── containers/
    │   ├── pytorch-main-py3-devel.sqsh                    15 GB
    │   └── vllm-rubin-py3-devel.sqsh                      11 GB
    ├── venv/qwen38-ft/                                    Python env on top of the PyTorch image
    ├── cache/{pip,triton,...}                             caches (Triton kernels persist across jobs)
    └── cricket/                                           ($CROOT) - all cricket outputs
        ├── data/
        │   ├── raw/raw/ball_by_ball_it20.csv              source CSV (70 MB)
        │   ├── questions/{train,test}.jsonl, stats.json   built questions (5.6 MB)
        │   └── gen/
        │       ├── train/                                 the dataset run (64 MB)
        │       │   ├── samples/train_rank{0..15}.jsonl    all 18,000 verified samples (raw)
        │       │   └── final/{train,test}.jsonl, stats.json, samples.md
        │       └── pilot-<MMDDHHMM>/                      pilot runs (same structure)
        ├── runs/general_sa-cricket.sft-<job>/             (10 MB)
        │   ├── config.json, train_result.json, metrics.jsonl
        │   ├── eval/{base,ft}_rank{0..3}.jsonl, {base,ft}_summary.json
        │   ├── report.md, comparison.json, plots/*.png, wandb/
        └── checkpoints/general_sa-cricket.sft-<job>/      (60 GB)
            ├── adapter_model.safetensors, adapter_config.json, README.md, tokenizer files   (final adapter)
            ├── checkpoint-{100,125,138}/                  2.5 GB each (adapter + optimizer state)
            └── merged/                                    full merged bf16 model for vLLM (51 GB)
        └── logs/
            ├── general_sa-cricket.<stage>-<job>.log       one log per job
            └── general_sa-cricket.<stage>-<job>/shard*.log   per-GPU vLLM worker logs
```

Local machine (Windows, `C:\Code\ClaudeCode\clustercodes`): `cricket/` (code + this README), `bin/` (cluster access
helpers), `pulled/cricket/...` (copies fetched with `bin/hsync pull`).

---

## 7. Files involved

### Code (`cricket/`)

| file | runs in | purpose |
|---|---|---|
| `config.env` | login node | account, partition, Lustre paths, images, model revision, secrets file, W&B project, HF repo ids |
| `submit.sh` | login node | entry point: stages `import-vllm`, `prepare`, `pilot`, `generate`, `push-dataset`, `train` (train chains SFT → merge → eval-base ∥ → eval-ft) |
| `job.sbatch` | SLURM | generic PyTorch-container job: `LAUNCH=python` (single process on node 0) or `LAUNCH=torchrun` (one process per GPU, `python -m torch.distributed.run`, c10d rendezvous) |
| `vllm.sbatch` | SLURM | vLLM-container job: one `gen_vllm.py` per GPU (shard = node × GPUs + local GPU), per-GPU logs, then optional `FINISH_SCRIPT` in the PyTorch container |
| `cricket_qa.py` | both | question builder (`build_splits`, `make_questions`, eligibility, DLS filter), prompts, answer parsing (`parse_answer`, `split_reasoning`) and grading (`is_correct`); stdlib only |
| `prepare.py` | PyTorch | downloads the CSV via `hf_hub_download`, writes `questions/` |
| `gen_vllm.py` | vLLM | sampling worker for distillation and evals; resumable; behind `if __name__ == "__main__"` (vLLM spawns a child that re-imports the file) |
| `finish_gen.py` | PyTorch | merges shard files, metrics, builds `final/` once nothing is pending, W&B datagen run |
| `generate.py` | PyTorch | `build_dataset()` (shortest correct trace per question, `stats.json`, `samples.md`); also a legacy HF-`generate()` torchrun path (slow on Rubin, kept as a fallback) |
| `sampling.py` | PyTorch | shared helpers: thinking-mode prompt, `collect()`, `metrics()`; plus the HF-`generate()` sampler used by the legacy path |
| `push_dataset.py` | PyTorch | strips internal fields, writes the dataset card, uploads to the HF dataset repo (created private if missing); refuses without `--confirm` |
| `train.py` | PyTorch (torchrun) | LoRA SFT (`--no_eval` in the pipeline), W&B training run; `finalize()` (report, W&B tables/plots, hub push) reused by `finish_eval.py` |
| `merge.py` | PyTorch | merges the adapter into a full checkpoint for vLLM |
| `finish_eval.py` | PyTorch | eval metrics for base and ft, `report.md`, W&B (resumes the SFT run), pushes adapter + model card + `eval_report/` |
| `report.py` | PyTorch / laptop | base-vs-ft report, `comparison.json`, plots (accuracy and tokens by type, training curves) |
| `tools/triton_smoke.py` | any image | checks that a container's Triton can compile for the GPU (catches the sm_107 failure in seconds) |
| `tools/env_report.sh` | SLURM | prints the hardware and both containers' software versions (source of §3–§4) |
| `preview/` | laptop | early preview built from the first (HF-`generate()`, old wording) pilot - historical only |

### Data formats

- `questions/*.jsonl`: `id` (`<match>-<innings>-<row>-<qtype>`), `qtype`, `answer_kind` (`rate`|`int`), `gold`,
  `messages` (system, user), `source` (row_index, match_id, date, venue, teams, innings, over, ball, batter, bowler,
  innings_runs, innings_wickets, balls_remaining, target_score, batter_runs, batter_balls_faced).
- `samples/*_rank*.jsonl` and `eval/*_rank*.jsonl`: one line per sample: `id`, `qtype`, `sample`, `text`, `n_tokens`,
  `finished`, `pred`, `gold`, `correct`.
- `final/train.jsonl`: `id`, `qtype`, `question`, `messages` (system, user, assistant `<think>…</think>` + answer),
  `reasoning`, `response`, `final_answer`, `gold_answer`, `answer_kind`, `reasoning_tokens`, `samples_generated`,
  `samples_correct`, `source`, `completion_text` (exact model output used as the SFT target; not pushed to the hub).
- `final/test.jsonl`: `id`, `qtype`, `question`, `messages` (system, user), `gold_answer`, `answer_kind`, `source`.

### Cluster access helpers (`bin/`, repository root)

| file | purpose |
|---|---|
| `bin/hecate-bridge` | run in PowerShell (`& 'C:\Program Files\Git\bin\bash.exe' bin/hecate-bridge`), complete MFA once, keep open: one SSH session that executes queued commands |
| `bin/hx` | run a command on the login node through the bridge (`bin/hx 'squeue --me'`) |
| `bin/hsync` | push the repository to `clustercodes/code` (refuses CRLF files); `bin/hsync pull <path>` copies results to `pulled/` |

---

## 8. How to repeat

Prerequisites (already in place on Hecate for this user): Lustre project directory, `secrets.env`, the PyTorch image
and venv (`bash finetune/submit.sh setup` from the FinNews project creates both), the Qwen3.8-27B weights in the HF
cache, enroot credentials for `gitlab-master` (`~/.config/enroot/.credentials`).

From the laptop: start the bridge, then `bin/hsync`. On the login node (or via `bin/hx`), from `/home/bbalakreshna`:

```bash
C=/lustre/fsw/general_sa/bbalakreshna/clustercodes/code/cricket/submit.sh
bash $C import-vllm                       # once: Rubin vLLM image -> containers/vllm-rubin-py3-devel.sqsh
bash $C prepare                           # questions (1 node, <1 min)
bash $C pilot                             # optional: 96 questions x 4 on 1 node (~5 min)
bash $C generate                          # dataset (4 nodes, ~5 min); resubmit the same command if it stops early
# review  .../cricket/data/gen/train/final/samples.md and stats.json, then (approval) :
bash $C push-dataset                      # HF dataset repo (new repos are created private; publish manually)
bash $C train                             # SFT -> merge -> eval-base | eval-ft + report + adapter push (~35 min)
```

`train` prints the four job ids and the run directory. Follow progress with
`tail -f /lustre/fsw/general_sa/bbalakreshna/clustercodes/cricket/logs/general_sa-cricket.<stage>-<job>.log`, the
per-GPU vLLM logs in `logs/general_sa-cricket.<stage>-<job>/`, and the W&B project. GPU use of a running job:
`srun --jobid=<job> --overlap --ntasks=1 nvidia-smi`.

### Common variations

| goal | how |
|---|---|
| more / fewer questions | `EXTRA_ARGS="--n_train 3000 --n_test 300" bash $C prepare` (3 questions per source row) |
| more samples per question or longer traces | edit `G="--k 4 --max_tokens 6144"` in `submit.sh` (raise `--max_model_len` in `gen_vllm.py` if prompt + output may exceed 8192) |
| new question types | add to `QTYPES` and `make_questions()` in `cricket_qa.py` (exact answer + unambiguous wording), run `prepare` and a `pilot` first |
| SFT hyperparameters | `EXTRA_ARGS="--epochs 3 --lr 5e-5 --lora_r 64 --lora_alpha 128" bash $C train` (flags of `train.py`) |
| push the adapter to a branch instead of `main` | add `--hub_revision <branch>` to the `eval-ft` FINISH args in `submit.sh` |
| re-evaluate an existing merged model | run `vllm.sbatch` via `submit_vllm` with `--model_dir <merged> --tag ft` and `FINISH_SCRIPT=finish_eval.py` (see the `train)` branch of `submit.sh`) |
| fresh dataset run (keep the old one) | change `out=$DATA_DIR/gen/train` in the `generate)` branch (and `--data_dir` / `--final_dir` in `train` / `push-dataset`) |

---

## 9. Reproducibility notes

- Seeds: question split and row sampling seed 0; vLLM sampling seed = shard index (distillation and both evals use
  the same seeds); SFT seed 42. vLLM sampling is not bit-exact across runs/GPU counts - expect ±0.15–0.3 pts on
  accuracy between repeats.
- The shard layout (16 shards for generation, 4 for evals) is part of the seed scheme; changing node counts changes
  which samples are drawn but not the expected metrics.
- The test questions come from matches that contribute nothing to training (split by `Match ID`).
- Inputs are pinned: CSV from the dataset repo, model revision in `config.env`, container squashfs files on Lustre.
  Re-importing an image tag later may give a newer build; keep the `.sqsh` to repeat exactly.

---

## 10. Problems hit and their fixes (keep for future runs)

| symptom | cause | fix |
|---|---|---|
| generation at ~110 tok/s per GPU, GPUs 0–36% busy | HF `generate()` is launch/overhead-bound for this hybrid model | vLLM, one engine per GPU (~6,000 tok/s per GPU) |
| HF pilot died with `Watchdog caught collective operation timeout` | finished ranks waited > 30 min at an all-reduce for the slowest rank | vLLM path has no collectives; HF paths now use a 2 h NCCL timeout |
| milestone 9% correct, 90% truncated at 4,096 tokens | ambiguous wording ("their current strike rate" - batter or team?) | named the batter and defined the strike rate in the question |
| vLLM: `An attempt has been made to start a new process before … bootstrapping phase` | vLLM spawns a child that re-imports the script | all work inside `main()` behind `if __name__ == "__main__"` |
| vLLM: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32` | NGC `vllm:26.08-py3` Triton cannot target sm_107 | `dl/dgx/vllm:rubin-py3-devel` (verify any new image with `tools/triton_smoke.py`) |
| workers launched by torchrun could not import transformers | the image's `torchrun` script uses system Python, skipping the venv | `python -m torch.distributed.run` |
| bash on the cluster failed with `pipefail: invalid option` | CRLF line endings from Windows edits | `bin/hsync` refuses CRLF files |

---

## 11. Next steps

- Harder question types so fine-tuning can move accuracy (multi-over chase scenarios, partnership and bowling-economy
  arithmetic, wicket what-ifs, questions combining both innings).
- Make the overs→balls conversion an explicit step in the training traces (all remaining chase_rate errors are there).
- Clean-up candidates on Lustre: empty failed pilot dirs `data/gen/pilot-10051325`, `pilot-10051330`; HF-`generate()`
  pilot `pilot-10051237`; intermediate `checkpoint-*` dirs (7.5 GB) once the adapter is no longer being iterated on.

---
---

# Part II - Full one-to-one dataset (500K) with vLLM, and fine-tuning on 32 GPUs

Dates: 2026-10-05 / 06. Same cluster, account, images, model revision and credentials as Part I unless stated.

## 12. Distilling the 500K-row reasoning dataset with vLLM

### 12.1 Goal and coverage

Requirement: **at least one verified reasoning row for every row** of `Valarmathy/CricketData`
(`raw/ball_by_ball_it20.csv`, dataset revision `a2518e9db3bd6e745dabc645a423eed6f053207f`, **425,119 rows**), about
500K rows in total. `prepare.py --mode full` re-counts the CSV with `csv.reader` and stops unless it has exactly
425,119 rows; the revision and row count are written to `questions-full/stats.json`.

The three Part I question types are undefined for many rows (extras, first balls, all-out innings, rain-affected
matches), so three types were added. Each row gets the **least-used type that is valid for it** (greedy, seeded
shuffle → balanced mix); 74,881 random training rows get a second, different type to reach exactly 500,000.

| type | question | exact answer | rows where valid |
|---|---|---|---|
| `run_rate` | current run rate (overs in cricket notation) | `round(runs*6/balls, 2)` | 424,776 |
| `chase_rate` | rate needed to win (inn. 2) / scored over rest of innings (inn. 1, full 20 overs); not in rain-revised matches | `round(need*6/balls_left, 2)` | 378,962 |
| `milestone` | balls to the next 50 at the striker's own strike rate | `ceil(need*faced/runs)` | 369,258 |
| `projection` *(new)* | projected 20-over total at the current rate (inn. 1) - wording defines the rate as *total runs ÷ overs bowled* | `floor(runs*120/balls + 0.5)` | 223,081 |
| `strike_rate` *(new)* | striker's strike rate | `round(runs*100/faced, 2)` | 400,092 |
| `legal_balls` *(new)* | innings opened with a wide / no-ball: legal deliveries still to come | balls remaining | 40 (rows where nothing else is defined) |

8 rows show "−1 balls remaining" (121 legal balls - a source-data anomaly) and get no overs-based question.
Scores with runs ≤ wickets are spelled out ("1/3 (1 run for the loss of 3 wickets)") - see §12.6.

Split: by match, identical to Part I (177 held-out matches → **test**, 41,298 rows; 1,597 + 68 rain-revised matches →
**train**, 458,702 rows). The Part I 600-question benchmark (`eval.jsonl`) lies entirely in the test matches and
keeps its exact wording, so evaluations stay comparable.

### 12.2 Inference design

| aspect | choice | why |
|---|---|---|
| engine | vLLM offline `LLM.generate` (`gen_vllm.py`) in `dl/dgx/vllm:rubin-py3-devel` (vLLM 0.30.0, torch 2.15 rubin, Triton 3.6, CUDA 13.5) | continuous batching; HF `generate()` reached only ~110 tok/s/GPU on this model (GPUs mostly idle) |
| parallelism | **data parallel: one independent engine per GPU** (`tensor_parallel_size=1`), shard = node × 4 + local GPU, 32 shards on 8 nodes | the bf16 model (51 GB) fits one 279 GiB GPU with a large KV cache; no inter-GPU communication at all, linear scaling, no NCCL timeouts |
| memory | `gpu_memory_utilization=0.90` (≈250 GiB per GPU: weights ~51 GiB, the rest KV cache + activations), `max_model_len 8192`, prefix caching on (the system prompt and match context repeat across the k samples) | Gated-DeltaNet layers keep a fixed-size state; only the 16 full-attention layers grow a KV cache, so hundreds of sequences fit per GPU |
| sampling | thinking mode, `n=4` per question, temperature 1.0, top_p 0.95, top_k 20, `max_tokens 6144`, `skip_special_tokens=False`, seed = shard | model-card settings; 4 samples to pick the shortest correct trace |
| work unit | 256 questions per `generate()` call (= 1,024 sequences queued); records appended to `samples/train_rank{shard}.jsonl` after each call | resume granularity; vLLM keeps the GPU saturated within a call |
| resume | at start each worker reads **all** shard files of its tag and skips sampled ids → safe to resubmit with a different node count | jobs are capped at 2 h; the chain resubmits automatically |
| deadline | workers stop starting new chunks after 100 min (job limit 2 h) | never killed mid-write |
| post-processing | `build_full_dataset.py` (PyTorch container, 1 process) streams the 2M samples, keeps the shortest correct finished trace per question, writes `final/`, `retry.jsonl`, coverage report, W&B summary | 4.7 GB of samples processed without loading everything |

### 12.3 Hardware (per `batch-xdr` node, measured with `tools/collect_stats.sh`)

| component | measured |
|---|---|
| GPUs | 4 × NVIDIA Rubin, compute capability 10.7 (sm_107), **286,524 MiB** each (≈279.5 GiB usable to PyTorch), power limit **2,300 W**, max SM clock 2,424 MHz, max memory clock 4,752 MHz, driver 620.43 |
| GPU ↔ GPU in a node | NVLink, `NV36` between every pair (36 links per GPU, reported 41 GB/s per link) |
| multi-node NVLink | GPU fabric state *Completed / Healthy, bandwidth Full* (nodes belong to an NVLink block, `nvlblk..` node feature) |
| network | **8 × InfiniBand 4X XDR, 800 Gb/s each** (2 per GPU, 1.6 Tb/s ≈ 200 GB/s per GPU, 6.4 Tb/s per node) + 4 × 400 Gb/s Ethernet (NDR) |
| PCIe/NUMA | GPU0-1 + NIC0-5 on NUMA 0 (CPUs 0-87,176-263); GPU2-3 + NIC6-11 on NUMA 1 (CPUs 88-175,264-351) |
| CPU / RAM | 2 × NVIDIA Vera ("Olympus", aarch64), 352 hardware threads; 1.43 TB RAM |

### 12.4 Jobs and compute

| stage | job | nodes × GPUs | wall time | node-h |
|---|---|---|---|---|
| prepare-full (download, verify 425,119 rows, build 500K questions) | 696931 / 696975 (rebuild after rewording) | 1 × 4 | 0:47 / 0:48 | 0.03 |
| pilot-full, 240 questions × 4 (40 per type) | 696932 / 696976 | 1 × 4 | 2:54 / 2:43 | 0.09 |
| **datagen-full** (main pass) | **697019** | **8 × 32** | **1:42:51** | **13.71** |
| datagen-full (continuation: last 2,448 questions) | 697021 | 8 × 32 | 8:20 | 1.11 |
| datagen-full (buffer; nothing left) | 697178 | 8 × 32 | 1:54 | 0.25 |
| retry-full (87 unsolved × 8 samples, then 247 incl. reruns) | 697022 | 1 × 4 | 5:15 | 0.09 |
| push-full (refused: 63 rows uncovered) | 697023 | 1 × 4 | 0:54 | 0.02 |
| fix-ambiguous (clarify runs ≤ wickets scores) | 700650 | 1 × 4 | 0:50 | 0.01 |
| retry-full, tag `retry2` (87 clarified questions × 8) | 700651 | 1 × 4 | 4:12 | 0.07 |
| **push-full** | **700652** | 1 × 4 | 1:12 | 0.02 |
| **total** | | | | **≈ 15.4 node-h ≈ 62 GPU-h** |

### 12.5 Throughput and GPU efficiency

| metric | value |
|---|---|
| samples generated | **2,002,672** (2,000,000 first pass + 1,976 retry + 696 retry2) |
| tokens generated | **1,581,921,168** (mean 790 per sample; prompts ~220 tokens each are prefill, not counted) |
| per-GPU generation throughput, main job | **8,238 tok/s mean** (min 8,003, max 8,386 over 32 GPUs - ±2%, i.e. even load) |
| aggregate throughput | **≈ 264,000 tok/s** on 32 GPUs; 497,552 questions in 99 min of generation |
| engine start-up | ~52 s per engine (model load from Lustre + CUDA graphs); 4 engines per node share the page cache |
| throughput in the 1-node pilots / retries | 8,800–9,500 tok/s per GPU on full chunks; 2,300–2,900 on tiny retry batches (87 questions: too few concurrent sequences) |
| tail inefficiency | the continuation job 697021 used 8 nodes for 2,448 questions (16 of 32 engines had work, 3,230 tok/s mean); a 1-node tail job would have cost 8× less |
| compute estimate | decode ≈ 2 × 27 B FLOP/token → 8,238 tok/s ≈ **0.45 PFLOP/s per GPU** of dense matmul (rough; excludes prefill, attention and sampling) |
| GPU memory | 90% of 279.5 GiB reserved by vLLM (weights ≈ 51 GiB; remainder KV cache / activations / CUDA graphs) |
| network | **no GPU↔GPU traffic** (independent engines); NVLink and InfiniBand idle apart from Lustre I/O: model load ≈ 52 GB per node per job, sample output 4.7 GB in total |
| cost per question | ≈ 7.4 GPU-seconds (4 samples), i.e. ≈ 480 questions per GPU-hour |

Throughput per GPU was ~55× the HF-`generate()` path (110 tok/s) used in Part I's first pilot.

### 12.6 Quality, retries and the ambiguity fix

Base-model results on all 500,000 questions (2.0 M samples, before selection):

| type | samples | generated tokens | correct per sample | finished (not truncated) | questions solved |
|---|---|---|---|---|---|
| run_rate | 400,648 | 172.1 M | 97.75% | 99.77% | 100% |
| chase_rate | 401,352 | 435.3 M | 92.52% | 98.97% | 100% |
| milestone | 400,024 | 350.6 M | 99.20% | 99.39% | 100% |
| projection | 400,496 | 396.8 M | 97.89% | 99.51% | 100% |
| strike_rate | 399,992 | 226.9 M | 99.13% | 99.21% | 100% |
| legal_balls | 160 | 0.14 M | 100% | 100% | 100% |
| **overall** | **2,002,672** | **1,581.9 M** | **97.30%** | 99.37% | **100%** |

Two wording problems were found and fixed **before** they could cost compute or coverage:

1. **Projection overthinking (pilot).** "Keep scoring at their current run rate" made the model debate whether the
   rate meant the recent over and reconstruct the batting order from distractor details: 88.7% correct, 10%
   truncated, 2,215 tokens. Defining the rate in the question ("total runs so far divided by the overs bowled so far")
   → 98.1% correct, 0.6% truncated, 869 tokens.
2. **Runs/wickets notation (after the main pass).** 87 questions stayed unsolved after 12 samples each, leaving 63
   source rows uncovered; the upload correctly refused. All 87 had **runs ≤ wickets** ("1/3"): in most countries
   runs/wickets, in Australia wickets/runs, and the model consistently chose the Australian reading.
   `tools/fix_ambiguous_scores.py` rewrote these questions in place ("1/3 (1 run for the loss of 3 wickets)"), the
   `retry2` pass solved all 87, and `_ctx()` now spells such scores out for future builds.

Final dataset (`data/gen/train-full/final/stats.json`): **500,000 rows** (train 458,702, test 41,298), **425,119 /
425,119 source rows covered**, 0 unsolved, kept traces 464 tokens on average (shortest correct of the samples).

### 12.7 Outputs and storage

| path (under `cricket/data/`) | size | content |
|---|---|---|
| `questions-full/{train,test,all,eval,pilot}.jsonl`, `stats.json` | 1.2 GB | questions (with `split`, `source`, exact `gold`) |
| `gen/train-full/samples/{train,retry,retry2}_rank*.jsonl` | 4.7 GB | every sample: text, tokens, finished, prediction, correct |
| `gen/train-full/final/train.jsonl` / `test.jsonl` | 2.5 GB / 228 MB | dataset rows (+ `completion_text`, the exact SFT target) |
| `gen/train-full/final/{eval.jsonl, stats.json, samples.md, uncovered_rows.jsonl}` | small | benchmark copy, statistics, review examples, coverage report (empty) |
| `gen/train-full/retry.jsonl` | - | unsolved questions (empty at the end) |
| Hugging Face `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` | 2.08 GB | train in 5 shards of 100K rows, test, eval, card, stats |

### 12.8 How to repeat

```bash
C=/lustre/fsw/general_sa/bbalakreshna/clustercodes/code/cricket/submit.sh
bash $C prepare-full                       # verify 425,119 rows, build 500K questions (+ eval.jsonl, pilot.jsonl)
bash $C pilot-full                         # 240 questions, check every type (accuracy, tokens, truncation)
bash $C generate-full                      # 8 nodes (GEN_NODES=..), resubmit / chain with --dependency=afterany:<job>
bash $C retry-full                         # 8 more samples for unsolved questions (tag retry; use EXTRA_ARGS="--tag retry2" for a 2nd round)
bash $C fix-ambiguous                      # only if unsolved questions have runs <= wickets scores
bash $C push-full                          # refuses while any source row is uncovered
```

Tips: size the continuation job to the remaining work (`GEN_NODES=1` for a small tail); run a stratified pilot after
any wording change; check `final/uncovered_rows.jsonl` before publishing.

---

## 13. Fine-tuning on the 500K dataset with 32 GPUs

### 13.1 Goal

One epoch over the 458,702 training rows (457,702 after a 1,000-row dev set) **inside the 5-hour job limit** of
`batch-xdr`, then the same evaluation as Part I (600-question benchmark, vLLM, k = 4) and an automatic push of the
adapter to `Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA`.

### 13.2 Throughput benchmark (1 node, 4 GPUs, 20,000 real rows, 30 steps each)

| variant | job | per-GPU batch | grad. ckpt | batching | rows/s (4 GPUs) | rows/s/GPU | result |
|---|---|---|---|---|---|---|---|
| D - Part I settings | 700694 | 2 × accum 2 | on | random | 1.88 | 0.47 | baseline; 1 epoch on 32 GPUs ≈ 8.5 h |
| **A - chosen** | 700695 | **8** | on | **length-grouped** | **4.55** | **1.14** | **2.4× faster**; 1 epoch on 32 GPUs ≈ 3.4 h |
| B | 700696 | 16 | on | length-grouped | - | - | **OOM**: tried to allocate 90.7 GiB with 252 GiB already in use |
| C | 700697 | 8 | **off** | length-grouped | - | - | **OOM**: activations without recomputation exceed 279 GiB |

Why B fails: the vocabulary has 248,320 tokens, so the output logits of a batch take
`batch × seq × 248,320 × 4 B` in fp32 - for 16 × 6,144 tokens that alone is ~98 GB, plus its gradient. Length
grouping in HF Trainer puts the **longest** mega-batch first, so a configuration that survives the first steps is
safe for the whole epoch (variant A passed this). Mean sequence length: 700 tokens (≈ 220 prompt + 464 reasoning +
answer); 0 rows exceed `max_len 6144`.

### 13.3 Configuration (stage `train-full`)

| parameter | value |
|---|---|
| nodes / GPUs | **8 nodes × 4 = 32 Rubin GPUs**, `--exclusive`, job limit 4:59:00, stop new steps at 280 min |
| parallelism | plain **DDP**, full bf16 model replica per GPU (≈ 51 GiB); only LoRA weights are trainable and all-reduced |
| LoRA | r 32, α 64, dropout 0.05, same targets as Part I → 217,579,520 trainable parameters (0.79%) |
| batch | **8 rows per GPU × 32 GPUs = global 256**, no gradient accumulation; length-grouped sampler (`train_sampling_strategy="group_by_length"`) |
| schedule | 1 epoch ≈ 1,788 steps, lr **2e-4** (2× Part I for the 4× larger batch), cosine, warmup 3%, AdamW, bf16, gradient checkpointing (non-reentrant) |
| data pipeline | global rank 0 tokenizes all rows once (batched fast tokenizer, ~5 min) into `cricket/cache/tok-*.npz` (flat int32 tokens + offsets + prompt lengths, 215 MB for the 20K benchmark caches; ~1.3 GB for the full set); every rank loads it; labels = tokens with the prompt masked |
| dev / checkpoints | fixed 1,000-row dev set; dev loss + checkpoint every 200 steps (~23 min), keep 2 |
| resume | a second `sft-full` job (afterany) resumes from the newest checkpoint if the first stops early, and exits immediately if `train_result.json` exists |
| chain | `sft-full` → resume-safety `sft-full` → `merge` → `eval-ft` (with `eval-base` run in parallel) → report + W&B + push |
| reference jobs | 700715 (SFT), 700716 (resume safety), 700717 (merge), 700718 (eval-base, done: 3:07), 700719 (eval-ft + push); run `general_sa-cricket.sft-full-10060441` |

### 13.4 Communication and network

Per optimizer step each GPU all-reduces the LoRA gradients: 217.6 M parameters × 4 B (fp32 adapters) ≈ **0.87 GB**.
A ring all-reduce moves ≈ 2 × (31/32) × 0.87 ≈ **1.7 GB per GPU per step**. At ≈ 7 s per step (256 rows ÷ 36.5
rows/s) that is ≈ **0.25 GB/s per GPU**, against ≈ 200 GB/s of InfiniBand per GPU (2 × 800 Gb/s XDR) plus NVLink
inside the node - **well under 1% of the available bandwidth**. LoRA + DDP is compute-bound; the network is not a
factor, and the startup broadcast of the frozen 51 GB of weights is skipped (`_ddp_params_and_buffers_to_ignore`,
every rank loads identical weights from Lustre instead). Measured values: see §13.6.

### 13.5 Memory per GPU (279.5 GiB usable)

| item | size |
|---|---|
| frozen base weights (bf16) | ≈ 51 GiB |
| LoRA weights + grads + AdamW states (fp32: 4 + 4 + 8 B per parameter) | ≈ 3.5 GB |
| activations with gradient checkpointing, batch 8 × up to 6,144 tokens | layer inputs kept, one layer recomputed at a time |
| output logits + gradient for the longest batch (8 × 6,144 × 248,320 × 4 B, ×2) | up to ≈ 98 GB transient |
| headroom | variant A fits; batch 16 or no checkpointing does not (§13.2) |

### 13.6 Live measurements during training

*Not yet run.* On 2026-10-06 the cluster could not schedule the job: a `fwupdates` firmware-update reservation took
144 nodes out of service (all `maint` nodes on batch-xdr and the idle ones on batch-spx), leaving ~140 batch-xdr nodes
against ~7,000 queued node requests ahead of ours (rank ≈ 380, priority ≈ 44,600, mostly fair-share). The chain was
tried as one 8-node × 5 h job (batch-xdr), 8 nodes × 8 h (backfill-xdr, preemptible), and 8-node × 2 h chunks on
`batch-xdr,batch-spx`, then cancelled by the user, to be rerun after the maintenance (planned: Thursday 2026-10-08).

To rerun (chunked, starts on whichever partition has room first):

```bash
PARTITION=batch-xdr,batch-spx GEN_NAME=train-full TRAIN_TIME=02:00:00 TRAIN_MIN=105 TRAIN_CHUNKS=4 SAVE_STEPS=100 \
  bash $C train-full
bash .../code/cricket/tools/measure_job.sh <sft job id> 60     # ~20 min into training: GPU util/mem/power + IB traffic, all nodes
```

Scheduling notes: `squeue -j <job> --start` and `sprio -j <job>` show the plan and priority; the number of jobs ahead
and the nodes they want: `squeue -p batch-xdr -t PD -h -o "%Q %D" | awk -v p=<prio> '$1>p{n++;d+=$2}END{print n,d}'`;
`scontrol show reservation` shows maintenance windows. `backfill-xdr` (8 h) is preemptible (`PreemptMode=CANCEL`,
lower tier) - only use it with frequent checkpoints and resume jobs.

### 13.7 Expected timeline

| phase | estimate |
|---|---|
| queue wait for 8 nodes × 5 h | variable (hours on a busy day) |
| model load + tokenization of 458K rows | ≈ 6 min |
| training, 1,788 steps at ≈ 7 s | **≈ 3.4 h** |
| merge (1 node) + eval-ft (1 node) + push | ≈ 10 min |

### 13.8 How to repeat

```bash
GEN_NAME=train-full bash $C train-full                 # 8 nodes (TRAIN_NODES=..), prints all job ids and the run dir
# throughput experiment on 1 node with any train.py flags:
GEN_NAME=train-full EXTRA_ARGS="--max_steps 30 --train_limit 20000 --per_device_batch 8 --sampling group_by_length" bash $C bench
```

## 14. Lessons from the scale-up

| lesson | detail |
|---|---|
| pilot every question type before a big run | two wordings (projection rate, runs/wickets scores) would otherwise have cost ~10% of 2 M samples or broken coverage |
| make the coverage check a hard gate | `push_full_dataset.py` refuses while any source row is uncovered - it caught the 63 rows |
| one vLLM engine per GPU is the right shape for a 27B model on 279 GiB GPUs | zero communication, ±2% throughput spread across 32 GPUs, trivially resumable |
| size continuation jobs to the remaining work | an 8-node job for 2,448 leftover questions ran at a quarter of the normal throughput |
| benchmark training settings on 1 node first | 5 minutes per variant found a 2.4× speed-up and exposed the two OOM settings |
| logits dominate memory at a 248K vocabulary | batch size, not model size, is the limit for SFT on these GPUs |
| tokenize once, share | a Lustre token cache replaces 32 parallel tokenizations of 458K rows |
