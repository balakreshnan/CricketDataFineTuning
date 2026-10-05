# Cricket T20I reasoning: data distillation, fine-tuning and evaluation on Hecate

End-to-end record of the pipeline that turns the ball-by-ball cricket dataset
[`Valarmathy/CricketData`](https://huggingface.co/datasets/Valarmathy/CricketData) into a **verified reasoning
dataset** distilled from `Qwen/Qwen3.8-27B` (thinking mode), fine-tunes a **LoRA reasoning adapter** on it, and
evaluates base vs fine-tuned on held-out matches. Everything runs as SLURM batch jobs on the Hecate (Vera Rubin)
cluster. This document is the runbook: it lists the compute, images, software versions, configuration, file-system
layout and every file involved, so the run can be repeated or modified.

Reference run: **2026-10-05**, SFT job `696652`, user `bbalakreshna`, account `general_sa`, partition `batch-xdr`.

## Published artifacts (public on Hugging Face)

| artifact | link | contents | revision at publication |
|---|---|---|---|
| **Reasoning dataset** | [Balab2021/CricketData-T20-Reasoning-Qwen3.8](https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8) | `data/train.jsonl` (4,500 verified reasoning rows, 3 per source delivery), `data/test.jsonl` (600 held-out questions with gold answers), `stats.json`, dataset card; license CC0-1.0 (as the source data) | `efa295283fc9b2a012b9fa06a77bf393e35de212` |
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
