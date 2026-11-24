# Tools used: Nemotron 3.5 Lightning 30B on Rhea (Vera Rubin)

What each piece of software does, grouped by which path uses it. Versions are the ones installed on Rhea on
2026-11-24.

There are two training paths for the same task (cricket questions with exact numeric answers, LoRA, reward from a
checker):

- **`rlvr.py` path**: one self-contained PyTorch script written for this project. Its queued 8-node run was
  cancelled on 2026-11-24 in favour of the NeMo RL path.
- **NeMo RL path**: the same job split into three cooperating services (trainer, vLLM, NeMo Gym) under Ray.
  Used for the full run on 4 nodes (100 steps, see section 6).

Evaluation is the same for both.

## 1. `rlvr.py` path (folder `rhea/`, `rhea/nemotron/`)

| Function | Tool | Notes |
|---|---|---|
| Model loading, forward and backward pass | PyTorch 2.15 (NVIDIA build) + Hugging Face `transformers` 5.22 | `NemotronHForCausalLM`, BF16 |
| LoRA adapters | Hugging Face `peft` 0.21 | rank 32, alpha 64, 22.7M trainable parameters |
| Generating answers during training | Project sampling loop (`sampling.py`) on top of `transformers` | no serving engine |
| Reward | `cricket_qa.py` (project code) | exact-answer checker; correct answer scores 1 - 0.4 x tokens/4096 |
| Algorithm | Project code (`rlvr.py`) | policy gradient with a per-question baseline, one update per batch |
| Multi-GPU and multi-node | PyTorch DDP, started by `torchrun` | gradients averaged across all GPUs |
| Container | `gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel` (arm64) | the PyTorch image built for Vera Rubin, Python 3.14 |

## 2. NeMo RL path (folder `rhea/nemorl/`)

| Function | Tool | Notes |
|---|---|---|
| Training loop and algorithm | NeMo RL (`nemo_rl`, main branch, commit a03ffa1) | GRPO: ratio clipping, importance-sampling correction, leave-one-out baseline. Driver: `examples/nemo_gym/run_grpo_nemo_gym.py` |
| Process orchestration | Ray | starts the trainer, vLLM and NeMo Gym workers as actors; local Ray cluster on one node |
| Policy training backend | NeMo Automodel 0.6.1 on PyTorch 2.13 | its own Nemotron implementation, FSDP across GPUs, its own LoRA layers (not `peft`) |
| Compiled kernels for training | Transformer Engine 2.18, flash-attn 2.8.1, mamba-ssm 2.3.2, causal-conv1d 1.6.2, DeepEP, grouped GEMM | built from source on a Rhea node (about 30 minutes the first time) |
| Generating answers | vLLM 0.33 on PyTorch 2.14, with FlashInfer 0.7.2 | one engine per GPU, shares the GPU with the trainer, exposed as an OpenAI-compatible HTTP server |
| Weight sync trainer -> vLLM | NeMo RL refit over CUDA IPC | `lora_refit_mode: merged`: LoRA merged into full weights before each push |
| Environment and reward | NeMo Gym (version pinned by NeMo RL) | cricket environment `nemorl/gym/cricket_qa`; calls vLLM over HTTP, scores answers with `cricket_qa.py` |
| Config | Hydra / OmegaConf YAML | `nemorl/cricket_rhea.yaml`, inherits NeMo RL's `grpo_math_1B.yaml` |
| Python environments | `uv` | one environment per worker type, Python 3.13.14, on Lustre |
| JIT kernel compiler | Triton 3.7 (trainer) / 3.8 (vLLM) | needs `nemorl/rubinfix/sitecustomize.py` to run on Vera Rubin |
| Container | same PyTorch image as above | used only for its compilers and CUDA toolkit; NeMo RL's Python stack comes from `uv` |

## 3. Evaluation (both paths)

| Function | Tool | Notes |
|---|---|---|
| Benchmark, before vs after | `eval.py` (project code) on `transformers` | 600 held-out questions x 4 answers |
| Environment evaluation | NeMo Gym 0.7.0 CLI (`gym eval run`) | cricket environment `nemotron/gym/cricket_qa`, Python 3.13 environment built with `uv` |
| Serving for that evaluation | vLLM 0.31 from `gitlab-master.nvidia.com/dl/dgx/vllm:rubin-py3-devel` | serves the base model and the LoRA adapter side by side; reasoning parser `nemotron_v3` |
| Summaries | `nemotron/gym_summary.py` (project code) | per-question-type accuracy, pass@k, tokens |

## 4. Shared infrastructure

| Function | Tool |
|---|---|
| Job scheduling | Slurm (account `general_sa`), containers through enroot / pyxis |
| Storage for models, data, environments, checkpoints, logs | Lustre (`/lustre/fsw/general_sa/bbalakreshna`) |
| Metrics | Weights & Biases |
| Model and adapter storage | Hugging Face Hub (`huggingface_hub`), private repos |
| GPU utilisation, memory, power, temperature | `nvidia-smi`, logged every 15 s on every node |
| Dataset | `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` (458,702 training questions, 600 held out) |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Access from the laptop | `rhea/bin/rhea-bridge` (run by the user), `rhea/bin/rx`, `rhea/bin/rhsync` |

## 5. What had to be done for Vera Rubin

| Item | `rlvr.py` path | NeMo RL path |
|---|---|---|
| PyTorch | NVIDIA Rubin build in the container | public PyTorch 2.13 / 2.14 wheels work for core operations |
| Triton | Rubin build in the container | public Triton cannot target Rubin; patched by `nemorl/rubinfix` |
| Kernels built from source | none | built with `TORCH_CUDA_ARCH_LIST="9.0 10.0"` (Rubin runs sm_100 code) |
| cuDNN | container | pip cuDNN must be first on `LD_LIBRARY_PATH` for Transformer Engine |
| Attention speed | cuDNN attention kernel disabled in `sampling.py` | not needed (vLLM generates) |

## 6. Status of the NeMo RL path

Works on 1, 2 and 4 nodes (4 Vera Rubin GPUs each). For more than one node, `nemorl/train.sbatch` starts a Ray head
on the first node and a Ray worker on every other node, and NeMo RL attaches to that cluster (`RAY_ADDRESS`).

Full run `nemorl-cricket-1124` (2026-11-24): 4 nodes, 16 GPUs, 100 steps of 64 questions x 8 answers, checkpoint
every 5 steps, metrics in W&B project `nemotron35-cricket-nemo-rl-rhea`.

| Job | What | Result |
|---|---|---|
| 131698 | Steps 1 to 70 | Stopped by its 90-minute limit |
| 131714 | Resumed from the step-65 checkpoint, steps 66 to 100 | Completed in 31 minutes |
| 132709 | NeMo Gym evaluation, 600 held-out questions x 4 answers, vLLM serving the adapter as a LoRA module | Completed |

| | Base model | Base + adapter |
|---|---|---|
| Accuracy | 86.50% | 97.71% |
| Average output tokens | 1,565 | 107 |
| Answers cut off at 4,096 tokens | 9.42% | 0.00% |
| `run_rate` accuracy | 91.75% | 99.00% |
| `chase_rate` accuracy | 68.12% | 94.62% |
| `milestone` accuracy | 99.62% | 99.50% |

Notes:

- Training reward on the sampled questions went from 0.72 (step 1) to about 0.96 (step 100); step time fell from
  233 s to about 40 s as answers got shorter.
- The output is a LoRA adapter only (`adapter_model.safetensors`, 95 MB, PEFT key names). The base weights are not
  changed, so the adapter must be applied to or merged into the base model at inference.
- A resumed job logs to a new W&B run (steps 1 to 70 in run `kqadpwsx`, 66 to 100 in `ijt9bvh1`).
- Jobs of 90 minutes started within minutes; longer jobs were estimated to wait 7 to 8 hours, so long runs are
  chains of 90-minute jobs that resume from the latest checkpoint.
- `reasoning_parser_plugin` in the config must be an absolute path when more than one node is used.
