# Tools used: Nemotron 3.5 Lightning 30B on Rhea (Vera Rubin)

What each piece of software does, grouped by which path uses it. Versions are the ones installed on Rhea on
2026-11-24.

There are two training paths for the same task (cricket questions with exact numeric answers, LoRA, reward from a
checker):

- **`rlvr.py` path**: one self-contained PyTorch script written for this project. Queued on 8 nodes.
- **NeMo RL path**: the same job split into three cooperating services (trainer, vLLM, NeMo Gym) under Ray.
  Smoke-tested on one node (job 131251, 3 steps, passed).

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

Tested on one node, 4 GPUs (job 131251, run `nemorl-smoke-1124`):

| Step | Average reward | Average answer length (tokens) | Step time |
|---|---|---|---|
| 1 | 0.72 | 1,667 | 238 s |
| 2 | 0.75 | 1,470 | 138 s |
| 3 | 0.82 | 1,460 | 128 s |

Each step: 16 questions x 8 answers. Peak GPU memory 106 GB of 286 GB. The checkpoint contains a LoRA adapter
(`adapter_model.safetensors`, 95 MB).

Not tested yet: more than one node (needs NeMo RL's Ray-on-Slurm launcher `ray.sub`), W&B logging, Hugging Face
upload, a full-length run.
