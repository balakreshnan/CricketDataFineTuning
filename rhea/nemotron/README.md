# Nemotron 3.5 Lightning 30B on Rhea: RLVR fine-tune and NeMo Gym evaluation

Goal: fine-tune `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on the cricket dataset
(`Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full`) with RLVR and a LoRA adapter, then evaluate the base and the
fine-tuned model with NeMo Gym in a cricket environment built from the 600 held-out questions. Everything runs on
Rhea as Slurm jobs; weights, containers, runs and logs are on Lustre; metrics go to W&B; the adapter goes to a
private Hugging Face repo.

Decisions (user, 24 Nov 2026): cricket dataset; RLVR with LoRA; NeMo Gym evaluation in a cricket environment.

## Images for Vera Rubin (tested on Rhea, job 128654, node rhea0123)

Rhea's nodes are aarch64 with 4 Vera Rubin GPUs each (compute capability 10.7, CUDA 13.5). An image must be built
for arm64 and for this GPU generation.

| Purpose | Image | On Lustre | Test result |
|---|---|---|---|
| Training (RLVR, LoRA) | `gitlab-master.nvidia.com/dl/dgx/pytorch:main-py3-devel` | `containers/pytorch-main-py3-devel-arm64.sqsh` (15 GB) | Pass: Python 3.14, torch 2.15 (NVIDIA internal build), bf16 matmul, attention and a freshly compiled Triton kernel all ran on the GPU. Same image as the Qwen RLVR runs. |
| Serving for the evaluation (vLLM) | `gitlab-master.nvidia.com/dl/dgx/vllm:rubin-py3-devel` | `containers/vllm-rubin-py3-devel-arm64.sqsh` (11 GB, imported by this job) | Pass for the basics: an arm64 build exists, vLLM 0.31.0 (Rubin build), torch 2.15 (Rubin build), bf16 matmul and attention ran on the GPU, and vLLM lists Nemotron model classes. Not yet tested: actually serving this model. The Triton line in the log says FAIL because of how the test defined its kernel, not because of the image. |
| NeMo Gym 0.7.0 | pip package `nemo-gym` (pure Python wheel), to be installed into a venv on Lustre | not installed yet | A dry-run install resolves on Python 3.14 / aarch64. Not yet installed or run. NeMo Gym is CPU-only orchestration; it talks to vLLM over HTTP. |

Images that do not work on Vera Rubin (earlier findings, other projects):

- Public NGC `nvcr.io/nvidia/vllm:26.08`: its Triton cannot compile for this GPU generation.
- Public `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1`: its precompiled kernels have no build for this GPU.

Not needed here: the NeMo / Megatron-Bridge image used for the full fine-tune on EOS (this project uses LoRA).

## Status (24 Nov 2026)

- Model on Lustre (61 GB, 32.9B parameters). It loads in the training image with transformers 5.22 as
  `NemotronHForCausalLM`: 52 layers = 23 Mamba, 23 mixture-of-experts (128 routed experts, 6 per token, one shared
  expert), 6 attention.
- RLVR code adapted (`../rlvr.py`, `../sampling.py`, `../eval.py`, `../rlvr_finish.py`; profile `PROFILE=nemotron` in
  `../config.env`). LoRA goes on 93 modules: attention projections, the Mamba input projection and the shared
  experts, 22.7M trainable weights (0.07%). The adapter library refuses the Mamba output projection, and the routed
  experts are stacked weights, so both stay frozen.
- Smoke test passed (job 128883: 3 steps, checkpoints, adapter, W&B, GPU log; peak memory 105 GB).
- Speed fix: the cuDNN attention kernel made generation 14 times slower (250 ms against 18 ms per step); it is now
  switched off in `../sampling.py` for every script. Batch 8 generates 417 tokens/s on one GPU.
- The optimized Mamba kernels (`mamba_ssm`, `causal_conv1d`) are not installed; the reference implementation is used.
  Generation is fast enough; the training update on long answers is slow (up to about 100 s per backward pass).
- NeMo Gym 0.7.0 works: venv on Python 3.13 (it fails on 3.14), cricket environment `gym/cricket_qa`, vLLM serving
  the base model and the adapter as a LoRA module. Test on 60 questions x 2 answers (job 129260): base 91.67%.
- Queued: training chain 129034 (8 nodes) -> 129035 (resume) -> 129036 (benchmark) -> 129037 (W&B table, private
  Hugging Face upload), and the full NeMo Gym evaluation after training. The cluster is full; the 8-node job waits.

## Commands

```bash
bin/rhsync
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr-smoke -- --max_new_tokens 4096 --len_coef 0.4 --train_batch 2'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr -N 8 -t 03:00:00 -n nemotron-rlvr-1124 -- --steps 60 --max_new_tokens 4096 --len_coef 0.4 --train_batch 4 --save_steps 10'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh eval -N 4 -t 02:00:00 -n nemotron-rlvr-1124 -- --batch_prompts 8'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr-finish -n nemotron-rlvr-1124'
bin/rx -t 60 'source ~/clustercodes/nemotron/nemotron.env; sbatch -p "$PARTITION" -t 02:00:00 -o "$NT_LOGS/%x-%j.out" --export=ALL,CODE_DIR=$HOME/clustercodes,RUN_NAME=nemotron-rlvr-1124,REPEATS=4,CONC=128 ~/clustercodes/nemotron/gym_eval.sbatch'
```

## Plan

1. Images (done) and model download to `models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (`download.sbatch`).
2. Inspect the model: class, layer names for LoRA, chat template. Adapt `../rlvr.py` (written for Qwen) as needed.
3. RLVR smoke test on 1 node, then the run on 8 nodes with GPU logging on every node, W&B logging.
4. NeMo Gym: install into `venvs/nemo-gym`, cricket environment (exact-answer checker), vLLM serving the base
   model and the model with the adapter, evaluation of both on the 600 held-out questions.
5. Upload the adapter (private) and the evaluation results; report.

## Where things go

| What | Where |
|---|---|
| Model weights | `/lustre/fsw/general_sa/bbalakreshna/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Runs, evaluation outputs | `/lustre/fsw/general_sa/bbalakreshna/nemotron-rhea/<run-name>/` |
| Job logs | `/lustre/fsw/general_sa/bbalakreshna/logs/nemotron-rhea/` |
| W&B | project `nemotron35-cricket-rlvr-rhea` |
| Hugging Face | private `Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-RLVR-Rhea` |

## Files

| File | Purpose |
|---|---|
| `nemotron.env` | settings: model, images, NeMo Gym version, W&B project, Hugging Face repo |
| `images.sbatch` | imports and tests the container images on a Vera Rubin node |
| `download.sbatch` | downloads the model to Lustre and prints what is needed to adapt the training code |
| `gen_probe.sbatch`, `gen_probe.py` | generation speed test (found the cuDNN attention slowdown) |
| `gym_setup.sbatch` | installs NeMo Gym into a Python 3.13 venv on Lustre and prints its CLI and scaffold |
| `gym/cricket_qa/` | the NeMo Gym environment: `app.py` (exact-answer verifier) and its config |
| `gym_prepare.py` | installs the environment into the Gym work directory and converts the 600 questions to tasks |
| `gym_eval.sbatch` | serves base model and adapter with vLLM, runs the NeMo Gym evaluation, logs GPU usage |
| `gym_summary.py` | accuracy and tokens per question type from the rollouts; W&B table |
