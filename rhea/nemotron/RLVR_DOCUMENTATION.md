# RLVR for Nemotron 3.5 Lightning 30B on Rhea: how it is built

Written 24 Nov 2026. Status at the time of writing: the code is adapted and smoke-tested; the full 8-node training
run (job 129034) is queued and has not started.

## 1. What is used, and what is not

**NeMo RL is not used in this project.** The training is done by a script written for this project,
[`rhea/rlvr.py`](../rlvr.py), in plain PyTorch with the Hugging Face `transformers` and `peft` libraries. The only
NVIDIA NeMo component in use is **NeMo Gym**, and only for the evaluation.

| Part | What does it | NeMo product? |
|---|---|---|
| Training (RLVR) | `rhea/rlvr.py`: own rollout loop, reward, policy-gradient update, LoRA | No |
| Launch on the cluster | Slurm + `torch.distributed.run`, one process per GPU, DDP gradient averaging | No |
| Standard before/after benchmark | `rhea/eval.py` + `rhea/sampling.py` (Hugging Face `generate`) | No |
| Evaluation in an environment | **NeMo Gym 0.7.0**: cricket environment `nemotron/gym/cricket_qa`, model served by vLLM | Yes |
| Metrics | W&B | No |
| Weights | LoRA adapter to Lustre and a private Hugging Face repo | No |

Why not NeMo RL: the project already had a working RLVR script from the Qwen runs on this cluster, and you chose
"RLVR with LoRA" for Nemotron, so the same script was adapted. NeMo RL was not installed or tested on Rhea.

How NeMo RL relates, for orientation only: NVIDIA's documentation describes NeMo RL as a training framework that uses
NeMo Gym environments to collect rollouts for GRPO and on-policy distillation, with NeMo Gym as CPU-only
orchestration ([NeMo Gym integration, NeMo RL docs](https://docs.nvidia.com/nemo/rl/latest/design-docs/nemo-gym-integration.html)).
I have not read its code or run it, so nothing in this document describes how NeMo RL works internally. The cricket
environment written for the evaluation (section 8) is the piece that such a framework would reuse.

## 2. Overview

```
                 training (8 nodes x 4 GPUs = 32 processes, one per GPU)
 cricket dataset ──> questions ──> rollouts ──> exact-answer check ──> reward ──> advantage ──> LoRA update
 (train split,        4 per GPU     8 answers     cricket_qa.py         1 - len      minus the     gradients averaged
  gold answers)       per step      per question                         penalty      group mean    over all GPUs (DDP)
                                                                                                        │
                        every 10 steps: checkpoint ─────────────────────────────────────────────────────┤
                        every step: metrics.jsonl + W&B; every 15 s: nvidia-smi on every node            ▼
                                                                                              adapter on Lustre
 evaluation
   eval.py        : base and adapter-merged model, 600 held-out questions x 4 answers  ──> before/after table (W&B)
   NeMo Gym       : vLLM serves base + adapter ──> cricket_qa environment verifies answers ──> rewards, summaries (W&B)
   rlvr_finish.py : model card + private Hugging Face upload
```

## 3. The model

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, facts from its `config.json` and weights on Lustre:

| Property | Value |
|---|---|
| Class in `transformers` 5.22 | `NemotronHForCausalLM` (`model_type: nemotron_h`) |
| Parameters | 32.9B in the checkpoint (31.6B counted by PEFT after loading) |
| Layers | 52: 23 Mamba (state-space), 23 mixture-of-experts, 6 attention |
| Mixture of experts | 128 routed experts per layer, 6 used per token, plus 1 shared expert |
| Hidden size, vocabulary | 2,688; 131,072 tokens |
| Stop tokens | ids 2 (`</s>`) and 11 (`<|im_end|>`); pad id 11 |
| Reasoning | chat template flag `enable_thinking`, on by default; prompt ends with `<think>` |
| Recommended sampling | temperature 1.0, top-p 0.95 |

## 4. `rlvr.py`, step by step

One Python process runs per GPU. All processes run the same script; `RANK` tells them apart.

### 4.1 Start-up

```python
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"), timeout=timedelta(hours=2))
WORLD, dev = dist.get_world_size(), torch.device(f"cuda:{LOCAL_RANK}")
```

The training questions are read from the dataset's `train-*.jsonl` files (first 100,000 rows), each with its gold
answer and answer kind. Every process shuffles them with the same seed, so they all hold the same list, and each takes
its own slice per step:

```python
base = (step - 1) * per_step + RANK * a.questions_per_gpu
qs = rows[base: base + a.questions_per_gpu]
```

No question is used twice in a run.

### 4.2 Loading the model and attaching LoRA

```python
tok = sampling.load_tokenizer(a.model_dir)
model = sampling.load_model(a.model_dir, LOCAL_RANK)
MTYPE = model.config.model_type
targets = a.lora_targets or next(v for k, v in LORA_TARGETS.items() if MTYPE.startswith(k))
...
model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                                         target_modules=targets, task_type="CAUSAL_LM"))
```

`sampling.load_model` picks the class from the model type: Qwen checkpoints load as vision-language models, everything
else (Nemotron) with `AutoModelForCausalLM`, in bfloat16 on that process's GPU. The base weights are frozen; only the
LoRA matrices train.

The LoRA target pattern for Nemotron:

```python
"nemotron_h": r"model\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj|in_proj|shared_experts\.(up_proj|down_proj))",
```

| Layer type | LoRA on | Left frozen | Why |
|---|---|---|---|
| Attention (6 layers) | `q_proj`, `k_proj`, `v_proj`, `o_proj` | | |
| Mamba (23 layers) | `in_proj` | `out_proj`, `conv1d` | PEFT refuses them for Mamba models: the Mamba kernel reads their weights directly, bypassing the module |
| Mixture of experts (23 layers) | shared expert `up_proj`, `down_proj` | the 128 routed experts, the router | routed experts are stored as stacked weights, not as separate linear modules |

Result: 93 modules, 22,718,464 trainable weights, 0.07% of the model (rank 32, alpha 64).

Gradient checkpointing is switched on if the model supports it (it trades compute for memory during the update).

### 4.3 Rollouts: the model answers

```python
@torch.inference_mode()
def rollouts(qs):
    model.eval()
    prompts = [tok(sampling.prompt_text(tok, q["messages"]), add_special_tokens=False)["input_ids"] for q in qs]
    ...
    gen = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), generation_config=gen_cfg)
```

- The prompt is the dataset's system and user message, rendered with the model's chat template in thinking mode.
- `gen_cfg` asks for `group` answers per question (`num_return_sequences`), plain sampling at temperature 1.0 with no
  top-p or top-k cut, so the answers are true samples from the current policy, up to `max_new_tokens`.
- Prompts are left-padded so that generation continues from the real end of each prompt.

### 4.4 Reward: the verifiable part

```python
cut = next((k for k, t in enumerate(toks) if t in stops), None)
finished = cut is not None
comp = toks[:cut] if finished else [t for t in toks if t != tok.pad_token_id]
correct = bool(finished and Q.is_correct(Q.parse_answer(tok.decode(comp, skip_special_tokens=False)), q["gold"], q["kind"]))
reward = 1.0 - a.len_coef * len(comp) / a.max_new_tokens if correct else 0.0
```

- `Q.parse_answer` (in `cricket_qa.py`) takes the text after the closing `</think>` and reads the last
  `Final answer: <number>`. No closing marker or no such line means no answer.
- `Q.is_correct` compares with the exact answer computed from the match data: integers must match exactly, rates within
  0.011 (one unit of rounding at two decimals).
- An answer that hit the token cap is not finished and earns 0.
- A correct answer earns `1 - len_coef x tokens / cap`, so shorter correct answers earn more. With `len_coef 0.4` and a
  4,096-token cap, a 400-token correct answer earns 0.96 and a 4,000-token one 0.61.

No reasoning traces are used as targets anywhere: the only supervision is whether the final number is right.

### 4.5 Advantage: better or worse than the model's own average

```python
for g in range(len(qs)):
    grp = [r for r in recs if r["g"] == g]
    m = sum(r["reward"] for r in grp) / len(grp)
    for r in grp:
        r["adv"] = r["reward"] - m
```

Each answer is compared with the mean reward of the answers to the same question. If all 8 answers to a question earn
the same reward, their advantages are zero and the question teaches nothing; the log reports the share of "mixed"
questions for that reason.

### 4.6 Update: policy gradient through LoRA

```python
logits = ddp(input_ids=ids, attention_mask=mask).logits[:, :-1]
nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none",
                      ignore_index=-100).view(tgt.shape)
seq_nll = nll.sum(1) / (tgt != -100).sum(1).clamp(min=1)
adv = torch.tensor([r["adv"] for r in batch], device=dev, dtype=seq_nll.dtype)
loss = (adv * seq_nll).sum() / len(recs)        # minimise A * NLL  ==  maximise A * log-prob
loss.backward()
```

- The sampled answers are fed back through the model in training mode, `train_batch` sequences at a time, sorted by
  length. Labels cover only the generated tokens (and the stop token of finished answers), not the prompt.
- The loss is the advantage times the mean negative log-likelihood per token of that answer. Minimising it raises the
  probability of answers that beat their group mean and lowers it for those below.
- `ddp` is `DistributedDataParallel`: at every backward pass the gradients of all processes are averaged, so every
  process makes the same number of passes per step.
- After all passes: gradient clipping at norm 1.0, one AdamW step (no weight decay). The learning rate ramps linearly
  over the first `warmup_steps` steps and is then constant.

Each step uses fresh rollouts for exactly one update, so the policy that generated the answers is the policy being
updated.

### 4.7 Checkpoints, resume, deadline

- Every `save_steps` steps process 0 saves the adapter and the optimizer state to `ckpt/step-N` and keeps the last
  three.
- On start, the script loads the latest checkpoint if one exists and continues from that step.
- `rlvr.sbatch` passes a deadline 12 minutes before the job's time limit; when it passes, the script saves and stops
  cleanly. A second job with the same run name is chained behind the first to continue.
- When all steps are done it saves the final adapter to `adapter/` and writes `train_result.json`.

### 4.8 Metrics

Per step, summed over all processes with `all_reduce` and written by process 0 to `metrics.jsonl` and W&B: rollout
accuracy (overall and per question type), mean reward, finished and cut-off rates, mean tokens, share of mixed and
all-correct questions, loss, gradient norm, learning rate, seconds in rollouts and in the update.

## 5. What had to change for Nemotron

The script was written for Qwen. These are all the changes, each found by running it on Rhea:

| Change | Where | Why |
|---|---|---|
| Model class chosen from the model type | `sampling.load_model` | Qwen loads as a vision-language class; Nemotron is a plain causal language model |
| Pad token taken from the model config if the tokenizer has none | `sampling.load_tokenizer` | batching needs a pad id |
| Stop tokens read from `generation_config.json` as well as the ChatML markers | `sampling.eos_ids` | Nemotron stops on ids 2 and 11 |
| The actual stop token is appended to a finished answer for the update | `rlvr.py`, `rollouts` | it was hard-coded to Qwen's `<|im_end|>` |
| LoRA target pattern per model type, plus `--lora_targets` | `rlvr.py` | different layer names; first pattern used `backbone.layers`, the loaded model uses `model.layers` |
| Mamba `out_proj` removed from the targets | `rlvr.py` | PEFT raises an error for it on Mamba models |
| Gradient checkpointing wrapped in a try | `rlvr.py` | not every architecture implements it |
| cuDNN attention kernel switched off | `sampling.py`, for every script | it made generation 14 times slower (250 ms against 18 ms per step); see section 9 |
| No top-k in evaluation sampling for non-Qwen models | `sampling.sample` | the Nemotron card recommends temperature 1.0, top-p 0.95 |
| `PROFILE=nemotron` | `config.env` | own model path, run directory, logs, W&B project and Hugging Face repo |
| Model card text and usage snippet per model | `rlvr_finish.py` | it described Qwen layers and the Qwen class |

## 6. Settings for the Nemotron run

| Setting | Value | Note |
|---|---|---|
| GPUs | 32 (8 nodes x 4 Vera Rubin) | |
| Questions per GPU per step | 4 | 128 questions per step |
| Answers per question | 8 | 1,024 answers per step |
| Steps | 60 | 61,440 answers in total |
| Token cap in training | 4,096 | the base model averaged about 1,500 tokens in the smoke test and 16% of answers hit the cap in step 1 |
| Length coefficient | 0.4 | as in the Qwen3.6 run |
| Sequences per backward pass | 4 | smoke test with 2 peaked at 105 GB of 286 GB |
| Learning rate, warm-up | 1e-4, 10 steps | |
| LoRA rank, alpha | 32, 64 | |
| Checkpoint every | 10 steps | |

## 7. Jobs and commands

Everything runs as Slurm jobs through `rhea/submit.sh` with `PROFILE=nemotron`. From `rhea/` in Git Bash, with the
bridge open:

```bash
bin/rhsync
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr-smoke -- --max_new_tokens 4096 --len_coef 0.4 --train_batch 2'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr -N 8 -t 03:00:00 -n nemotron-rlvr-1124 -- --steps 60 --max_new_tokens 4096 --len_coef 0.4 --train_batch 4 --save_steps 10'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh eval -N 4 -t 02:00:00 -n nemotron-rlvr-1124 -- --batch_prompts 8'
bin/rx -t 60 'cd ~/clustercodes && PROFILE=nemotron bash submit.sh rlvr-finish -n nemotron-rlvr-1124'
```

`rlvr.sbatch` does three things around the script: starts `nvidia-smi` on every node (utilisation, memory, power and
temperature every 15 s, to `gpulog/<node>-<job>.csv`), starts the 32 processes inside the PyTorch container
(`pytorch-main-py3-devel-arm64.sqsh`) with the venv from Lustre, and stops the logger at the end.

| Output | Location |
|---|---|
| Adapter, checkpoints, `metrics.jsonl`, GPU logs | `/lustre/fsw/general_sa/bbalakreshna/nemotron-rhea/runs/<run-name>/` |
| Job logs | `/lustre/fsw/general_sa/bbalakreshna/logs/nemotron-rhea/` |
| W&B | project `nemotron35-cricket-rlvr-rhea` |
| Hugging Face (private) | `Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-RLVR-Rhea` |

## 8. Evaluation with NeMo Gym

This is the part that uses NeMo software. Files: `gym/cricket_qa/`, `gym_prepare.py`, `gym_eval.sbatch`,
`gym_summary.py`.

**Environment.** `gym/cricket_qa/app.py` is a NeMo Gym resources server. Its `verify` method receives the model's
response for one task and returns the reward:

```python
class CricketQAVerifier:
    async def verify(self, body: CricketQAVerifyRequest) -> CricketQAVerifyResponse:
        resp = body.response
        text = resp.output_text or ""
        finished = getattr(resp, "status", "completed") in (None, "completed") and getattr(resp, "incomplete_details", None) is None
        pred = Q.parse_answer("</think>" + text) if text.strip() else None
        correct = bool(finished and Q.is_correct(pred, body.gold, body.answer_kind))
        ...
        return CricketQAVerifyResponse(**body.model_dump(), reward=float(correct), correct=correct, finished=finished, pred=pred, ...)
```

It uses the same parser and checker as training (`cricket_rules.py` is a copy of `cricket_qa.py`). vLLM's reasoning
parser has already separated the thinking from the answer, so `output_text` is the answer part only.

**Tasks.** `gym_prepare.py` converts the 600 held-out questions into NeMo Gym task rows:
`{"responses_create_params": {"input": <system and user message>}, "qid", "qtype", "answer_kind", "gold"}`.

**Flow of one evaluation job (1 node).**

1. vLLM (Rubin image) serves the base model as `base` and the adapter as LoRA module `tuned`, with the model card's
   Mamba settings and `--reasoning-parser nemotron_v3`.
2. `gym eval run --resources-server cricket_qa --model-type vllm_model --split validation --agent cricket_qa_simple_agent --num-repeats 4 ...`
   starts the environment and agent servers, sends every question to vLLM, and passes each response to `verify`.
3. `gym_summary.py` computes accuracy, pass rate and tokens per question type and logs a table to W&B.

```bash
bin/rx -t 60 'source ~/clustercodes/nemotron/nemotron.env; sbatch -p "$PARTITION" -t 02:00:00 -o "$NT_LOGS/%x-%j.out" --export=ALL,CODE_DIR=$HOME/clustercodes,RUN_NAME=nemotron-rlvr-1124,REPEATS=4,CONC=128 ~/clustercodes/nemotron/gym_eval.sbatch'
```

Tested on 60 questions x 2 answers (job 129260): base model 91.67%.

## 9. Findings so far

- **cuDNN attention kernel.** On fresh prompts Nemotron generated at about 250 ms per step with it and 18 ms without
  (417 tokens/s at batch 8 on one GPU). It is switched off for all scripts.
- **Mamba kernels.** `mamba_ssm` and `causal_conv1d` are not installed, so `transformers` uses its reference PyTorch
  implementation. Generation is fast enough. The training update is slow on long answers: in the smoke test, up to about
  100 s per backward pass when answers reached the 4,096-token cap. Installing the kernels has not been tried.
- **NeMo Gym needs Python 3.13.** On the container's Python 3.14 its CLI fails; its venv is built with a standalone
  Python 3.13 on Lustre.
- **Smoke test (job 128883, 4 GPUs, 3 small steps):** rollout accuracy 75%, 97%, 91%; mean tokens 1,493, 956, 1,397;
  16%, 3% and 6% of answers cut off. With 32 answers per step these numbers are noisy and show only that the loop runs.

## 10. How this differs from GRPO-style training

| | This script | Typical GRPO |
|---|---|---|
| Baseline | mean reward of the answers to the same question | the same idea, usually also divided by the group's standard deviation |
| Updates per batch of rollouts | one | several, with an importance ratio and clipping |
| Regulariser | none | often a KL term to a reference policy |
| Loss per answer | advantage x mean per-token negative log-likelihood | advantage x per-token terms with the clipped ratio |
| Reward | exact-answer check with a length penalty | task-specific |

So this is a simple on-policy policy gradient with a per-question baseline, not a full GRPO implementation.

## 11. Limits

- The full run has not happened yet; everything about results is still to come.
- Only 0.07% of the weights train, and the routed experts and Mamba output projections cannot be adapted in this setup.
- The length penalty pushed the Qwen models to very short reasoning; whether that transfers beyond these question
  types is untested.
- The benchmark covers three of the dataset's six question types.
- NeMo RL was not evaluated as an alternative.
