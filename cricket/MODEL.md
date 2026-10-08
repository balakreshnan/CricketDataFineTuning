# Fine-tuned model: Qwen3.8-27B Cricket Reasoning (LoRA)

LoRA adapters that make `Qwen/Qwen3.8-27B` solve cricket match-situation arithmetic (run rates, chase rates,
milestones, projections, strike rates) with **shorter, more reliable step-by-step reasoning**. They were trained by
**self-distillation**: the base model answered each question several times in thinking mode, every answer was checked
against an exact value computed from the ball-by-ball data, and the shortest correct reasoning trace became the
training target.

Companion documents: [`DATASET.md`](DATASET.md) (source data), [`ARCHITECTURE.md`](ARCHITECTURE.md) (pipeline
design and diagrams), [`README.md`](README.md) (runbook, all measurements).

---

## 1. Links

### Models

| model | Hugging Face | revision | trained on | recommended |
|---|---|---|---|---|
| **Full-dataset adapter** (Part II) | **https://huggingface.co/Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA** | `5f21ce653ee676011e0ebaa2477a08dab750c837` | 457,689 rows, 1 epoch, 32 GPUs | **yes** |
| Reference adapter (Part I) | https://huggingface.co/Balab2021/Qwen3.8-27B-Cricket-Reasoning-LoRA | `40af7e4f6122ac69372d7fa36a0a00f0202ad761` | 4,410 rows, 2 epochs, 16 GPUs | baseline / comparison |
| Base model | https://huggingface.co/Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | - | - |

Both adapter repos are **public** (verified 2026-10-07). Each contains the LoRA adapter (`adapter_model.safetensors`,
870 MB), tokenizer files and chat template, a model card with the results, `eval_report/` (report, metrics, plots),
**and a `merged/` folder with the full merged bf16 model** (12 safetensors shards, ~54.7 GB) that loads directly in
vLLM or transformers without PEFT. (`merged/` was uploaded because the merge step writes it inside the adapter
folder; the pipeline now skips it unless `--push_merged` is given.)

### Datasets

| dataset | Hugging Face | revision | used for |
|---|---|---|---|
| **Full one-to-one reasoning dataset** | **https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full** | `ad669766ef046c8ccc5330d077c77118b692ec00` | training the full-dataset adapter (`train`), held-out rows (`test`), benchmark (`eval`) |
| Reference reasoning dataset | https://huggingface.co/datasets/Balab2021/CricketData-T20-Reasoning-Qwen3.8 | `efa295283fc9b2a012b9fa06a77bf393e35de212` | training the reference adapter |
| Source data | https://huggingface.co/datasets/Valarmathy/CricketData | `a2518e9db3bd6e745dabc645a423eed6f053207f` | 425,119 T20I deliveries (CC0) the questions are built from |

Training and evaluation logs: W&B project `balabala76/qwen38-cricket-reasoning`, runs
`general_sa-cricket.sft-full-10070553` (full) and `general_sa-cricket.sft-696652` (reference).

---

## 2. What the model does

**Task.** Given a T20 International match situation (teams, venue, score after N overs in cricket notation, the two
batters' scores, the bowler) and a question, reason step by step and end with `Final answer: <number>`.

Question types (answers are exact numbers):

| type | example question | answer |
|---|---|---|
| `run_rate` | "What is Bulgaria's current run rate in runs per over? Round to 2 decimal places." | 5.67 |
| `chase_rate` | "What run rate do Nigeria need over the rest of their 20 overs to reach the target?" | e.g. 6.11 |
| `milestone` | "How many more balls must KC D'Souza face to reach 50 at their own current strike rate? Round up." | 42 |
| `projection` | "…keep scoring at exactly that rate for the rest of their 20 overs, what total will they finish on?" | e.g. 120 |
| `strike_rate` | "What is AJ Finch's current strike rate (runs per 100 balls faced)?" | 33.33 |
| `legal_balls` | "The innings has opened with a wide… How many legal deliveries are still to be bowled?" | 120 |

The key skills are reading the match context (ignoring distractors), converting cricket notation (14.3 overs = 87
balls), and multi-step arithmetic with the requested rounding.

**Intended use:** research and demonstration of reasoning distillation and efficient fine-tuning on Vera Rubin; a
compact, verifiable benchmark for numeric reasoning in a sports domain.

**Out of scope:** predicting match outcomes, betting, player evaluation, or any question whose answer is not
computable from the given numbers; general chat (the adapters were trained only on this task - use the base model for
anything else).

---

## 3. Prompt format and decoding

Use the Qwen chat template **in thinking mode** with the system prompt used in training:

```
system: You are a cricket analyst. Work through the problem step by step, then give the final answer on its own
        line as 'Final answer: <number>'.
user:   T20 International: Bulgaria v Serbia at Lisicji Jarak Cricket Ground, 2022-07-08.
        Bulgaria are batting first. After 6.0 overs (cricket notation: overs.balls) they are 34/2. KC D'Souza is on 8
        off 8 balls and H Lakov is on 13 off 17. A Mene-Ejegi bowled the last delivery.

        What is Bulgaria's current run rate in runs per over? Round to 2 decimal places.
```

* `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)`
* sampling as in training and evaluation: **temperature 1.0, top_p 0.95, top_k 20**, `max_tokens` up to 6,144
  (the fine-tuned model averages ~460 tokens; greedy decoding is not recommended for Qwen thinking mode)
* parse the answer from the text after the closing think tag: the last `Final answer: <number>` line
* write scores with runs ≤ wickets explicitly ("1/3 (1 run for the loss of 3 wickets)") - the model may otherwise read
  them in the Australian wickets/runs order

Typical fine-tuned output for the example (reasoning shortened):

```
<think> ... 34 runs in 6.0 overs = 36 balls. Current run rate = 34 / 6 = 5.666... = 5.67 ... </think>

Bulgaria have scored 34 runs in 6.0 overs.
Current run rate = 34 ÷ 6 = 5.666... ≈ 5.67 runs per over.

Final answer: 5.67
```

---

## 4. Usage

### 4.1 transformers + PEFT (adapter)

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3.8-27B"
adapter_id = "Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA"
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id, revision="5f21ce653ee676011e0ebaa2477a08dab750c837")

messages = [{"role": "system", "content": "You are a cricket analyst. Work through the problem step by step, then give "
                                          "the final answer on its own line as 'Final answer: <number>'."},
            {"role": "user", "content": "<match context>\n\n<question>"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=2048, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
```

Requirements: transformers ≥ 5.8 (architecture `qwen3_5`), peft ≥ 0.18; one GPU with ≥ ~60 GB for bf16. The
hybrid Gated-DeltaNet layers are much faster with `flash-linear-attention` installed.

### 4.2 vLLM (merged weights, fastest)

```bash
huggingface-cli download Balab2021/Qwen3.8-27B-Cricket-Reasoning-Full-LoRA --include "merged/*" --local-dir ./cricket-full
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="./cricket-full/merged", dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"image": 0, "video": 0})
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
out = llm.generate([prompt], SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=6144, n=1))
print(out[0].outputs[0].text)
```

This is exactly how the evaluation runs (`gen_vllm.py`), one engine per GPU. On Rubin (sm_107) use the
`dl/dgx/vllm:rubin-py3-devel` image - NGC `vllm:26.08` cannot compile its Triton kernels for this GPU. Serving the
LoRA on top of the base model in vLLM was not tested.

---

## 5. Training

| | full-dataset adapter | reference adapter |
|---|---|---|
| data | `CricketData-T20-Reasoning-Qwen3.8-Full` train: 458,702 rows → 457,689 train + 1,000 dev (13 rows over 6,144 tokens dropped) | `CricketData-T20-Reasoning-Qwen3.8`: 4,500 rows → 4,410 train + 90 dev |
| questions | 6 types, every one of the 425,119 source deliveries covered | 3 types, 1,500 sampled deliveries |
| targets | the base model's own shortest verified-correct trace (of 4 samples) - self-distillation / rejection-sampling fine-tuning | same |
| loss | cross-entropy on the completion only (reasoning + answer + `<|im_end|>`), prompt masked | same |
| method | LoRA r 32, α 64, dropout 0.05 on `q,k,v,o_proj`, `linear_attn.in_proj_qkv / in_proj_z / out_proj`, `mlp.gate/up/down_proj` of the language model (vision tower frozen) - 217.6 M trainable params (0.79%) | same |
| schedule | 1 epoch, 1,788 steps, global batch 256 (8 per GPU), lr 2e-4 cosine, warmup 3%, AdamW, bf16, gradient checkpointing, length-grouped batches | 2 epochs, 138 steps, global batch 64 (2 × accum 2), lr 1e-4 cosine, warmup 5% |
| hardware | 8 nodes × 4 NVIDIA Rubin (279.5 GiB each) = 32 GPUs, DDP over multi-node NVLink, Hecate `batch-spx` | 4 nodes × 4 = 16 GPUs, `batch-xdr` |
| time | ≈ 4.0 h of training in two allocations (43 min to step 272 + 196 min resumed) ≈ 128 GPU-hours | 20.2 min ≈ 5.4 GPU-hours |
| loss | train ≈ 0.330 → 0.311, dev 0.326 → 0.318 (decreasing the whole epoch) | train 0.32 → 0.31, dev ≈ 0.311 (flat) |
| software | PyTorch 2.15 (NVIDIA build), transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, CUDA 13.4 | same |

Distillation of the full dataset: 2,002,672 samples (1.58 B generated tokens) with vLLM on 32 GPUs in ≈ 1.9 h; base
model accuracy 97.3% per sample, 100% of questions solved (after clarifying 87 ambiguous scores).

---

## 6. Evaluation

**Benchmark:** 600 questions (200 run_rate, 200 chase_rate, 200 milestone) from **177 matches never used in
training** (the `eval` split of the full dataset). Each question is answered **4 times** in thinking mode with
identical vLLM sampling and seeds for every model; answers are checked against the exact values (rates within 0.011,
integers exactly).

| question type | base Qwen3.8-27B | reference adapter (4.5K) | **full-dataset adapter (500K)** |
|---|---|---|---|
| run_rate accuracy | 98.38% | 99.00% | **99.00%** |
| chase_rate accuracy | 94.38% | 92.25% | **94.62%** |
| milestone accuracy | 99.12% | 99.88% | **100.00%** |
| **overall accuracy** | **97.29%** | 97.04% | **97.88%** |
| pass@4 (any of 4 correct) | 100% | 100% | 100% |
| mean output tokens | 748 | 482 (−36%) | **458 (−39%)** |
| run_rate / chase_rate / milestone tokens | 403 / 939 / 904 | 280 / 610 / 557 | **266 / 586 / 522** |
| truncated at the token limit | 0.38% | 0% | **0%** |

Base-model numbers are from the full run's eval (97.29%); the reference run's own base eval measured 97.25% - the
difference is sampling noise (≈ ±0.15-0.3 pts between identical setups).

**Reading the results**

* **Efficiency is the main gain:** the full-dataset adapter reasons in 39% fewer tokens with no truncated answers,
  i.e. ~40% less inference compute per question at equal or better accuracy.
* **Accuracy:** +0.58 pts overall vs base (≈ +1.2 standard errors over 2,400 samples) - small but consistent;
  milestone reaches 100%. The base model is already at ~97% on these types, so there is little headroom.
* **More data fixed the reference adapter's weak spot:** its chase_rate dipped to 92.25% (errors at mid-over positions
  needing overs→balls conversion); the full-dataset adapter is at 94.62%, above base.
* Per question, 44 of 600 became more reliable and 31 less reliable compared with base.

Reports, metrics and plots are in each model repo's `eval_report/` (and in
`pulled/cricket/runs/<run>/` locally).

---

## 7. Limitations

* **Narrow task:** trained only on synthetic numeric questions about T20I match states; not evaluated on other
  cricket formats, other sports or general reasoning, where behaviour may change.
* **Self-distilled:** the targets are the base model's own outputs, so the adapter mostly makes existing behaviour
  shorter and more consistent; it does not add knowledge the base model lacks.
* **Benchmark coverage:** the benchmark has the three original question types; the newer types (projection,
  strike_rate, legal_balls) were checked only during data generation (97.9-100% base accuracy), not in a held-out
  base-vs-fine-tuned comparison.
* **Data conventions:** inherits the source's quirks (e.g. balls-faced counting includes some wides) - questions show
  the source's numbers, so answers follow them (DATASET.md §6).
* **Notation:** ambiguous scores (runs ≤ wickets) and unusual phrasing can still trigger long deliberation.
* **Size:** the merged model is ~55 GB in bf16 and needs a GPU with ≥ ~60 GB.

## 8. Reproducing

Everything was produced by the batch pipeline in this folder on the Hecate cluster:

```bash
C=/lustre/fsw/general_sa/bbalakreshna/clustercodes/code/cricket/submit.sh
bash $C prepare-full && bash $C pilot-full && bash $C generate-full && bash $C push-full     # dataset (README §12)
PARTITION=batch-xdr,batch-spx GEN_NAME=train-full TRAIN_TIME=04:59:00 TRAIN_MIN=280 SAVE_STEPS=100 \
  bash $C train-full                                                                         # adapter (README §13)
```

Pinned inputs: source revision `a2518e9d…`, base model revision `1d4bf0f2…`, container images in
`clustercodes/containers/` (README §4). Seeds: data split 0, sampling = shard index, SFT 42.

## 9. License and attribution

* Base model `Qwen/Qwen3.8-27B`: **Apache-2.0** (its `LICENSE` is included in `merged/`). The adapters and merged
  weights are derivatives and follow the base model's license; the repo cards do not declare a separate license.
* Source data `Valarmathy/CricketData`: **CC0-1.0**; the reasoning datasets are published as CC0-1.0.
* Match data originates from public T20I scorecards (match ids are ESPNcricinfo ids).
