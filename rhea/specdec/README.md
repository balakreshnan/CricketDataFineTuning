# Speculative decoding on Rhea

A small **draft** model proposes several tokens; the large **target** model checks them all in one forward pass and
keeps the ones it agrees with. With speculative sampling the output distribution is the target's own, so accuracy is
unchanged; the gain is speed, and it depends on how often the draft's tokens are accepted.

Everything follows the same rules as the rest of `rhea/`: code in `~/clustercodes/specdec` on the cluster, all
weights, records and logs on Lustre, every cluster command a Slurm job, metrics in W&B, results in a private
Hugging Face repo.

## Models and what each is for

| Model | Role here | Where it lives |
|---|---|---|
| `Qwen/Qwen3.8-27B` (dense, 27B) | **Target**: the model whose answers we want, only faster. Default target. | `models/Qwen3.8-27B` on Lustre |
| RLVR adapter `cricket-rlvr-1114` (LoRA, 870 MB) | Merged into the target, so the target is the cricket-tuned model (99.96% on the benchmark, about 120 tokens per answer). Set `TARGET_ADAPTER=` to test the untuned base model instead. | `cricket-rhea/runs/cricket-rlvr-1114/adapter`; Hugging Face `Balab2021/Qwen3.8-27B-Cricket-Reasoning-RLVR-120steps` |
| `Qwen/Qwen3.6-35B-A3B` (mixture of experts) plus its RLVR adapter | **Alternative target** with `PROFILE=qwen36`. Not benchmarked yet. | `models/Qwen3.6-35B-A3B`, `cricket-rhea/runs/qwen36-rlvr-1113/adapter` |
| `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B` | **Drafts**: small models that guess the next few tokens for the target to check. Used as downloaded, not fine-tuned. | `models/Qwen3.5-*` on Lustre |
| Dataset `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` | Questions with exact answers. Its eval split (600 held-out questions) is the benchmark; its train split would be used to fine-tune a draft in stage 2. | `datasets/CricketData-T20-Reasoning-Qwen3.8-Full` on Lustre |

The target decides what is written; the draft only affects speed. A wrong draft guess is discarded and replaced by the
target's own token.

## Which small models can be drafts

A draft must share the target's tokenizer and vocabulary exactly. Both targets used so far (Qwen3.8-27B and
Qwen3.6-35B-A3B) use a 248,320-token vocabulary and the `qwen3_5` layer design. The Qwen3.5 small models match
(checked from each model's `config.json` on Hugging Face, 23 Nov 2026):

| Draft | Parameters | Layers | Hidden size | Weights | Share of 27B target | Licence |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 0.87B | 24 | 1,024 | 1.7 GB | 3% | Apache-2.0 |
| `Qwen/Qwen3.5-2B` | 2.27B | 24 | 2,048 | 4.5 GB | 8% | Apache-2.0 |
| `Qwen/Qwen3.5-4B` | 4.66B | 32 | 2,560 | 9.3 GB | 17% | Apache-2.0 |
| `Qwen/Qwen3.5-9B` | 9.65B | 32 | 4,096 | 19.3 GB | 36% | Apache-2.0 |

- A smaller draft is cheaper per proposed token but is accepted less often; a larger draft is the reverse.
  Which one wins has to be measured: `submit.sh bench` compares all four.
- No small model exists in the Qwen3.6 or Qwen3.8 families on Hugging Face (only 27B, 35B-A3B and larger), so the
  drafts are one or two generations older than the targets. That lowers acceptance.
- The RLVR-tuned targets write short answers (about 120 to 150 tokens); the base drafts write long ones. A base draft
  will therefore disagree with a tuned target more often. Stage 2 below addresses that.
- Not usable as drafts: `Qwen3.5-*-Base` variants (no chat tuning), anything with a different tokenizer.

## Status (23 Nov 2026)

Working end to end: draft download, benchmark, W&B logging, private Hugging Face upload. **Result so far: speculative
decoding with these drafts is slower than the target alone.**

Run `specdec-qwen38-rlvr-1123` (job 120765, 2 nodes, 8 GPUs, 40 min): RLVR-tuned Qwen3.8-27B target, 200 questions,
sampling, one answer per question per configuration.

| Draft | Draft tokens per round | Tokens/s | Against target alone | Draft tokens accepted | Tokens per target pass |
|---|---|---|---|---|---|
| none (target alone) | | 36 | 1.00x | | 1.00 |
| Qwen3.5-0.8B | 2 / 4 / 8 | 24 / 22 / 17 | 0.66x / 0.60x / 0.46x | 43% / 29% / 17% | 1.82 / 2.11 / 2.27 |
| Qwen3.5-2B | 2 / 4 / 8 | 25 / 24 / 18 | 0.70x / 0.66x / 0.51x | 49% / 35% / 20% | 1.93 / 2.32 / 2.51 |
| Qwen3.5-4B | 2 / 4 / 8 | 24 / 22 / 17 | 0.66x / 0.61x / 0.47x | 55% / 40% / 25% | 2.04 / 2.52 / 2.83 |
| Qwen3.5-9B | 2 / 4 / 8 | 24 / 23 / 18 | 0.68x / 0.64x / 0.50x | 57% / 42% / 27% | 2.10 / 2.60 / 2.99 |

- Accuracy was 100% (200 of 200) in every configuration, and answers averaged 119 to 120 tokens in all of them.
- The target alone through this loop (36 tokens/s) matches `generate()` (35 tokens/s), so the loop is a fair baseline.
- Best case is Qwen3.5-2B with 2 draft tokens per round: 0.70x, still 30% slower than no draft.
- Why (interpretation, not separately measured): one prompt at a time, a pass through the 27B target takes about
  28 ms, and that time is mostly per-layer overhead, not GPU compute. A draft has 24 to 32 layers against the target's
  64, so a draft pass costs roughly a third to a half of a target pass, not a few percent. Each round also copies both
  caches for rollback. With only 2 to 3 tokens gained per target pass, the draft's cost outweighs the saving.

Two things found on the way:

- `transformers` 5.22.2 refuses assisted generation for these models ("not supported with stateful models"): their
  linear-attention layers keep a recurrent state that cannot be rewound. `specdec_bench.py` has its own decoding loop
  that snapshots and restores the cache. With greedy decoding its answers match the target-only answers.
- The cuDNN attention kernel costs about 5 ms per call the first time it sees a sequence length. Decoding changes the
  length every token, so a fresh prompt ran at 8 tokens/s instead of about 40. `specdec_bench.py` switches it off
  (`torch.backends.cuda.enable_cudnn_sdp(False)`). The RLVR rollouts and benchmarks in `rhea/` do not switch it off;
  whether they pay the same cost has not been measured.

What could still make it pay off, none of it tried: a draft fine-tuned to imitate the tuned target (higher
acceptance), compiled forward passes for draft and target (lower per-pass overhead), a much shallower draft, or a
target whose answers are long (the base model writes about 730 tokens per answer, the tuned one about 120).

## Stages

1. **Download the drafts** to `$LUSTRE_DIR/models/<name>` (35 GB in total).
2. **Smoke test**: 1 node, 8 questions. Proves that target and drafts run together in this software stack.
3. **Benchmark**: target alone vs target + each draft, on the eval split of
   `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` (the same 600 held-out questions as the RLVR benchmark).
4. **Stage 2, not scripted yet**: fine-tune the best draft on the same dataset (LoRA, reusing `../train.py`), or on
   the tuned target's own answers, so that it imitates the target; rerun the benchmark; upload the draft adapter to a
   private Hugging Face model repo.

## Commands

On the laptop, from `rhea/` in Git Bash, with the bridge open:

```bash
bin/rhsync
bin/rx -t 60 'bash ~/clustercodes/specdec/submit.sh drafts'
bin/rx -t 60 'bash ~/clustercodes/specdec/submit.sh smoke'
bin/rx -t 60 'PUSH=1 bash ~/clustercodes/specdec/submit.sh bench -N 2 -t 02:00:00 -n specdec-qwen38-rlvr'
```

Options (environment variables before `bash`, benchmark arguments after `--`):

| Setting | Meaning | Default |
|---|---|---|
| `PROFILE=qwen36` | target is Qwen3.6-35B-A3B instead of Qwen3.8-27B | Qwen3.8-27B |
| `TARGET_ADAPTER=<dir>` | LoRA adapter merged into the target; `TARGET_ADAPTER=` for the base model | the RLVR adapter of that target |
| `DRAFTS="Qwen3.5-2B Qwen3.5-4B"` | drafts to compare | all four |
| `PUSH=1` | upload results to the private dataset repo when the benchmark ends | off |
| `-- --limit 200` | number of questions (0 = all 600) | 200 |
| `-- --gammas 4:8:16` | draft tokens proposed per round, each value is its own configuration | 8 |
| `-- --max_new_tokens 1024` | answer cap | 1024 |
| `-- --hf_reference` | also time the target through `generate()` | off |
| `-- --greedy` | greedy decoding instead of the model card's thinking-mode sampling | sampling |

## What is measured

Per configuration: tokens per second, speed-up over the target alone, seconds per answer, accuracy, answer length,
the share of draft tokens accepted, and **tokens per target pass** (about 1.0 without speculation, at most
`gamma + 1`). GPU utilisation, memory and power are logged every 15 s on every node.

The benchmark runs one prompt at a time. It measures
single-request latency, not batched serving throughput.

## Where things go

| What | Where |
|---|---|
| Draft weights | `/lustre/fsw/general_sa/bbalakreshna/models/Qwen3.5-*` |
| Records, summary, GPU logs | `/lustre/fsw/general_sa/bbalakreshna/cricket-rhea/specdec/<run-name>/` |
| Job logs | `/lustre/fsw/general_sa/bbalakreshna/logs/cricket-rhea/general_sa-cricket-rhea.specdec*-<jobid>.out` |
| W&B | project `cricket-specdec-rhea`, one run per benchmark, results table plus per-configuration summary values |
| Hugging Face | private dataset repo `Balab2021/Cricket-SpecDec-Rhea`, one folder per run (summary, records, card) |

## Files

| File | Purpose |
|---|---|
| `specdec.env` | settings: draft list, target adapter, W&B project, Hugging Face repo |
| `submit.sh` | submits the stages as Slurm jobs |
| `download_drafts.sbatch`, `download_drafts.py` | download drafts, check vocabulary against the target |
| `specdec.sbatch`, `specdec_bench.py` | the benchmark, one process per GPU |
| `specdec_push.py` | results card and upload to Hugging Face |
