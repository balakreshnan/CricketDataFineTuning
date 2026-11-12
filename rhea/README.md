# Rhea: Qwen3.8-27B reasoning LoRA on the cricket dataset

All Rhea code lives here. Code runs from `$CODE_DIR` on the cluster (synced by `bin/rhsync`); **datasets, model weights,
HF/pip/torch caches, venv, checkpoints and logs all live on Lustre** (`$LUSTRE_DIR`, see `env.sh`). Every command that
touches the cluster runs as a Slurm job (`setup.sbatch`, `train.sbatch`); the login node only submits and reads.

- Model: `Qwen/Qwen3.8-27B` -> `$LUSTRE_DIR/models/Qwen3.8-27B`
- Dataset: `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full` (private) -> `$LUSTRE_DIR/datasets/...`
- Runs/checkpoints: `$LUSTRE_DIR/cricket-rhea/runs/<run-name>/{ckpt,adapter}`; Slurm logs `$LUSTRE_DIR/logs/cricket-rhea/`
- Recipe (same as the Hecate run, which took accuracy 97.29 -> 97.88% and cut reasoning tokens 39%): LoRA r=32/alpha=64 on
  attention, Gated-DeltaNet and MLP projections of the language model, lr 1e-4 cosine, bf16, max_len 6144, DDP.
- The public dataset has no `completion_text`; `train.py` rebuilds the target as `reasoning + "\n</think>\n\n" + response`
  and checks that the prompt ends with `<think>\n`.

## Steps

```powershell
# 1. edit rhea/ssh_config (login host/user) once, then open the relay and finish MFA; leave the window open
& 'C:\Program Files\Git\bin\bash.exe' rhea/bin/rhea-bridge
# 2. in another terminal: set the HF token + W&B key (typed hidden, stored 0600 on Lustre)
& 'C:\Program Files\Git\bin\bash.exe' rhea/bin/set-secrets
```

Then (Claude runs these): `rhea/bin/rhsync`, `rhea/bin/rx 'bash ~/clustercodes/discover.sh'` (partitions, GPUs, idle nodes,
Lustre path), set `PARTITION/ACCOUNT/GPUS_PER_NODE` in `config.env`, then
`submit.sh setup` -> `submit.sh smoke` -> `submit.sh train -N <n> -t <limit>`. A re-submit of the same `-n run-name`
resumes from the last checkpoint (jobs stop 7 minutes before the wall limit after saving).

Notes: 27B in bf16 is ~54 GB per GPU with plain DDP, so this needs >= 80 GB GPUs (96+ GB comfortable at batch 2).
On smaller GPUs, use `-- --per_device_batch 1 --grad_accum 4` or add FSDP (not implemented here).

## Is speculative decoding usable in fine-tuning?

**Not for the SFT/LoRA step itself.** Speculative decoding speeds up autoregressive *decoding*, where each token needs a
sequential, memory-bound forward pass and a small draft model lets the big one verify several tokens per pass. SFT is
teacher-forced: the whole 27B sequence goes through one parallel forward/backward, which is already compute-bound and
has no sequential token loop to shorten. Adding a draft model would only add memory and no speedup.

Where it does pay off around this project:

1. **Evaluation and any generation** (benchmark of base vs adapter, rejection sampling, RL / on-policy distillation).
   Decoding is the bottleneck there. In vLLM: `speculative_config={"method":"ngram", ...}` (or suffix decoding) is a good
   free first try because cricket reasoning keeps copying numbers from the prompt; MTP or EAGLE-3 heads give more if the
   checkpoint ships them (check `config.json`/weights for `mtp` before relying on it; not verified for Qwen3.8).
   It is lossless, so accuracy is unchanged; merge the LoRA first (or the acceptance rate drops because the draft
   was fit to the base distribution).
2. **Training the drafter itself** (EAGLE-3/MTP head on these 500K reasoning traces, afterwards) is a separate job whose
   only benefit is faster serving of the fine-tuned model.

Recommendation: do plain LoRA now, add vLLM + n-gram speculation to the eval stage afterwards.
