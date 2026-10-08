# Nemotron-3.5-Lightning cricket SFT (EOS, H100)

Full SFT of `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on `Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full`
with Megatron-Bridge (NeMo 26.08.01 container). Result: public model
[Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-Reasoning](https://huggingface.co/Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-Reasoning).

Code runs from `~/clustercodes/nemotron-cricket` (synced by `eos/bin/esync`); data, checkpoints, logs and caches
live under `/lustre/fsw/general_sa/bbalakreshna` (`nemotron-cricket/`, `hf_cache/`, `logs/nemotron-cricket/`).
Secrets: `$LUSTRE_DIR/secrets.env` (HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY).

## Pipeline (run on the EOS login node)

```bash
python3 ~/clustercodes/nemotron-cricket/prep_data.py      # needs PROJ, HF_HOME (source config.env, export PROJ)
S=~/clustercodes/nemotron-cricket/submit.sh
PARTITION=interactive bash $S import                       # HF -> Megatron (3 min)
PARTITION=interactive bash $S bench 2 tp2ep8-sel --tp 2 --ep 8 --recompute selective   # also packs the data once
PARTITION=interactive TIME=02:00:00 EXIT_MINS=100 bash $S train 2 sft-full-e1 --tp 2 --ep 8 --recompute selective \
    --save-interval 150 --eval-interval 100 --eval-iters 5   # chain a 2nd copy with DEPEND=<jobid> to resume
PARTITION=interactive bash $S export sft-full-e1           # latest checkpoint -> runs/sft-full-e1/hf
# then copy the base model's config.json / generation_config.json / tokenizer_config.json / special_tokens_map.json
# / LICENSE into runs/sft-full-e1/hf (the export writes max_position_embeddings=4096, num_nextn_predict_layers=2)
PARTITION=interactive bash $S eval base; PARTITION=interactive bash $S eval sft-full-e1
python3 ~/clustercodes/nemotron-cricket/add_results.py $PROJ sft-full-e1
PARTITION=interactive bash $S push sft-full-e1
```

## Cluster config (benchmarked, 4K packed sequences, global batch 128)

| config | result |
|---|---|
| 2 nodes, TP2 + SP, EP8, selective recompute | **9.35 s/step, 93 TFLOP/s/GPU, 65 GB peak** (used) |
| 2 nodes, TP1, EP8, no recompute | OOM (74 GB after step 1) |
| 2 nodes, TP1, EP8, selective recompute | OOM |

`interactive` (2 h, max 2 nodes / 2 jobs per user, 8-node shared cap) started jobs within minutes; `batch` (4 h)
was booked ~8 h out; `backfill` is preemptible (REQUEUE) and was slow on 2026-10-08.

## Results (sft-full-e1, 573 steps, 1 h 37 min on 16 H100)

| | base | fine-tuned |
|---|---|---|
| benchmark accuracy (600 q x 4 samples) | 89.6% | **97.0%** |
| chase_rate / run_rate / milestone | 77.8 / 91.8 / 99.4% | 94.1 / 97.1 / 99.6% |
| mean generated tokens | 1,700 | 519 |
| truncated at 6,144 tokens | 4.9% | 0% |

Validation loss 0.394 (step 100) -> 0.371 (final). W&B: https://wandb.ai/balabala76/nemotron35-cricket-reasoning/runs/8qh82u2z

## Gotchas hit

- `convert.sh` re-resolves deps with `uv` (pulls huggingface-hub 2.x, breaks transformers): call
  `run_conversion.py` with `/opt/venv/bin/python -m torch.distributed.run` (no `--executor/--gpus-per-node`).
- `~/.bashrc` (env.sh) is re-read inside the container and resets `CODE_DIR`: job.sbatch pins it.
- Offline packing with >1 tokenizer worker shares every sample via file_system shm -> ENOMEM; use 1 worker
  (single-process packing took 65 min once; cached next to the tokenizer snapshot, `*_pad_seq_to_mult2_sft_*`).
- Triton / inductor caches on Lustre race across ranks ("CUDA driver error: file not found"): node-local `/tmp`.
