"""Full SFT of Nemotron-3.5-Lightning-30B-A3B on the Olympics reasoning set with Megatron-Bridge (GB200).

Starts from the container's verified `nemotron_3_5_lightning_sft_config` recipe, swaps in our chat JSONL
(offline-packed to --seq tokens, loss on the assistant turn only) and sets parallelism / batch / logging from CLI.
Launched once per GPU by job.sbatch (torchrun). Two modes:
  --bench N : N steps, no checkpoint/eval, step time logged to W&B group "bench" (picks the cluster config)
  default   : full run; resumes from --save if it already holds a checkpoint, else starts from --pretrained
"""
import argparse
import glob
import json
import math
import os

import torch
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, GPTSFTDatasetConfig
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_5_lightning_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.gpt_step import forward_step


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--data", required=True, help="dir with training.jsonl / validation.jsonl")
    p.add_argument("--hf-model", required=True, help="local HF snapshot (tokenizer + chat template)")
    p.add_argument("--pretrained", required=True, help="imported Megatron checkpoint (iter_0000000 parent)")
    p.add_argument("--save", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--etp", type=int, default=1)
    p.add_argument("--recompute", choices=["none", "selective"], default="selective")
    p.add_argument("--dispatcher", choices=["hybridep", "alltoall"], default="hybridep",
                   help="MoE token dispatcher: the recipe's DeepEP rejects GB200; HybridEP targets GB200 NVL72")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--gbs", type=int, default=32)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--min-lr", type=float, default=5e-7)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--train-iters", type=int, default=0, help="override; 0 = epochs x packed sequences / gbs")
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=5)
    p.add_argument("--exit-mins", type=int, default=0, help="save + exit after this many minutes (wall-time limit)")
    p.add_argument("--bench", type=int, default=0)
    return p.parse_args()


def packing_tokenizer(data_dir, hf_model):
    """Per-dataset tokenizer path for offline packing.

    Megatron-Bridge caches the packed set next to the tokenizer as <tokenizer>_pad_seq_to_mult<k>_sft_<hash>/ and the
    hash does NOT depend on the data, so packing two datasets with the same tokenizer path silently reuses the first
    one's packed set (bit us: the real run picked up the 8.7K-row bench pack). A symlink inside the data dir gives
    every dataset its own cache, stored next to the data.
    """
    link = os.path.join(data_dir, "tokenizer")
    try:
        os.symlink(hf_model, link)
    except FileExistsError:
        pass  # every rank tries; the first one wins
    return link


def packed_train_sequences(tokenizer, seq):
    """Packed training-sequence count for this dataset (None until it has been packed once)."""
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(f"{tokenizer}_pad_seq_to_mult*_sft_*/training_{seq}.idx.parquet"),
                       key=os.path.getmtime, reverse=True):
        return pq.ParquetFile(path).metadata.num_rows, path
    return None, None


def main():
    a = parse()
    pack_tok = packing_tokenizer(a.data, a.hf_model)
    cfg = nemotron_3_5_lightning_sft_config()
    m = cfg.model

    # parallelism: TP/SP for the dense + Mamba parts, EP for the 128 experts; DP fills the rest
    m.tensor_model_parallel_size = a.tp
    m.sequence_parallel = a.tp > 1
    m.expert_model_parallel_size = a.ep
    m.expert_tensor_parallel_size = a.etp
    m.pipeline_model_parallel_size = 1
    m.context_parallel_size = 1
    m.seq_length = a.seq
    if a.recompute == "selective":
        m.recompute_granularity = "selective"
        m.recompute_modules = ["moe", "layernorm", "core_attn", "mlp"]
    else:
        m.recompute_granularity = None
        m.recompute_modules = None
    m.recompute_method = None
    m.recompute_num_layers = None
    m.hf_model_id = a.hf_model
    if a.dispatcher == "alltoall":
        m.moe_token_dispatcher_type = "alltoall"
        m.moe_flex_dispatcher_backend = None
    else:
        apply_flex_dispatcher_backend(m, a.dispatcher)

    cfg.tokenizer.tokenizer_model = a.hf_model
    cfg.tokenizer.hf_tokenizer_kwargs = {}

    cfg.dataset = GPTSFTDatasetConfig(
        seq_length=a.seq,
        dataset_root=a.data,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        enable_offline_packing=True,
        offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=a.seq, pad_seq_to_mult=max(2, a.tp),
                                                  tokenizer_model_name=pack_tok,
                                                  # >1 shares every sample via file_system shm: 458K mmaps -> ENOMEM
                                                  num_tokenizer_workers=1),
        dataset_kwargs={"pad_to_max_length": True},
        do_validation=not a.bench,
        do_test=False,
        seed=1234,
        dataloader_type="batch",
        num_workers=2,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
    )

    t = cfg.train
    t.global_batch_size = a.gbs
    t.micro_batch_size = a.mbs
    t.manual_gc = True
    t.manual_gc_interval = 100
    t.empty_unused_memory_level = 0
    n, packed = packed_train_sequences(pack_tok, a.seq)
    if n is not None and int(os.environ.get("RANK", "0")) == 0:
        print(f"[train.py] {n} packed training sequences in {packed}", flush=True)
    if a.bench:
        iters = a.bench
    elif a.train_iters:
        iters = a.train_iters
    else:
        if n is None:
            raise SystemExit("no packed training set yet: run a --bench job first or pass --train-iters")
        iters = math.ceil(a.epochs * n / a.gbs)
        print(f"[train.py] {n} packed sequences x {a.epochs} epochs / gbs {a.gbs} -> {iters} iters", flush=True)
    t.train_iters = iters
    if a.exit_mins:
        t.exit_duration_in_mins = a.exit_mins

    cfg.optimizer.lr = a.lr
    cfg.optimizer.min_lr = a.min_lr
    cfg.scheduler.lr_warmup_iters = min(a.warmup, max(1, iters // 10))
    cfg.scheduler.lr_decay_iters = iters
    cfg.scheduler.lr_decay_style = "cosine"

    cfg.validation.eval_interval = 0 if a.bench else a.eval_interval
    cfg.validation.eval_iters = 0 if a.bench else a.eval_iters

    ck = cfg.checkpoint
    ck.pretrained_checkpoint = a.pretrained
    if a.bench:
        ck.save = None
        ck.load = None
        ck.save_interval = None
    else:
        ck.save = a.save
        ck.load = a.save if glob.glob(f"{a.save}/iter_*") else None   # resume across 5 h allocations
        ck.save_interval = a.save_interval
        ck.save_optim = True
        ck.save_rng = True
        ck.most_recent_k = 2
        ck.async_save = False

    lg = cfg.logger
    lg.log_interval = 1 if a.bench else 5
    lg.log_throughput = True
    lg.tensorboard_dir = None
    lg.wandb_project = os.environ.get("WANDB_PROJECT")
    lg.wandb_entity = os.environ.get("WANDB_ENTITY")
    lg.wandb_exp_name = a.run_name
    lg.wandb_save_dir = os.environ.get("WANDB_DIR")
    if os.environ.get("WANDB_MODE") == "disabled":
        lg.wandb_project = None

    cfg.dist.distributed_timeout_minutes = 120
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[train.py] TP{a.tp} EP{a.ep} ETP{a.etp} SP{m.sequence_parallel} recompute={a.recompute} "
              f"dispatcher={m.moe_token_dispatcher_type}/{m.moe_flex_dispatcher_backend} "
              f"seq {a.seq} gbs {a.gbs} mbs {a.mbs} lr {a.lr} iters {iters} world {os.environ.get('WORLD_SIZE')}",
              flush=True)
    finetune(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
