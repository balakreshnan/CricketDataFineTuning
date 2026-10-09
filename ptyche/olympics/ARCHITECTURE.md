# Architecture: Olympics reasoning distillation + Nemotron SFT on Ptyche

How the pieces fit together: the end-to-end flow, the distillation flow (Dynamo + SGLang), the fine-tuning flow
(Megatron-Bridge on GB200), the job chain that runs it without supervision, and the data contracts between stages.
The numbers, results and reproduction commands are in [README.md](README.md).

---

## 1. End-to-end flow

```mermaid
flowchart LR
    SRC[("pranjalvoid/olympics-dataset<br/>athlete_events.csv<br/>271,116 rows")] --> DEDUP["dedup exact rows<br/>269,731 unique"]
    DEDUP --> Q["Question builder<br/>rows_qa.py<br/>1 verifiable question / row<br/>gold answer computed"]
    BASE[("Nemotron-3.5-Lightning<br/>30B-A3B BF16")] --> DYN
    Q --> DYN["Distillation<br/>NVIDIA Dynamo + SGLang<br/>K samples / question"]
    DYN --> GRADE{"grade vs gold<br/>correct + finished?"}
    GRADE -- "no correct trace" --> RETRY["resample with more K /<br/>longer budget /<br/>alternate template"] --> DYN
    GRADE -- "yes" --> PICK["keep shortest<br/>correct trace per row"]
    PICK --> DS[("Distilled dataset<br/>269,723 rows<br/>HF public")]
    PICK --> SFT["SFT files<br/>(prompt+trace ≤ 8k)"]
    SFT --> PACK["offline packing<br/>8,192-token seqs"]
    BASE --> IMP["HF → Megatron<br/>checkpoint"]
    PACK --> TRAIN["Full SFT<br/>Megatron-Bridge<br/>16× GB200, EP8, HybridEP"]
    IMP --> TRAIN
    TRAIN --> EXP["Megatron → HF<br/>export"]
    EXP --> EVAL["Eval through Dynamo<br/>base vs fine-tuned"]
    EVAL --> REP["report → W&B<br/>+ model card facts"]
    EXP --> PUSH[("Fine-tuned model<br/>HF private")]
    REP --> PUSH
    DYN -. stats .-> WB[("W&B<br/>nemotron35-olympics-reasoning")]
    TRAIN -. loss / throughput .-> WB
    REP -.-> WB
```

The same base model is the **teacher** (served by Dynamo for distillation), the **student** (fine-tuned), and the
**baseline** (evaluated against the student). That makes this rejection-sampling self-distillation; see README §6.

---

## 2. System view: where things run

```mermaid
flowchart TB
    subgraph LAPTOP["Laptop (Windows, Claude Code)"]
        BR["ptyche-bridge<br/>(1 MFA SSH session)"]
        PX["px / psync<br/>queue commands + code"]
        PX --> BR
    end
    subgraph LOGIN["login-ptyche01 (x86_64)"]
        SH["bash -s<br/>executes queued commands"]
        SB["sbatch / squeue<br/>pipeline.sh"]
        SH --> SB
    end
    subgraph SLURM["Slurm, partition tcpo · GB200 NVL72 nodes (aarch64)"]
        G["gen.sbatch nodes<br/>Dynamo-SGLang image"]
        T["job.sbatch nodes<br/>NeMo 26.08.01 image"]
        D["data.sbatch / rows.sbatch<br/>NeMo image, CPU work"]
    end
    subgraph LUSTRE["/lustre/fsw/general_sa/bbalakreshna"]
        L1["hf_cache/ · models · containers/*.sqsh"]
        L2["olympics/data · gen · ckpts · runs"]
        L3["logs/olympics · secrets.env"]
    end
    HOME["/home/bbalakreshna/clustercodes/olympics<br/>(code only)"]
    BR == ssh ==> SH
    SB --> G & T & D
    G & T & D <--> LUSTRE
    G & T & D --> HOME
    T -- push --> HF[("Hugging Face")]
    D -- push --> HF
    T & D -- metrics --> WANDB[("W&B")]
```

* **Code** lives in `~/clustercodes/olympics`, synced from the laptop's `ptyche/` folder by `psync`.
  **Everything large** (weights, HF cache, containers, data, samples, checkpoints, logs) lives on Lustre.
* The login node only submits jobs. It kills long CPU-heavy processes and cannot run `enroot import`, so all CPU
  work (question building, dataset builds) runs in the NeMo container on a compute node.
* Once a chain is submitted, it needs neither the laptop nor the bridge.

---

## 3. Distillation flow

### 3.1 Question construction

```mermaid
flowchart LR
    R["source row r<br/>(athlete, event, Games)"] --> E{"event entrants ≤ 150<br/>and name unique?"}
    E -- yes --> EC["event table<br/>Name·NOC·Age·Height·Weight·Medal<br/>(fixed row order per event)"]
    E -- no --> AC
    R --> AC["athlete career table ≤ 80 rows<br/>Games·Year·City·Sport·Event·Age·Medal"]
    EC --> CE["ev_same_noc · ev_older ·<br/>ev_taller · ev_noc_medal_count"]
    AC --> CA["at_events_at · at_medals_at ·<br/>at_events_before · at_other_events_at"]
    CE & CA --> SEL["template = hash(row) mod n<br/>skip already-used prompts<br/>(--fallback N: N-th next)"]
    SEL --> OUT["question JSONL<br/>id · qtype · gold · source · messages"]
```

The gold answer is computed from the same table that goes into the prompt, so it is fully determined and can be
checked exactly. The split is by athlete (10 % test), matching the benchmark split.

### 3.2 One Dynamo deployment per node (`gen.sbatch`)

```mermaid
flowchart LR
    subgraph NODE["GB200 node (one container, one srun task)"]
        C["gen_client.py<br/>shard i/N of questions<br/>512–1,024 in-flight requests<br/>K samples each"]
        F["dynamo.frontend :8000<br/>OpenAI API<br/>KV-aware router<br/>nemotron_v3 reasoning parser"]
        KV[("file discovery store<br/>/tmp/dynamo-kv-JOB")]
        subgraph W["dynamo.sglang workers (TP1)"]
            W0["GPU0 · SGLang"]
            W1["GPU1 · SGLang"]
            W2["GPU2 · SGLang"]
            W3["GPU3 · SGLang"]
        end
        C -- "chat.completions<br/>T=1.0 top_p=0.95<br/>enable_thinking" --> F
        F -- "TCP request plane" --> W0 & W1 & W2 & W3
        W0 & W1 & W2 & W3 -. register .-> KV
        F -. watch .-> KV
        W0 & W1 & W2 & W3 -. "KV events (ZMQ)" .-> F
    end
    Q[("questions_*.jsonl")] --> C
    C --> S[("samples *.shardI.jsonl<br/>reasoning · content · tokens ·<br/>finish_reason · pred · correct")]
```

* **Workers:** each GPU holds a full copy of the 30B-A3B model (TP1), with about 8.7M tokens of KV and Mamba-state
  capacity. Settings follow the model card's Blackwell recipe: FlashInfer Mamba kernels, an FP16 SSM state with
  stochastic rounding, and a decode CUDA graph sized for up to 256 batched requests.
* **Router:** the KV-aware router sends questions that share a table prefix (the same event) to the worker that
  already has that prefix cached.
* **No etcd or NATS:** discovery uses a node-local file store and requests go over TCP, so every node is a
  self-contained deployment, and N nodes are just N independent shards.
* **Resumable:** the client appends one JSON line per sample and skips `(id, sample)` pairs already present.
* **Throughput:** about 31.7k output tokens/s per node on long runs (12 nodes: 380k tok/s, about 156 requests/s); startup takes about 3.5 min. Per-node tables are in README §3.4.

### 3.3 Coverage loop (rejection sampling to ≥ 1:1)

```mermaid
flowchart TB
    P1["Pass 1: all 269,731 row questions<br/>K=4, 16k tokens, 12 nodes"] --> B1["build_dataset.py<br/>per row: shortest correct,<br/>finished, ≤ 28k trace"]
    B1 --> U1{"rows without<br/>trace?"}
    U1 -- "40,752 q<br/>(incl. reworded template)" --> P2["Retry: same questions<br/>K=6"] --> B2["build"] --> U2{"unsolved?"}
    U2 -- "5,938 rows" --> P3["Fallback: next template<br/>K=8"] --> B3["build"] --> U3{"unsolved?"}
    U3 -- "1,146 rows" --> P4["Retry2: all their questions<br/>K=16, 28k tokens"] --> B4["build"] --> U4{"unsolved?"}
    U4 -- "188 rows" --> P5["Fallback2: third template<br/>K=16, 28k tokens"] --> B5["build"] --> G{"rows ≥ 269,731?<br/>or --allow-missing N<br/>(owner-approved)"}
    G -- "269,723 (8 missing, approved)" --> PUSH["dataset card lists the missing rows<br/>→ HF public + W&B artifact stats"]
```

Every build merges **all** sample files (`rows*.jsonl`) by question id and every question file
(`questions_rows_{train,test,fallback*}.jsonl`) by source row, so later passes only add to earlier ones.

---

## 4. Fine-tuning flow

### 4.1 Stages

```mermaid
flowchart LR
    B[("build_dataset.py<br/>data/sft/training.jsonl<br/>validation.jsonl")] --> SNAP["frozen snapshot<br/>data/sft-rows-v1<br/>227,051 + 2,237 rows"]
    SNAP --> TOK["per-dataset tokenizer link<br/>sft-rows-v1/tokenizer → model snapshot"]
    TOK --> PK["pack job (2 nodes, 1 step)<br/>ChatSFT, loss on assistant turn<br/>8,192-token packed parquet<br/>50,446 sequences"]
    HFB[("HF base weights")] --> IMP["import job<br/>run_conversion.py import<br/>EP4 → ckpts/base-megatron"]
    PK --> TR["train job (4 nodes)<br/>train.py → finetune()<br/>789 steps, ckpt every 100"]
    IMP --> TR
    TR --> CK[("ckpts/sft-rows-e1/iter_0000789")]
    CK --> EX["export job<br/>run_conversion.py export<br/>+ base config/tokenizer files"]
    EX --> HFO[("runs/sft-rows-e1/hf<br/>14 shards, 65.8 GB")]
    HFO --> EV["eval job = gen.sbatch<br/>MODEL_DIR=fine-tuned<br/>550 table + 26,997 row test q, K=4"]
    EV --> RP["report job<br/>eval_report.py<br/>base vs FT on common questions"]
    RP --> PU["push job<br/>upload_hf.py (private)"]
    HFO --> PU
```

* **Why a frozen snapshot:** training started while the last coverage passes were still running. Training reads a
  copy that cannot change underneath it, and its stats travel with it into the model card.
* **Why a per-dataset tokenizer link:** Megatron-Bridge names its packed cache `<tokenizer>_pad_seq_to_mult2_sft_<hash>`
  and the hash does not depend on the data. Pointing the packer at `<data>/tokenizer` gives every dataset its own
  cache.
* **Same harness for evaluation:** the fine-tuned export is served by exactly the same Dynamo deployment and
  sampling settings that produced the base model's samples, so the comparison is like for like.

### 4.2 Multi-GPU layout (16 GB200, TP1 · EP8 · DP16)

```mermaid
flowchart TB
    subgraph G1["Expert-parallel group A (EP8)"]
        direction LR
        N1["node 1<br/>GPU0-3<br/>experts 0-63"] --- N2["node 2<br/>GPU0-3<br/>experts 64-127"]
    end
    subgraph G2["Expert-parallel group B (EP8)"]
        direction LR
        N3["node 3<br/>GPU0-3<br/>experts 0-63"] --- N4["node 4<br/>GPU0-3<br/>experts 64-127"]
    end
    G1 <== "expert data-parallel (2 replicas):<br/>expert grad all-reduce" ==> G2
    NOTE["Dense attention + Mamba + embeddings: replicated on all 16 GPUs (DP16)<br/>distributed optimizer shards their optimizer state<br/>HybridEP flex dispatcher: token all-to-all inside each EP8 group over NVLink (NVL72)"]
```

* Each of the 128 experts lives on exactly one GPU per EP group (16 experts per GPU). Tokens are routed to their
  top-k experts by the HybridEP dispatcher, which is designed for GB200 NVL72's NVLink domain.
* **Memory:** at EP4 on a single node, each GPU had to hold 32 experts plus their FP32 optimizer state, and ran
  out of memory at 171 of 184 GB. EP8 halves that; peak was 97–122 GB.
* Selective activation recompute (`moe, layernorm, core_attn, mlp`) trades extra compute for activation
  memory at 8k sequence length.
* Each step processes 64 packed sequences of 8,192 tokens, about 0.5M tokens, at about 3.3 s per step and
  ~270 TFLOP/s per GPU.

---

## 5. Unattended execution: the Slurm chain

`pipeline.sh` submits every stage with `--dependency=afterok:<previous>`, so a failure cancels everything after it
and nothing runs on bad inputs. The one exception: packing waits with `afterany` on the final data build, so a
coverage gate that refuses the HF push does not block training.

```mermaid
flowchart LR
    subgraph DATA["data chain (STAGES=data)"]
        d1[data retry] --> g1[gen retry] --> d2[data fallback] --> g2[gen fallback] --> d3["data final<br/>1:1 gate · W&B · HF push"]
    end
    subgraph TRAIN["train chain (STAGES=train, SFT_DATA=frozen copy)"]
        p[pack] --> s[SFT] --> e[export] --> v[Dynamo eval] --> r[report] --> u[HF model push]
    end
    d3 -. "afterany (or none when STAGES=train)" .-> p
```

Switches: `FROM=retry2` resumes the data chain at the second retry; `STAGES=data|train|all` picks a chain;
`SFT_DATA` selects the frozen training copy. Every job logs to `$LOG_DIR/olympics/<job-name>-<jobid>.out`; the
chain itself appends to `pipeline-<run>.log`.

**As actually executed on 2026-10-09:** the data chain retry → fallback → final ended 1,146 rows short. Retry2
and fallback2 were then added (still 8 short), and the owner approved a final push with `--allow-missing 8`. The
train chain ran in parallel from the 99.58 % snapshot after two packing restarts: one for the stale-cache bug, one
for the 1 h time limit.

---

## 6. Data contracts

| Stage output | Format (one JSON object per line) |
|---|---|
| Question (`data/questions_*.jsonl`) | `id` (`row<idx>-<qtype>[-fb[N]]`), `qtype`, `answer_kind`, `gold`, `source` {`row_index`, `athlete_id`, `name`, `games`, `event`, `noc`}, `messages` [system, user] |
| Sample (`gen/<tag>/<split>.shard<i>.jsonl`) | `id`, `qtype`, `sample`, `gold`, `reasoning`, `content`, `finish_reason`, `completion_tokens`, `prompt_tokens`, `pred`, `correct` |
| Published row (`hf/data/{train,test}.jsonl`) | `id` (`row<idx>`), `source_row_index`, `question_id`, `qtype`, athlete/Games/event/NOC metadata, `question`, `answer`, `reasoning`, `response`, `reasoning_tokens`, `prompt_tokens`, `messages` |
| SFT row (`data/sft*/training.jsonl`) | `messages`: system · user · assistant {`reasoning_content`, `content`} |
| Eval report (`runs/<run>/eval/report.json`) | per benchmark × {base, finetuned} × qtype: `samples`, `pass@1`, `mean_tokens`, `truncated_pct`; `questions_compared` |

**Grading** (`olympics_qa.is_correct`): the answer is taken from the last `Final answer:` after the reasoning
block. Integers must match exactly, 1-decimal answers within 0.05, and names or NOC codes are compared
case- and whitespace-normalised. Truncated samples never count as correct.

## 7. Storage layout (`$LUSTRE_DIR/olympics`)

```
data/      questions_*.jsonl, rows_stats.json, rows_dataset_stats.json, unsolved_rows.jsonl
           sft/ (latest build) · sft-rows-v1/ (frozen training copy + its packed cache) · sft-bench/ · hf/ (pushed files)
gen/       full/ (9.8k table set) · rows/ (all row passes) · eval-sft-rows-e1/ · smoke*/   + logs/node*.{frontend,worker*,client-*}.log
ckpts/     base-megatron/ (imported base) · sft-rows-e1/ (iter_0000700, iter_0000789; 859 GB with optimizer state)
runs/      sft-rows-e1/hf (exported model) · eval/report.json · card_facts.json
```
