"""Upload the exported HF checkpoint (<run_dir>/hf) to a private Hugging Face model repo with a model card.

usage: python upload_hf.py <run_dir> <repo_id>
Reads <run_dir>/card_facts.json (training settings + benchmark results, written by eval_report.py) for the card.
"""
import json
import os
import sys

from huggingface_hub import HfApi

BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DATA = os.environ.get("HF_DATASET_REPO", "Balab2021/Olympics-Reasoning-Nemotron-3.5-Lightning")


def card(repo_id, facts):
    rows = "\n".join(f"| {k} | {v} |" for k, v in facts.items())
    return f"""---
license: other
license_name: nvidia-open-model-license
base_model: {BASE}
datasets: [{DATA}]
library_name: transformers
pipeline_tag: text-generation
tags: [nemotron, reasoning, olympics, sft, distillation, megatron-bridge, dynamo]
---

# {repo_id.split('/')[-1]}

Full-parameter SFT of [{BASE}](https://huggingface.co/{BASE}) on [{DATA}](https://huggingface.co/datasets/{DATA}):
verified step-by-step reasoning traces over tables from the 120-year Olympics athlete-events data, one per source
row. The traces were distilled from the base model itself, served with NVIDIA Dynamo (SGLang backend), keeping the
shortest trace whose answer matched the exact gold answer (rejection-sampling self-distillation).

Trained with Megatron-Bridge (NeMo 26.08.01) on NVIDIA GB200 NVL72 nodes; the assistant turn is trained in
Nemotron's native reasoning format (`<think>...</think>`, then `Final answer: <answer>`).

| setting / result | value |
|---|---|
{rows}

## Usage

Same as the base model (vLLM / SGLang / Dynamo / Transformers, `enable_thinking=True`, T=1.0, top_p=0.95). The
training system prompt: "You are a sports data analyst. Use only the table you are given. Work through the problem
step by step, then give the final answer on its own line as 'Final answer: <answer>'."
"""


def main():
    run_dir, repo_id = sys.argv[1], sys.argv[2]
    hf_dir = os.path.join(run_dir, "hf")
    assert os.path.exists(os.path.join(hf_dir, "config.json")), f"no exported model in {hf_dir}"
    facts_path = os.path.join(run_dir, "card_facts.json")
    facts = json.load(open(facts_path)) if os.path.exists(facts_path) else {}
    with open(os.path.join(hf_dir, "README.md"), "w") as f:
        f.write(card(repo_id, facts))
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    # resumable, multi-commit upload for the ~60 GB of shards
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=hf_dir, num_workers=16)
    print("pushed", hf_dir, "->", f"https://huggingface.co/{repo_id}", "rev", api.model_info(repo_id).sha, flush=True)


if __name__ == "__main__":
    main()
