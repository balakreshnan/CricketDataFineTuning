"""Upload the exported HF checkpoint (<run_dir>/hf) to a public Hugging Face model repo with a model card.

usage: python upload_hf.py <run_dir> <repo_id>
Reads optional <run_dir>/card_facts.json (training settings + results) to fill in the card.
"""
import json
import os
import sys

from huggingface_hub import HfApi

BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DATA = "Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full"


def card(repo_id, facts):
    rows = "\n".join(f"| {k} | {v} |" for k, v in facts.items())
    return f"""---
license: other
license_name: nvidia-open-model-license
base_model: {BASE}
datasets: [{DATA}]
library_name: transformers
pipeline_tag: text-generation
tags: [nemotron, reasoning, cricket, sft, megatron-bridge]
---

# {repo_id.split('/')[-1]}

Full-parameter SFT of [{BASE}](https://huggingface.co/{BASE}) on
[{DATA}](https://huggingface.co/datasets/{DATA}): 458K verified step-by-step reasoning traces for T20
International cricket arithmetic (run rate, required rate, projections, strike rate, milestones).

Trained with Megatron-Bridge (NeMo 26.08 container) on NVIDIA H100 nodes; the assistant turn is trained in
Nemotron's native reasoning format (`<think>...</think>` then the answer, ending in `Final answer: <number>`).

| setting | value |
|---|---|
{rows}

## Usage

Same as the base model (vLLM / Transformers, `enable_thinking=True`). Use the dataset's system prompt:
"You are a cricket analyst. Work through the problem step by step, then give the final answer on its own line
as 'Final answer: <number>'."
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
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    # resumable, multi-commit upload for the ~60 GB of shards
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=hf_dir, num_workers=16)
    print("pushed", hf_dir, "->", f"https://huggingface.co/{repo_id}",
          "rev", api.model_info(repo_id).sha, flush=True)


if __name__ == "__main__":
    main()
