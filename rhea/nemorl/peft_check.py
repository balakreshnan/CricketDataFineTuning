#!/usr/bin/env python
"""Check that the NeMo RL adapter works with plain PEFT: apply it to the base model, then merge it.

usage: peft_check.py <code_dir> <model_dir> <adapter_dir> <validation.jsonl> <out.json>
Prints how many adapter tensors PEFT loaded, and compares greedy answers of base, base+adapter and merged model.
Exit code 0 only if every adapter tensor was loaded and the adapter changes the output.
"""
import json
import sys

import torch
from safetensors import safe_open

code, model_dir, adapter, val, out = sys.argv[1:6]
sys.path.insert(0, code)
import sampling  # noqa: E402
import peft  # noqa: E402
import transformers  # noqa: E402
from peft import PeftModel  # noqa: E402

tok = sampling.load_tokenizer(model_dir)
model = sampling.load_model(model_dir, 0).eval()
rows = [json.loads(x) for x in open(val)][:4]


def answer(m):
    res = []
    for r in rows:
        ids = tok.apply_chat_template(r["responses_create_params"]["input"], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(m.device)
        with torch.no_grad():
            o = m.generate(**ids, max_new_tokens=2048, do_sample=False)
        n = o.shape[1] - ids["input_ids"].shape[1]
        res.append({"gold": r["gold"], "tokens": n, "tail": tok.decode(o[0, -60:], skip_special_tokens=True)})
    return res


rep = {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__}
rep["base"] = answer(model)
with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
    file_keys = set(f.keys())
pm = PeftModel.from_pretrained(model, adapter)
have = {n.replace(".default", "") for n, _ in pm.named_parameters() if "lora_" in n}
rep["file_tensors"], rep["loaded"] = len(file_keys), len(file_keys & have)
rep["not_loaded"] = sorted(file_keys - have)[:12]
rep["not_loaded_count"] = len(file_keys - have)
rep["adapter"] = answer(pm)
merged = pm.merge_and_unload()
rep["merged"] = answer(merged)
json.dump(rep, open(out, "w"), indent=1)
for k in ("transformers", "peft", "torch", "file_tensors", "loaded", "not_loaded_count", "not_loaded"):
    print(k, rep[k])
for tag in ("base", "adapter", "merged"):
    for x in rep[tag]:
        print(f"[{tag}] gold={x['gold']} tokens={x['tokens']} tail={x['tail'][-160:]!r}")
changed = [a["tokens"] for a in rep["adapter"]] != [b["tokens"] for b in rep["base"]]
same = [a["tokens"] for a in rep["adapter"]] == [b["tokens"] for b in rep["merged"]]
print("adapter changes output:", changed, "| merged matches adapter (token counts):", same)
sys.exit(0 if changed else 1)
