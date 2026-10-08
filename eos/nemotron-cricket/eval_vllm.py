"""Accuracy on the dataset's fixed 600-question `eval` benchmark (held-out matches) with vLLM, one engine, TP8.

Same protocol as the Qwen3.8 cricket evals: k samples per question in thinking mode, accuracy = mean over samples,
answer = last 'Final answer: <number>' after </think>; rates within 0.011 (2 dp rounding), integers exact.
usage: python eval_vllm.py --model <hf dir> --out <json> [--k 4]
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

_FINAL = re.compile(r"final answer\s*[:：]\s*\**\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I)
RATE_TOL = 0.011


def parse_answer(text):
    if "</think>" not in text:
        return None   # never closed its reasoning (truncated)
    hits = _FINAL.findall(text.split("</think>", 1)[1])
    try:
        return float(hits[-1].replace(",", "")) if hits else None
    except ValueError:
        return None


def is_correct(pred, gold, kind):
    if pred is None:
        return False
    return abs(pred - gold) < 1e-6 if kind == "int" else abs(pred - gold) <= RATE_TOL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--tp", type=int, default=8)
    a = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path = glob.glob(f"{os.environ['HF_HOME']}/hub/datasets--Balab2021--CricketData-T20-Reasoning-Qwen3.8-Full/"
                     f"snapshots/*/data/eval-*.jsonl")[0]
    rows = [json.loads(line) for line in open(path)]
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for r in rows]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=8192, trust_remote_code=True,
              gpu_memory_utilization=0.85, seed=1234)
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, max_tokens=a.max_tokens, seed=1234)
    outs = llm.generate(prompts, sp)

    per_type, samples, ntok, trunc = defaultdict(list), [], [], 0
    for r, o in zip(rows, outs):
        for c in o.outputs:
            pred = parse_answer(c.text)
            ok = is_correct(pred, r["gold"], r["answer_kind"])
            per_type[r["qtype"]].append(ok)
            ntok.append(len(c.token_ids))
            trunc += c.finish_reason == "length"
            samples.append({"id": r["id"], "qtype": r["qtype"], "gold": r["gold"], "pred": pred, "correct": ok,
                            "tokens": len(c.token_ids), "finish": c.finish_reason, "text": c.text})
    allok = [s["correct"] for s in samples]
    res = {"model": a.model, "questions": len(rows), "k": a.k, "accuracy": statistics.mean(allok),
           "accuracy_by_type": {t: statistics.mean(v) for t, v in sorted(per_type.items())},
           "mean_tokens": statistics.mean(ntok), "truncated_rate": trunc / len(samples),
           "no_answer_rate": sum(s["pred"] is None for s in samples) / len(samples)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    with open(a.out.replace(".json", "_samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":   # vLLM spawns workers that re-import this module
    main()
