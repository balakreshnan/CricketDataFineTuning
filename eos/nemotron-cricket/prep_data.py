"""Convert the cricket reasoning dataset into Megatron-Bridge chat-SFT JSONL.

Writes $PROJ/data/sft/{training,validation}.jsonl, one {"messages": [...]} per line. The assistant turn
carries the trace as `reasoning_content`, so Nemotron's own chat template renders it as
`<think>\n{reasoning}</think>{response}` (the format the model reasons in at inference).
validation.jsonl is a fixed random sample of the by-match held-out `test` split (dev loss only);
the 600-question `eval` benchmark is left untouched for accuracy evals.

Stdlib only, so it runs on the login node:  python3 prep_data.py
"""
import glob
import json
import os
import random
import re
import statistics

HF_HOME = os.environ["HF_HOME"]
SNAP = glob.glob(f"{HF_HOME}/hub/datasets--Balab2021--CricketData-T20-Reasoning-Qwen3.8-Full/snapshots/"
                 + os.environ.get("DATASET_REVISION", "*"))[0]
OUT = os.path.join(os.environ["PROJ"], "data", "sft")
N_VALID = 4000
MAX_REASONING_TOKENS = 3300   # longer traces would be cut off at the 4,096-token packed length (455 train rows)
THINK = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", re.S)


def to_chat(row):
    sys_msg, user_msg, asst = row["messages"]
    assert sys_msg["role"] == "system" and user_msg["role"] == "user" and asst["role"] == "assistant", row["id"]
    m = THINK.match(asst["content"])
    reasoning, response = (m.group(1), m.group(2)) if m else (row["reasoning"], row["response"])
    assert reasoning.strip() and "Final answer:" in response, row["id"]
    return {"messages": [
        {"role": "system", "content": sys_msg["content"]},
        {"role": "user", "content": user_msg["content"]},
        {"role": "assistant", "reasoning_content": reasoning.strip(), "content": response.strip()},
    ], "id": row["id"], "qtype": row["qtype"]}


def read(split):
    for path in sorted(glob.glob(f"{SNAP}/data/{split}-*.jsonl")):
        with open(path) as f:
            for line in f:
                yield json.loads(line)


def main():
    os.makedirs(OUT, exist_ok=True)
    n, dropped, rtoks = 0, 0, []
    with open(f"{OUT}/training.jsonl", "w") as f:
        for row in read("train"):
            if row["reasoning_tokens"] > MAX_REASONING_TOKENS:
                dropped += 1
                continue
            f.write(json.dumps(to_chat(row), ensure_ascii=False) + "\n")
            rtoks.append(row["reasoning_tokens"])
            n += 1
    test = list(read("test"))
    test = [r for r in test if r["reasoning_tokens"] <= MAX_REASONING_TOKENS]
    random.Random(1234).shuffle(test)
    with open(f"{OUT}/validation.jsonl", "w") as f:
        for row in test[:N_VALID]:
            f.write(json.dumps(to_chat(row), ensure_ascii=False) + "\n")
    stats = {"snapshot": SNAP, "training_rows": n, "training_rows_dropped_too_long": dropped, "validation_rows": min(N_VALID, len(test)),
             "reasoning_tokens_mean": statistics.mean(rtoks), "reasoning_tokens_p99": sorted(rtoks)[int(0.99 * n)],
             "reasoning_tokens_max": max(rtoks)}
    with open(f"{OUT}/prep_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
