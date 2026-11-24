#!/usr/bin/env python
"""Install the cricket environment into NeMo RL's NeMo Gym checkout and write the task files NeMo RL trains on.

usage: prepare.py <code_dir> <data_dir> <nemo_rl_repo> <out_data_dir>
Writes <out_data_dir>/train.jsonl and validation.jsonl (flat NeMo Gym rows routed with agent_ref) and copies them, plus
the environment, to <nemo_rl_repo>/3rdparty/Gym-workspace/Gym/resources_servers/cricket_qa. Standard library only.
"""
import glob
import json
import re
import shutil
import sys
from pathlib import Path

code, data, repo, out = (Path(p) for p in sys.argv[1:5])
gym = repo / "3rdparty" / "Gym-workspace" / "Gym"
dst = gym / "resources_servers" / "cricket_qa"
(dst / "configs").mkdir(parents=True, exist_ok=True)
(dst / "data").mkdir(exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
src = code / "nemorl" / "gym" / "cricket_qa"
shutil.copy(src / "app.py", dst / "app.py")
shutil.copy(src / "configs" / "cricket_qa.yaml", dst / "configs" / "cricket_qa.yaml")
shutil.copy(code / "cricket_qa.py", dst / "cricket_rules.py")       # the project answer parser and checker
# Same dependency file as a simple existing environment, under this environment's name.
pp = (gym / "resources_servers" / "mcqa" / "pyproject.toml").read_text()
pp = pp.replace("gym-resources-servers-mcqa", "gym-resources-servers-cricket-qa")
pp = re.sub(r'\n    "tiktoken",', "", pp)
(dst / "pyproject.toml").write_text(pp)

AGENT = {"type": "responses_api_agents", "name": "cricket_qa_simple_agent"}


def task(r, gold):
    return {"responses_create_params": {"input": r["messages"][:2]}, "agent_ref": AGENT, "qid": r["id"],
            "qtype": r["qtype"], "answer_kind": r["answer_kind"], "gold": gold}


train = []
for f in sorted(glob.glob(str(data / "data" / "train-*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        train.append(task(r, r["gold_answer"]))
val = [task(r, r["gold"]) for r in (json.loads(l) for l in open(data / "data" / "eval-00000.jsonl", encoding="utf-8"))]
for name, rows in (("train.jsonl", train), ("validation.jsonl", val), ("example.jsonl", val[:6])):
    text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in rows)
    (dst / "data" / name).write_text(text, encoding="utf-8")
    if name != "example.jsonl":
        (out / name).write_text(text, encoding="utf-8")
print(f"cricket_qa installed in {dst}; {len(train)} train tasks, {len(val)} validation tasks; data in {out}")
