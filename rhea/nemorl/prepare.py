#!/usr/bin/env python
"""Install the cricket environment into NeMo RL's NeMo Gym checkout and write the task files NeMo RL trains on.

usage: prepare.py <code_dir> <data_dir> <nemo_rl_repo> <out_data_dir>
Writes <out_data_dir>/train.jsonl and validation.jsonl (flat NeMo Gym rows routed with agent_ref) and copies them, plus
the environment, to <nemo_rl_repo>/3rdparty/Gym-workspace/Gym/resources_servers/cricket_qa. Standard library only.
"""
import glob
import json
import os
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


def put(path, text):
    """Write only if the content differs, and atomically: several jobs may run this at the same time."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


put(dst / "app.py", (src / "app.py").read_text(encoding="utf-8"))
cfg = (src / "configs" / "cricket_qa.yaml").read_text(encoding="utf-8")
put(dst / "configs" / "cricket_qa.yaml", cfg)
# One config file per length coefficient (NRL_LEN_COEF), selected by NRL_GYM_CFG in cricket_rhea.yaml.
coef = os.environ.get("NRL_LEN_COEF", "0.4")
cfg_coef, n = re.subn(r"len_coef: [0-9.]+", f"len_coef: {float(coef)}", cfg)
assert n == 1, "len_coef line not found in cricket_qa.yaml"
put(dst / "configs" / f"cricket_qa_len{coef}.yaml", cfg_coef)
put(dst / "cricket_rules.py", (code / "cricket_qa.py").read_text(encoding="utf-8"))   # the project answer parser and checker
# Same dependency file as a simple existing environment, under this environment's name.
pp = (gym / "resources_servers" / "mcqa" / "pyproject.toml").read_text()
pp = pp.replace("gym-resources-servers-mcqa", "gym-resources-servers-cricket-qa")
pp = re.sub(r'\n    "tiktoken",', "", pp)
put(dst / "pyproject.toml", pp)

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
    put(dst / "data" / name, text)
    if name != "example.jsonl":
        put(out / name, text)
print(f"cricket_qa installed in {dst} (len_coef {coef}); {len(train)} train tasks, {len(val)} validation tasks; data in {out}")
