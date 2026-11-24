#!/usr/bin/env python
"""Install the cricket environment into the NeMo Gym work directory and convert the benchmark questions to task rows.

usage: gym_prepare.py <code_dir> <data_dir> <gym_work_dir>
Standard library only.
"""
import json
import shutil
import sys
from pathlib import Path

code, data, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dst = work / "resources_servers" / "cricket_qa"
(dst / "configs").mkdir(parents=True, exist_ok=True)
(dst / "data").mkdir(exist_ok=True)
src = code / "nemotron" / "gym" / "cricket_qa"
shutil.copy(src / "app.py", dst / "app.py")
shutil.copy(src / "configs" / "cricket_qa.yaml", dst / "configs" / "cricket_qa.yaml")
shutil.copy(code / "cricket_qa.py", dst / "cricket_rules.py")       # the project's answer parser and checker
(dst / "__init__.py").write_text("")
(dst / "requirements.txt").write_text("nemo-gym[dev]\n")
for stale in list((dst / "data").glob("*_metrics*.json")) + list((dst / "data").glob("*_prepare.jsonl")):
    stale.unlink()   # NeMo Gym caches dataset statistics next to the data; they must match the current config
rows = [json.loads(l) for l in open(data / "data" / "eval-00000.jsonl", encoding="utf-8")]
tasks = [{"responses_create_params": {"input": q["messages"]}, "qid": q["id"], "qtype": q["qtype"],
          "answer_kind": q["answer_kind"], "gold": q["gold"]} for q in rows]
for name, part in (("benchmark.jsonl", tasks), ("example.jsonl", tasks[:6])):
    with open(dst / "data" / name, "w", encoding="utf-8") as f:
        for t in part:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"cricket_qa environment installed in {dst}; {len(tasks)} benchmark tasks")
