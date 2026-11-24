#!/usr/bin/env python
"""Summarise NeMo Gym rollouts of the cricket environment: accuracy, pass@k and tokens per question type.

usage: gym_summary.py <out_dir> <tag> [<tag> ...]     reads <out_dir>/<tag>.jsonl, writes <out_dir>/<tag>_summary.json
With WANDB_API_KEY set and wandb importable, also logs a comparison table to W&B (project $WANDB_PROJECT).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

out, tags = Path(sys.argv[1]), sys.argv[2:]
QTYPES = ("run_rate", "chase_rate", "milestone")


def summarise(tag):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl", encoding="utf-8") if l.strip()]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    def agg(qids):
        rs = [r for q in qids for r in by_q[q]]
        if not rs:
            return None
        toks = [r["output_tokens"] for r in rs if r.get("output_tokens") is not None]
        k = max(len(by_q[q]) for q in qids)
        return {"questions": len(qids), "samples": len(rs), "accuracy": sum(r["reward"] for r in rs) / len(rs),
                f"pass@{k}": sum(any(r["reward"] > 0 for r in by_q[q]) for q in qids) / len(qids),
                "mean_tokens": sum(toks) / len(toks) if toks else None,
                "truncated_rate": sum(not r.get("finished", True) for r in rs) / len(rs),
                "no_answer_rate": sum(r.get("finished", True) and r.get("pred") is None for r in rs) / len(rs)}

    s = {qt: agg([q for q, rs in by_q.items() if rs[0]["qtype"] == qt]) for qt in QTYPES}
    s["overall"] = agg(list(by_q))
    s = {k: v for k, v in s.items() if v}
    (out / f"{tag}_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    o = s["overall"]
    tk = "n/a" if o["mean_tokens"] is None else f"{o['mean_tokens']:.0f}"
    print(f"[{tag}] accuracy {o['accuracy'] * 100:.2f}% over {o['samples']} answers to {o['questions']} questions, "
          f"tokens {tk}, cut off {o['truncated_rate'] * 100:.2f}%, no answer {o['no_answer_rate'] * 100:.2f}%", flush=True)
    for qt in QTYPES:
        if qt in s:
            print(f"   {qt}: {s[qt]['accuracy'] * 100:.2f}%", flush=True)
    return s


res = {t: summarise(t) for t in tags}
if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
    try:
        import wandb
    except ImportError:
        print("wandb not importable in this environment: summaries written, nothing logged")
        raise SystemExit(0)
    wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "nemotron35-cricket-rlvr-rhea"), name=os.environ.get("GYM_RUN_NAME", out.name),
                    job_type="nemo-gym-eval", dir=os.environ.get("WANDB_DIR"), config={"tags": tags, "out_dir": str(out)})
    cols = ["model", "question_type", "questions", "samples", "accuracy", "pass_at_k", "mean_tokens", "truncated_rate", "no_answer_rate"]
    data = [[t, qt, m["questions"], m["samples"], m["accuracy"], next(v for k, v in m.items() if k.startswith("pass@")),
             m["mean_tokens"], m["truncated_rate"], m["no_answer_rate"]] for t, s in res.items() for qt, m in s.items()]
    wb.log({"nemo_gym/cricket_qa": wandb.Table(columns=cols, data=data)})
    for t, s in res.items():
        for k in ("accuracy", "mean_tokens", "truncated_rate"):
            if s["overall"][k] is not None:
                wb.summary[f"nemo_gym/{t}/{k}"] = s["overall"][k]
    print("logged NeMo Gym results to W&B run", wb.id)
    wb.finish()
