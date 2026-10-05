#!/usr/bin/env python
"""Base vs fine-tuned report for a cricket reasoning run:  python report.py <run_dir> [k]

Reads eval/{base,ft}_summary.json, eval/{base,ft}_rank*.jsonl and metrics.jsonl; writes report.md,
comparison.json and (with matplotlib) plots/*.png. Standard library otherwise.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

QTYPES = ("run_rate", "chase_rate", "milestone")


def _samples(run_dir, tag):
    recs = []
    for p in sorted((Path(run_dir) / "eval").glob(f"{tag}_rank*.jsonl")):
        recs += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return recs


def generate(run_dir, k=2):
    run_dir = Path(run_dir)
    base = json.loads((run_dir / "eval" / "base_summary.json").read_text())
    ft = json.loads((run_dir / "eval" / "ft_summary.json").read_text())
    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "metrics.jsonl").exists() else []
    tr = json.loads((run_dir / "train_result.json").read_text()) if (run_dir / "train_result.json").exists() else {}
    names = [("accuracy", "accuracy (avg over samples)"), (f"pass@{k}", f"pass@{k}"), ("mean_tokens", "mean output tokens"),
             ("truncated_rate", "truncated (hit token limit)"), ("no_answer_rate", "finished but no parsable answer")]
    rows = []
    for qt in (*QTYPES, "overall"):
        if qt in base and qt in ft:
            for key, _ in names:
                rows.append({"qtype": qt, "metric": key, "base": base[qt][key], "finetuned": ft[qt][key],
                             "delta": ft[qt][key] - base[qt][key]})

    # per-question flips: mean correctness per question, base vs fine-tuned
    per = defaultdict(lambda: {"base": [], "ft": []})
    for tag in ("base", "ft"):
        for r in _samples(run_dir, tag):
            per[r["id"]][tag].append(r["correct"])
    better = sum(1 for v in per.values() if v["base"] and v["ft"] and sum(v["ft"]) / len(v["ft"]) > sum(v["base"]) / len(v["base"]))
    worse = sum(1 for v in per.values() if v["base"] and v["ft"] and sum(v["ft"]) / len(v["ft"]) < sum(v["base"]) / len(v["base"]))
    comparison = {"rows": rows, "questions_improved": better, "questions_regressed": worse}
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    fmt = lambda key, v: f"{v:,.0f}" if key == "mean_tokens" else f"{100 * v:.2f}%"
    dfmt = lambda key, v: f"{v:+,.0f}" if key == "mean_tokens" else f"{100 * v:+.2f} pts"
    md = [f"# Qwen3.8-27B cricket reasoning LoRA - run `{run_dir.name}`", ""]
    if tr:
        md += [f"- SFT: {tr.get('train_rows')} verified reasoning rows, {tr.get('global_step')} steps "
               f"({tr.get('completed_epochs', 0):.2f} epochs) in {tr.get('train_runtime', 0) / 60:.1f} min, "
               f"final train loss {tr.get('train_loss', float('nan')):.4f}"]
    md += [f"- Eval: {base['overall']['questions']} held-out questions (unseen matches) x {k} samples, thinking mode, "
           "temperature 1.0 / top_p 0.95 / top_k 20, answers checked against exact values", "",
           "| question type | metric | base | fine-tuned | delta |", "|---|---|---|---|---|"]
    label = dict(names)
    for r in rows:
        md.append(f"| {r['qtype']} | {label[r['metric']]} | {fmt(r['metric'], r['base'])} | "
                  f"{fmt(r['metric'], r['finetuned'])} | {dfmt(r['metric'], r['delta'])} |")
    md += ["", f"Questions answered more reliably after fine-tuning: **{better}**; less reliably: **{worse}** "
           f"(of {len(per)}; compares the fraction of correct samples per question)."]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = run_dir / "plots"
        out.mkdir(exist_ok=True)
        qts = [q for q in (*QTYPES, "overall") if q in base and q in ft]
        for key, title, scale in (("accuracy", "Test accuracy (avg over samples, %)", 100), ("mean_tokens", "Mean output tokens", 1)):
            fig, ax = plt.subplots(figsize=(7, 4))
            x = range(len(qts))
            ax.bar([i - 0.2 for i in x], [scale * base[q][key] for q in qts], 0.4, label="base")
            ax.bar([i + 0.2 for i in x], [scale * ft[q][key] for q in qts], 0.4, label="fine-tuned")
            ax.set_xticks(list(x)), ax.set_xticklabels(qts), ax.set_title(title), ax.legend(), ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(), fig.savefig(out / f"{key}_by_qtype.png", dpi=120), plt.close(fig)
        trn = [m for m in metrics if "loss" in m and "eval_loss" not in m]
        ev = [m for m in metrics if "eval_loss" in m]
        if trn:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([m["epoch"] for m in trn], [m["loss"] for m in trn], label="train loss")
            if ev:
                ax.plot([m["epoch"] for m in ev], [m["eval_loss"] for m in ev], "o-", label="dev loss")
            ax.set_xlabel("epoch"), ax.legend(), ax.grid(alpha=0.3)
            fig.tight_layout(), fig.savefig(out / "training_curves.png", dpi=120), plt.close(fig)
        md += ["", "## Plots", ""] + [f"![{p.name}](plots/{p.name})" for p in sorted(out.glob("*.png"))]
    except ImportError:
        pass
    (run_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:len(rows) + 8]))
    return comparison


if __name__ == "__main__":
    generate(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
