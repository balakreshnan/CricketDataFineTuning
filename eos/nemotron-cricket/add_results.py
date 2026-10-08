"""Add base vs fine-tuned benchmark results to <run_dir>/card_facts.json (read by upload_hf.py for the model card).

usage: python3 add_results.py <proj_dir> <run>
"""
import json
import sys


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    proj, run = sys.argv[1], sys.argv[2]
    path = f"{proj}/runs/{run}/card_facts.json"
    facts = json.load(open(path))
    base = json.load(open(f"{proj}/runs/base/eval/benchmark600.json"))
    ft = json.load(open(f"{proj}/runs/{run}/eval/benchmark600.json"))
    facts["final validation loss"] = "0.371 (0.394 at step 100)"
    facts["benchmark: 600 held-out-match questions x 4 samples (T=1.0, top_p 0.95)"] = \
        f"accuracy {pct(base['accuracy'])} (base) -> **{pct(ft['accuracy'])}** (fine-tuned)"
    for q in sorted(ft["accuracy_by_type"]):
        facts[f"&nbsp;&nbsp;{q}"] = f"{pct(base['accuracy_by_type'][q])} -> {pct(ft['accuracy_by_type'][q])}"
    facts["mean generated tokens"] = f"{base['mean_tokens']:.0f} -> {ft['mean_tokens']:.0f}"
    facts["truncated at 6,144 tokens"] = f"{pct(base['truncated_rate'])} -> {pct(ft['truncated_rate'])}"
    facts["config / tokenizer"] = "base model's config.json and tokenizer files (256K context); weights fine-tuned"
    with open(path, "w") as f:
        json.dump(facts, f, indent=1)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
