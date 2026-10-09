"""Per-question-type accuracy / length / truncation of sampled traces, plus a few failures.
   python3 inspect_gen.py <gen_dir> [split=test] [n_examples=4]"""
import glob
import json
import statistics
import sys
from collections import defaultdict

gen, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "test")
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rs = [json.loads(l) for f in sorted(glob.glob(f"{gen}/{split}.shard*.jsonl")) for l in open(f)]
by = defaultdict(list)
for r in rs:
    by[r["qtype"]].append(r)
by["ALL"] = rs
for qt, v in by.items():
    toks = [r["completion_tokens"] or 0 for r in v]
    print(f"{qt:18s} n={len(v):6d} acc={100 * sum(r['correct'] for r in v) / len(v):5.1f}% "
          f"trunc={sum(r['finish_reason'] == 'length' for r in v):4d} median_tok={statistics.median(toks):6.0f} "
          f"p95_tok={sorted(toks)[int(0.95 * len(toks))]:6d}")
wrong = [r for r in rs if not r["correct"] and r["finish_reason"] == "stop"]
print(f"\n{len(wrong)} wrong (finished), {sum(r['finish_reason'] == 'length' for r in rs)} truncated")
for r in wrong[:n_ex]:
    print(f"\nWRONG {r['qtype']} {r['id']} gold={r['gold']!r} pred={r['pred']!r}\n  content tail: {r['content'][-200:]!r}")
for r in [r for r in rs if r["finish_reason"] == "length"][:2]:
    print(f"\nTRUNCATED {r['qtype']} {r['id']} gold={r['gold']!r}\n  reasoning tail: {r['reasoning'][-600:]!r}")
