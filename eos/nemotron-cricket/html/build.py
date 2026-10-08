"""Build index.html = template.html with metrics.json inlined (single self-contained page).

Refresh the data first (on EOS):  python3 tools/collect_metrics.py > html/metrics.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
metrics = json.load(open(os.path.join(HERE, "metrics.json"), encoding="utf-8"))
blob = json.dumps(metrics, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read().replace("/*METRICS*/", blob)
with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8", newline="\n") as f:
    f.write(html)
print("wrote index.html", len(html) // 1024, "KiB")
