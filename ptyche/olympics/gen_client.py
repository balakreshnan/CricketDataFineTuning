"""Sample reasoning traces from the Dynamo frontend (OpenAI-compatible) for one shard of the questions.
   python gen_client.py --questions q.jsonl --out samples.jsonl --shard 0/4 --k 4 --url http://localhost:8000
Resumable: (id, sample) pairs already in --out are skipped. Standard library only (runs in the Dynamo image)."""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olympics_qa import is_correct, parse_final  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--questions", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default=None, help="served model name (default: first of /v1/models)")
ap.add_argument("--concurrency", type=int, default=256)
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--limit", type=int, default=0, help="first N questions of the shard (smoke tests)")
a = ap.parse_args()

si, sn = map(int, a.shard.split("/"))
qs = [json.loads(l) for f in a.questions for l in open(f)]
random.Random(0).shuffle(qs)  # files are grouped by qtype: mix so every shard (and --limit) sees all types
qs = [x for i, x in enumerate(qs) if i % sn == si]
if a.limit:
    qs = qs[:a.limit]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try:
            r = json.loads(l)
            done.add((r["id"], r["sample"]))
        except json.JSONDecodeError:
            pass  # torn last line from a killed run
todo = [(x, s) for x in qs for s in range(a.k) if (x["id"], s) not in done]


def post(path, body=None, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


model = a.model or post("/v1/models")["data"][0]["id"]
print(f"model={model} shard={a.shard} questions={len(qs)} todo={len(todo)} (done {len(done)})", flush=True)

lock = threading.Lock()
fout = open(a.out, "a")
stats = {"n": 0, "ok": 0, "tok": 0, "err": 0, "trunc": 0}
t0 = time.time()


def run(item):
    x, s = item
    body = {"model": model, "messages": x["messages"], "max_tokens": a.max_tokens, "temperature": a.temperature,
            "top_p": a.top_p, "chat_template_kwargs": {"enable_thinking": True}}
    for attempt in range(5):
        try:
            r = post("/v1/chat/completions", body)
            break
        except Exception as e:  # worker restart / transient frontend error
            if attempt == 4:
                with lock:
                    stats["err"] += 1
                print(f"FAILED {x['id']}#{s}: {e}", flush=True)
                return
            time.sleep(10 * (attempt + 1))
    ch = r["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    rec = {"id": x["id"], "qtype": x["qtype"], "sample": s, "gold": x["gold"], "answer_kind": x["answer_kind"],
           "reasoning": reasoning, "content": content, "finish_reason": ch.get("finish_reason"),
           "completion_tokens": r.get("usage", {}).get("completion_tokens"),
           "prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "pred": parse_final(content),
           "correct": is_correct(x, content)}
    with lock:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["n"] += 1
        stats["ok"] += rec["correct"]
        stats["tok"] += rec["completion_tokens"] or 0
        stats["trunc"] += rec["finish_reason"] == "length"
        if stats["n"] % 200 == 0:
            fout.flush()
            dt = time.time() - t0
            print(f"[{time.strftime('%T')}] {stats['n']}/{len(todo)} acc={stats['ok'] / stats['n']:.3f} "
                  f"trunc={stats['trunc']} err={stats['err']} {stats['tok'] / dt:,.0f} tok/s "
                  f"mean_tok={stats['tok'] / stats['n']:.0f}", flush=True)


with ThreadPoolExecutor(a.concurrency) as pool:
    list(pool.map(run, todo))
fout.close()
dt = time.time() - t0
print(f"DONE shard {a.shard}: {stats} in {dt / 60:.1f} min ({stats['tok'] / max(dt, 1):,.0f} tok/s)", flush=True)
