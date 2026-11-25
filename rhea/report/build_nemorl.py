#!/usr/bin/env python
"""Build report/nemorl_cricket_report.html from report/data/nemorl_raw.txt.

Run: NeMo RL (GRPO + LoRA) on Nemotron-3.5-Lightning-30B, T20 cricket questions, Rhea, 2026-11-24.
Training jobs 131698 (steps 1-70) and 131714 (resumed from the step-65 checkpoint, steps 66-100);
NeMo Gym evaluation job 132709.

The page is fully self-contained: data is embedded as JSON, charts are hand-drawn SVG, no external
scripts, fonts or CDNs.

Usage:  python build_nemorl.py
"""
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Flip this to "uploaded" once the adapter is on Hugging Face, then re-run the builder.
HF_STATUS = "uploaded"
# ---------------------------------------------------------------------------------------------

HF_URL = "https://huggingface.co/Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-NeMo-RL-Rhea"
WANDB_PROJECT = "https://wandb.ai/balabala76/nemotron35-cricket-nemo-rl-rhea"
WANDB_RUN_1 = WANDB_PROJECT + "/runs/kqadpwsx"  # steps 1-70, job 131698
WANDB_RUN_2 = WANDB_PROJECT + "/runs/ijt9bvh1"  # steps 66-100, job 131714
WANDB_EVAL = WANDB_PROJECT + "/runs/zovlu0p2"  # NeMo Gym eval, job 132709

JOB_FIRST, JOB_RESUME = 131698, 131714
RESUME_STEP = 65  # job 131714 restarted from the step-65 checkpoint
GPU_MEM_MIB = 286524
N_GPUS = 16

# Held-out evaluation with NeMo Gym: 600 questions x 4 sampled answers = 2,400 answers each.
EVAL = {
    "base": {"acc": 86.50, "tokens": 1565, "cutoff": 9.42, "noanswer": 0.00,
             "run_rate": 91.75, "chase_rate": 68.12, "milestone": 99.62},
    "tuned": {"acc": 97.71, "tokens": 107, "cutoff": 0.00, "noanswer": 0.00,
              "run_rate": 99.00, "chase_rate": 94.62, "milestone": 99.50},
}

here = Path(__file__).resolve().parent
RAW = here / "data" / "nemorl_raw.txt"
OUT = here / "nemorl_cricket_report.html"


def parse(path):
    step_re = re.compile(r"^\d+,\d+,")
    gpu_re = re.compile(r"^\d{4}/\d\d/\d\d \d\d:\d\d,")
    by_job, gpu, in_gpu = {}, [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("---GPU"):
            in_gpu = True
            continue
        if not in_gpu and step_re.match(line):
            job, step, rew, tok, sec = line.split(",")
            by_job.setdefault(int(job), {})[int(step)] = {
                "s": int(step), "j": int(job), "r": float(rew), "t": float(tok), "sec": float(sec)}
        elif in_gpu and gpu_re.match(line):
            ts, util, mem, peak, power, temp, n = line.split(",")
            hh, mm = ts.split(" ")[1].split(":")
            gpu.append({"t": f"{hh}:{mm}", "m": int(hh) * 60 + int(mm), "u": float(util),
                        "ma": float(mem), "mp": float(peak), "w": float(power), "c": float(temp),
                        "n": int(n)})
    first, resumed = by_job[JOB_FIRST], by_job[JOB_RESUME]
    final = dict(first)
    final.update(resumed)  # job 131714 wins for the overlapping steps 66-70
    steps = [final[s] for s in sorted(final)]
    superseded = [first[s] for s in sorted(first) if s in resumed]
    m0 = gpu[0]["m"]
    for g in gpu:
        g["m"] -= m0
    assert [r["s"] for r in steps] == list(range(1, 101)), "expected steps 1..100"
    return steps, superseded, gpu


def main():
    steps, superseded, gpu = parse(RAW)
    data = {
        "steps": steps, "superseded": superseded, "gpu": gpu, "eval": EVAL,
        "resumeStep": RESUME_STEP, "jobs": [JOB_FIRST, JOB_RESUME],
        "gpuMemMiB": GPU_MEM_MIB, "nGpus": N_GPUS,
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (TEMPLATE
            .replace("__DATA__", blob)
            .replace("__HF_STATUS__", HF_STATUS)
            .replace("__HF_CLASS__", "done" if HF_STATUS.strip().lower() == "uploaded" else "pending")
            .replace("__HF_URL__", HF_URL)
            .replace("__WANDB_RUN_1__", WANDB_RUN_1)
            .replace("__WANDB_RUN_2__", WANDB_RUN_2)
            .replace("__WANDB_EVAL__", WANDB_EVAL))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    total_min = sum(r["sec"] for r in steps) / 60
    wasted_min = sum(r["sec"] for r in superseded) / 60
    kwh = sum(g["w"] * N_GPUS / 60 / 1000 for g in gpu)
    print(f"wrote {OUT} ({len(html):,} chars)")
    print(f"steps: {len(steps)} final + {len(superseded)} superseded; step time {total_min:.1f} min "
          f"(+{wasted_min:.1f} min superseded); gpu minutes: {len(gpu)}; energy {kwh:.2f} kWh; "
          f"peak mem {max(g['mp'] for g in gpu):,.0f} MiB")


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeMo RL Cricket Report</title>
<style>
:root{
  --bg:#f4f6f9; --surface:#ffffff; --surface2:#edf0f5; --text:#17202b; --muted:#556072;
  --border:#d3d9e3; --grid:#e2e6ed; --link:#1f63b8;
  --blue:#2a78d6; --orange:#eb6834; --green:#1baf7a;
  --blue-soft:rgba(42,120,214,.15); --orange-soft:rgba(235,104,52,.16); --green-soft:rgba(27,175,122,.16);
  --code-bg:#eef1f6; --tip-bg:#ffffff; --shadow:0 1px 3px rgba(20,30,50,.08);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0e1217; --surface:#161b23; --surface2:#1e252f; --text:#e7ebf1; --muted:#9aa6b7;
    --border:#2d3643; --grid:#262e3a; --link:#73b0f4;
    --blue:#2a78d6; --orange:#eb6834; --green:#1baf7a;
    --blue-soft:rgba(42,120,214,.28); --orange-soft:rgba(235,104,52,.26); --green-soft:rgba(27,175,122,.24);
    --code-bg:#1b222c; --tip-bg:#202834; --shadow:0 1px 3px rgba(0,0,0,.4);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:16px;line-height:1.55;overflow-x:hidden}
main{max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:clamp(1.5rem,4vw,2.2rem);line-height:1.2;margin:0 0 8px}
h2{font-size:1.3rem;margin:0 0 6px}
h3{font-size:1.05rem;margin:18px 0 6px}
p{margin:8px 0}
a{color:var(--link)}
code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace;font-size:.88em}
code{background:var(--code-bg);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;margin:8px 0}
pre code{background:none;padding:0;overflow-wrap:normal}
.muted{color:var(--muted)}
.small{font-size:.86rem}
header.hero{padding:20px 0 8px}
.kicker{color:var(--muted);font-size:.9rem;letter-spacing:.02em;margin:0 0 6px}
.lead{font-size:1.05rem;max-width:70ch}
section.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:18px;margin:18px 0;box-shadow:var(--shadow)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
section.card .tile{background:var(--surface2);box-shadow:none}
.tile .lab{color:var(--muted);font-size:.82rem}
.tile .val{font-size:1.45rem;font-weight:650;line-height:1.25;margin:2px 0;font-variant-numeric:tabular-nums}
.tile .sub{color:var(--muted);font-size:.82rem}
.c-blue{color:var(--blue)} .c-orange{color:var(--orange)} .c-green{color:var(--green)}
.arrow{color:var(--muted);font-weight:400;padding:0 2px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button{font:inherit;font-size:.92rem;color:var(--text);background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:6px 14px;cursor:pointer;min-height:36px}
button:hover{border-color:var(--blue)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
button:focus-visible,input:focus-visible,.comp:focus-visible{outline:3px solid var(--orange);outline-offset:2px}
input[type=range]{flex:1 1 200px;min-width:140px;accent-color:var(--blue);height:28px}
.caption{background:var(--surface2);border-left:4px solid var(--blue);border-radius:6px;padding:10px 12px;margin:10px 0;min-height:3.2em}
.caption b{display:block}
.scroll-x{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;color:var(--muted);margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:18px;height:0;border-top:3px solid var(--blue)}
.sw.o{border-color:var(--orange)} .sw.g{border-color:var(--green)} .sw.m{border-color:var(--muted)}
.sw.dash{border-top-style:dashed}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--blue)}
.dot.o{background:var(--orange)} .dot.g{background:var(--green)} .dot.m{background:var(--muted)}
.dot.hollow{background:none;border:2px solid var(--muted)}

/* architecture diagram */
#arch-svg{display:block;width:100%;min-width:760px;height:auto}
#arch-svg text{font-family:inherit;pointer-events:none}
.a-lab{font-size:13px;fill:var(--text)}
.a-sub{font-size:11px;fill:var(--muted)}
.a-mono{font-size:10.5px;fill:var(--muted);font-family:ui-monospace,Consolas,monospace}
.comp{cursor:pointer}
.comp rect{fill:var(--surface);stroke:var(--border);stroke-width:1.2;transition:fill .25s,stroke .25s}
.comp.box rect{fill:var(--surface2)}
.comp.node rect{fill:var(--surface)}
.comp.vllm rect{stroke:var(--blue)}
.comp.am rect{stroke:var(--orange)}
.comp.gym rect{stroke:var(--green)}
.comp.on rect{fill:var(--blue-soft);stroke:var(--blue);stroke-width:2}
.comp.am.on rect{fill:var(--orange-soft);stroke:var(--orange)}
.comp.gym.on rect{fill:var(--green-soft);stroke:var(--green)}
.comp:hover rect,.comp.sel rect{stroke:var(--text);stroke-width:2}
.link{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4;fill:none;opacity:.7}
.pkt{stroke:var(--surface);stroke-width:1}

/* data flow */
ol.flow{list-style:none;margin:12px 0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;counter-reset:st}
ol.flow li{position:relative;border:1px solid var(--border);border-radius:10px;padding:10px 12px 14px;background:var(--surface2);
  transition:border-color .3s,background .3s,opacity .3s;opacity:.72;overflow:hidden}
ol.flow li .num{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;
  background:var(--border);color:var(--text);font-size:.8rem;font-weight:650;margin-right:6px}
ol.flow li .ttl{font-weight:650}
ol.flow li .det{display:block;color:var(--muted);font-size:.84rem;margin-top:4px}
ol.flow li.done{opacity:1}
ol.flow li.done .num{background:var(--green);color:#fff}
ol.flow li.active{opacity:1;border-color:var(--orange);background:var(--orange-soft)}
ol.flow li.active .num{background:var(--orange);color:#fff}
ol.flow li .prog{position:absolute;left:0;bottom:0;height:4px;width:0;background:var(--orange)}
ol.flow li.active.run .prog{animation:prog var(--dur,2.6s) linear forwards}
@keyframes prog{from{width:0}to{width:100%}}
.side{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:10px 0}
.side div{border:1px dashed var(--border);border-radius:10px;padding:8px 12px;font-size:.86rem;color:var(--muted);transition:border-color .3s,color .3s,background .3s}
.side div b{color:var(--text)}
.side div.lit{border-color:var(--blue);border-style:solid;background:var(--blue-soft);color:var(--text)}

/* charts */
.chart-wrap{position:relative;margin:8px 0}
.chart-wrap svg{display:block;width:100%;height:auto;touch-action:pan-y}
.chart-wrap svg text{font-family:inherit;fill:var(--muted);font-size:11px}
.chart-wrap svg text.ptitle{fill:var(--text);font-size:12.5px;font-weight:600}
.gridl{stroke:var(--grid);stroke-width:1}
.axisl{stroke:var(--border);stroke-width:1}
.tip{position:absolute;z-index:5;pointer-events:none;background:var(--tip-bg);color:var(--text);border:1px solid var(--border);
  border-radius:8px;padding:8px 10px;font-size:.82rem;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.18);
  max-width:260px;display:none;font-variant-numeric:tabular-nums}
.tip b{display:block;margin-bottom:2px}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:10px 0}
.readout div{background:var(--surface2);border-radius:8px;padding:8px 10px}
.readout .lab{font-size:.78rem;color:var(--muted)}
.readout .val{font-size:1.1rem;font-weight:650;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:8px 0}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--muted);font-weight:600}
.tbl-x{overflow-x:auto}

/* before / after bars */
.bars{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px 28px;margin:12px 0}
.bg h3{margin:0 0 4px;font-size:.98rem}
.bg .scale{font-size:.78rem;color:var(--muted);font-weight:400}
.brow{display:grid;grid-template-columns:52px 1fr 70px;gap:8px;align-items:center;margin:4px 0;font-size:.88rem}
.btrack{background:var(--surface2);border-radius:5px;height:18px;overflow:hidden}
.bar{height:100%;width:0;border-radius:5px;background:var(--blue);transition:width 1.1s cubic-bezier(.2,.7,.2,1)}
.bar.t{background:var(--orange)}
.bval{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.delta{display:inline-block;font-size:.8rem;border-radius:999px;padding:1px 9px;margin-top:2px;background:var(--surface2);color:var(--muted)}
.delta.good{background:var(--green-soft);color:var(--text);border:1px solid var(--green)}
ul.clean{padding-left:1.2em;margin:8px 0}
ul.clean li{margin:6px 0}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:0 28px}
.badge{display:inline-block;font-size:.78rem;border-radius:999px;padding:1px 9px;border:1px solid var(--orange);background:var(--orange-soft);color:var(--text);white-space:nowrap}
.badge.done{border-color:var(--green);background:var(--green-soft)}
footer{color:var(--muted);font-size:.84rem;padding:8px 0 28px}
@media (max-width:480px){
  section.card{padding:14px 12px}
  .brow{grid-template-columns:44px 1fr 62px}
}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{transition:none !important;animation:none !important}
}
</style>
</head>
<body>
<main>

<header class="hero">
  <p class="kicker">Training run report &middot; 2026-11-24 &middot; cluster Rhea</p>
  <h1>Reinforcement-learning fine-tune of Nemotron 3.5 Lightning on T20 cricket questions</h1>
  <p class="lead">Goal: fine-tune <code>nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16</code> on T20 cricket
  rate and milestone questions with reinforcement learning and LoRA using NVIDIA NeMo RL, and evaluate the result
  with NeMo Gym. The run used 4 nodes &times; 4 NVIDIA Vera Rubin GPUs (286 GB each, aarch64, Slurm).</p>
  <div class="tiles">
    <div class="tile"><div class="lab">Held-out accuracy</div>
      <div class="val"><span class="c-blue">86.50%</span><span class="arrow">&rarr;</span><span class="c-orange">97.71%</span></div>
      <div class="sub">base &rarr; base + adapter (+11.2 points)</div></div>
    <div class="tile"><div class="lab">Mean output tokens per answer</div>
      <div class="val"><span class="c-blue">1,565</span><span class="arrow">&rarr;</span><span class="c-orange">107</span></div>
      <div class="sub">about 15&times; shorter</div></div>
    <div class="tile"><div class="lab">Training steps</div>
      <div class="val">100</div>
      <div class="sub">512 rollouts per step, two 90-minute-limit jobs</div></div>
    <div class="tile"><div class="lab">GPUs</div>
      <div class="val">16</div>
      <div class="sub">4 nodes &times; 4 Vera Rubin, 286 GB each</div></div>
  </div>
  <p class="small muted">Colour key used throughout: <span class="dot"></span> blue = base model / primary series,
  <span class="dot o"></span> orange = tuned model / highlight, <span class="dot g"></span> green = positive change or reward.</p>
</header>

<noscript><p><b>This report needs JavaScript for its diagrams and charts.</b> The text sections remain readable without it.</p></noscript>

<!-- ============================ 2. architecture ============================ -->
<section class="card" id="sec-arch" aria-labelledby="h-arch">
  <h2 id="h-arch">System architecture</h2>
  <p>One Slurm job spans four nodes. Ray orchestrates the work: the Ray head runs on node 1 and Ray workers on
  nodes 2&ndash;4. On every GPU a vLLM generation worker and a NeMo Automodel policy worker share the same GPU.
  The NeMo Gym servers and the driver script run on the head node. Hover, click or tab to a component for a description.</p>
  <div class="controls">
    <button type="button" class="primary" id="arch-play" aria-pressed="false">Pause</button>
    <button type="button" id="arch-next">Next phase</button>
    <span class="small muted" id="arch-phase"></span>
  </div>
  <div class="scroll-x" tabindex="-1">
    <svg id="arch-svg" viewBox="0 0 960 530" role="group" aria-label="Architecture diagram: Slurm job with four nodes, Ray head and workers, vLLM and Automodel workers on each GPU, NeMo Gym servers and driver script on the head node"></svg>
  </div>
  <div class="legend">
    <span><span class="dot m"></span>orchestration (Ray tasks)</span>
    <span><span class="dot"></span>generation requests and answers (vLLM)</span>
    <span><span class="dot g"></span>verification and rewards (NeMo Gym)</span>
    <span><span class="dot o"></span>policy update and refit (Automodel)</span>
  </div>
  <div class="caption" id="arch-cap"></div>
  <div class="caption" id="arch-info" aria-live="polite" style="border-left-color:var(--orange)"><b>Component</b>Select a component in the diagram to see what it does.</div>
</section>

<!-- ============================ 3. data flow ============================ -->
<section class="card" id="sec-flow" aria-labelledby="h-flow">
  <h2 id="h-flow">Data flow of one GRPO step</h2>
  <p>Method: GRPO in NeMo RL. Each step samples 64 questions and generates 8 answers for each, giving 512 rollouts.
  The sequence below repeats 100 times.</p>
  <div class="controls">
    <button type="button" class="primary" id="flow-play" aria-pressed="false">Pause</button>
    <button type="button" id="flow-prev">Previous stage</button>
    <button type="button" id="flow-next">Next stage</button>
    <span class="small muted" id="flow-pos"></span>
  </div>
  <ol class="flow" id="flow-list"></ol>
  <div class="caption" id="flow-cap"></div>
  <h3>Side outputs</h3>
  <div class="side" id="flow-side">
    <div data-side="wandb"><b>W&amp;B metrics</b><br>Reward, answer length and step time logged each step.</div>
    <div data-side="ckpt"><b>Checkpoints</b><br>Every 5 steps; the LoRA adapter is 95 MB.</div>
    <div data-side="gpu"><b>GPU logs</b><br>Sampled every 15 s on all nodes for the whole job.</div>
  </div>
  <h3>Settings</h3>
  <div class="cols small">
    <ul class="clean">
      <li><b>Generation:</b> vLLM 0.33 colocated on the training GPUs; temperature 1.0, top-p 0.95, max 4,096 new tokens.</li>
      <li><b>Reward:</b> NeMo Gym environment <code>cricket_qa</code>. A finished, exactly-correct final number scores
        <code>1 &minus; 0.4 &times; output_tokens / 4096</code>; anything else scores 0.</li>
      <li><b>Loss:</b> leave-one-out baseline, normalized rewards, ratio clip 0.2, KL penalty 0.</li>
    </ul>
    <ul class="clean">
      <li><b>Policy training:</b> NeMo Automodel backend (Transformer Engine attention, grouped-GEMM experts, expert parallel size 4).</li>
      <li><b>LoRA:</b> rank 32, alpha 64, on attention q/k/v/o, Mamba in_proj and shared-expert up/down projections.</li>
      <li><b>Optimizer:</b> AdamW, learning rate 1e-4, 10-step warm-up. 100 steps, checkpoint every 5.</li>
    </ul>
  </div>
</section>

<!-- ============================ 4. training replay ============================ -->
<section class="card" id="sec-train" aria-labelledby="h-train">
  <h2 id="h-train">Training replay</h2>
  <p>Per-step averages over the 512 rollouts. Use the slider or the play button to replay steps 1&ndash;100;
  hover or tap the chart to inspect a step, click to move the slider there.</p>
  <div class="controls">
    <button type="button" class="primary" id="tr-play" aria-pressed="false">Play</button>
    <label for="tr-slider" class="small muted">Step</label>
    <input type="range" id="tr-slider" min="1" max="100" step="1" value="100" aria-label="Training step">
  </div>
  <div class="readout" id="tr-readout" aria-live="off"></div>
  <div class="chart-wrap" id="tr-wrap">
    <svg id="tr-svg" role="img" aria-label="Three stacked line charts over training steps 1 to 100: average reward, mean answer length in tokens, and step time in seconds"></svg>
    <div class="tip" id="tr-tip"></div>
  </div>
  <div class="legend">
    <span><span class="sw g"></span>average reward</span>
    <span><span class="sw o"></span>mean answer length (tokens)</span>
    <span><span class="sw"></span>step time (s)</span>
    <span><span class="sw m dash"></span>resume point: job 131714 restarts from the step-65 checkpoint</span>
    <span><span class="dot hollow"></span>steps 66&ndash;70 from job 131698 (superseded by the resumed job)</span>
  </div>
  <div class="tbl-x">
  <table>
    <thead><tr><th>Slurm job</th><th>Window (PST)</th><th>Steps</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td>131698</td><td>12:37&ndash;14:07</td><td>1&ndash;70</td><td>Hit its 90-minute limit at step 70. Steps 1&ndash;65 are used in the final curve.</td></tr>
      <tr><td>131714</td><td>14:14&ndash;14:45</td><td>66&ndash;100</td><td>Resumed from the step-65 checkpoint and finished step 100. Logs to a second W&amp;B run.</td></tr>
    </tbody>
  </table>
  </div>
  <p class="small muted">About 100 minutes of step time in total (<span id="tr-total"></span>). Long jobs waited hours in the queue,
  so training ran as 90-minute jobs that resume from checkpoints. Step 66 of the resumed job is slower than its
  neighbours; it is the first step after the restart.</p>
</section>

<!-- ============================ 5. GPU telemetry ============================ -->
<section class="card" id="sec-gpu" aria-labelledby="h-gpu">
  <h2 id="h-gpu">GPU telemetry, minute by minute</h2>
  <p>Each point is one minute, averaged over all 16 GPUs (4 nodes &times; 4) from logs sampled every 15 s. The interval
  between the two jobs (14:07 to 14:14) has no data and is shown as a gap.</p>
  <div class="tiles" id="gpu-tiles"></div>
  <div class="controls">
    <label for="gpu-slider" class="small muted">Inspect minute</label>
    <input type="range" id="gpu-slider" min="0" max="1" step="1" value="0" aria-label="Minute to inspect in the GPU telemetry charts">
  </div>
  <div class="chart-wrap" id="gpu-wrap">
    <svg id="gpu-svg" role="img" aria-label="Four stacked line charts on a shared time axis: GPU utilisation, GPU memory average and peak, power per GPU and total, and temperature"></svg>
    <div class="tip" id="gpu-tip"></div>
  </div>
  <div class="legend">
    <span><span class="sw"></span>average over 16 GPUs</span>
    <span><span class="sw o dash"></span>peak memory on any GPU in that minute</span>
    <span><span class="sw m"></span>shaded band: no job running (between jobs)</span>
  </div>
  <p class="small muted">Memory is logged in MiB and plotted as GB with 1 GB = 1,024 MiB; each GPU has 286,524 MiB.
  Total power is the per-GPU average &times; 16. Energy is estimated as the sum over logged minutes of
  average power &times; 16 / 60 / 1000; it covers the GPUs only (no CPUs, cooling or the idle gap), and the first and
  last minute of each job contain fewer samples but are counted as full minutes. Utilisation is a per-minute mean across
  all GPUs, so short busy and idle phases within a minute average out.</p>
</section>

<!-- ============================ 6. before / after ============================ -->
<section class="card" id="sec-eval" aria-labelledby="h-eval">
  <h2 id="h-eval">Before and after: held-out evaluation</h2>
  <p>Evaluated with NeMo Gym (job 132709): 600 held-out questions &times; 4 sampled answers = 2,400 answers for each
  model. vLLM served the adapter as a LoRA module on top of the unchanged base model.</p>
  <div class="controls">
    <button type="button" id="ev-replay">Replay bars</button>
    <span class="legend" style="margin:0"><span><span class="dot"></span>base</span><span><span class="dot o"></span>base + adapter</span></span>
  </div>
  <div class="bars" id="ev-bars"></div>
  <div class="tbl-x">
  <table>
    <thead><tr><th>Metric</th><th>Base</th><th>Base + adapter</th></tr></thead>
    <tbody id="ev-table"></tbody>
  </table>
  </div>
</section>

<!-- ============================ 7. insights ============================ -->
<section class="card" id="sec-tools" aria-labelledby="h-tools">
  <h2 id="h-tools">Tools and SDKs used</h2>
  <p>Which tool did which job, for training with NeMo RL and for evaluation with NeMo Gym. Versions are the ones
  installed for this run.</p>
  <h3>Training (NeMo RL)</h3>
  <div class="tbl-x">
  <table>
    <thead><tr><th>Function</th><th>Tool or SDK</th><th>How it was used</th></tr></thead>
    <tbody>
      <tr><td>Training loop and algorithm</td><td>NeMo RL (<code>nemo_rl</code>, main branch, commit a03ffa1)</td><td>GRPO with ratio clipping, importance-sampling correction and a leave-one-out baseline. Driver script <code>examples/nemo_gym/run_grpo_nemo_gym.py</code>.</td></tr>
      <tr><td>Process orchestration</td><td>Ray</td><td>Ray head on the first node and a Ray worker on each of the other three, started by the Slurm job script; NeMo RL attaches to that cluster and runs the trainer, vLLM and NeMo Gym as Ray actors.</td></tr>
      <tr><td>Policy training backend</td><td>NeMo Automodel 0.6.1 on PyTorch 2.13</td><td>Its own Nemotron implementation, sharded across the 16 GPUs, with its own LoRA layers (not <code>peft</code>). Saves the adapter with PEFT-style names.</td></tr>
      <tr><td>Compiled kernels for training</td><td>Transformer Engine 2.18, flash-attn 2.8.1, mamba-ssm 2.3.2, causal-conv1d 1.6.2, DeepEP, grouped GEMM</td><td>Built from source on a Rhea node, about 30 minutes the first time.</td></tr>
      <tr><td>Generating answers</td><td>vLLM 0.33 on PyTorch 2.14, with FlashInfer 0.7.2</td><td>One engine per GPU, sharing the GPU with the trainer, exposed as an OpenAI-compatible HTTP server.</td></tr>
      <tr><td>Weight sync, trainer to vLLM</td><td>NeMo RL refit</td><td><code>lora_refit_mode: merged</code>: the LoRA weights are merged into full weights and pushed into the vLLM engines after every step.</td></tr>
      <tr><td>Environment and reward</td><td>NeMo Gym (version pinned by NeMo RL)</td><td>Cricket environment <code>cricket_qa</code>: an agent server, a resources server that checks the final number, and a model proxy that calls vLLM over HTTP.</td></tr>
      <tr><td>Configuration</td><td>Hydra / OmegaConf YAML</td><td><code>nemorl/cricket_rhea.yaml</code>, which inherits NeMo RL's <code>grpo_math_1B.yaml</code>.</td></tr>
      <tr><td>Python environments</td><td><code>uv</code>, Python 3.13.14</td><td>One environment per worker type, stored on Lustre.</td></tr>
      <tr><td>JIT kernel compiler</td><td>Triton 3.7 (trainer) and 3.8 (vLLM)</td><td>Needs the project's <code>sitecustomize.py</code> patch to run on Vera Rubin.</td></tr>
      <tr><td>Container</td><td>NVIDIA PyTorch development image for Rubin (arm64)</td><td>Used for its compilers and CUDA toolkit; NeMo RL's Python stack comes from <code>uv</code>.</td></tr>
    </tbody>
  </table>
  </div>
  <h3>Evaluation (NeMo Gym)</h3>
  <div class="tbl-x">
  <table>
    <thead><tr><th>Function</th><th>Tool or SDK</th><th>How it was used</th></tr></thead>
    <tbody>
      <tr><td>Environment evaluation</td><td>NeMo Gym 0.7.0 command line (<code>gym eval run</code>)</td><td>Runs the <code>cricket_qa</code> environment on the 600 held-out questions, 4 sampled answers each, for the base model and for base + adapter.</td></tr>
      <tr><td>Serving for the evaluation</td><td>vLLM 0.31, NVIDIA Rubin build</td><td>Serves the base model and the adapter side by side (<code>--enable-lora --lora-modules</code>), reasoning parser <code>nemotron_v3</code>.</td></tr>
      <tr><td>Scoring</td><td>Project code <code>cricket_qa.py</code> inside the NeMo Gym resources server</td><td>Exact check of the final number; records correct, finished, output tokens and reasoning tokens for every answer.</td></tr>
      <tr><td>Summaries</td><td>Project code <code>gym_summary.py</code></td><td>Accuracy, pass@k, average output tokens, cut-off rate and no-answer rate, overall and per question type; logged to Weights &amp; Biases.</td></tr>
      <tr><td>Adapter compatibility check</td><td>Hugging Face <code>transformers</code> 5.22.2 and <code>peft</code> 0.21.2</td><td>Applies the adapter to the base model, merges it, and compares answers with the base model on sample questions.</td></tr>
    </tbody>
  </table>
  </div>
  <h3>Shared infrastructure</h3>
  <div class="tbl-x">
  <table>
    <thead><tr><th>Function</th><th>Tool or SDK</th><th>How it was used</th></tr></thead>
    <tbody>
      <tr><td>Job scheduling</td><td>Slurm with enroot / pyxis containers</td><td>Training as two 90-minute, 4-node jobs; evaluation, check and upload as one-node jobs.</td></tr>
      <tr><td>Storage</td><td>Lustre</td><td>Models, data, Python environments, checkpoints and logs.</td></tr>
      <tr><td>Metrics</td><td>Weights &amp; Biases</td><td>Training metrics per step and the evaluation summary.</td></tr>
      <tr><td>Adapter hosting</td><td>Hugging Face Hub (<code>huggingface_hub</code>)</td><td>Adapter only, public repository, with a model card.</td></tr>
      <tr><td>GPU telemetry</td><td><code>nvidia-smi</code></td><td>Utilisation, memory, power and temperature every 15 s on every node.</td></tr>
      <tr><td>Model</td><td><code>nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16</code></td><td>Base model; its weights are not changed.</td></tr>
      <tr><td>Dataset</td><td><code>Balab2021/CricketData-T20-Reasoning-Qwen3.8-Full</code></td><td>458,702 training questions, 600 held out for evaluation.</td></tr>
    </tbody>
  </table>
  </div>
</section>

<section class="card" id="sec-ins" aria-labelledby="h-ins">
  <h2 id="h-ins">Insights and caveats</h2>
  <div class="cols">
    <div>
      <h3>What changed</h3>
      <ul class="clean">
        <li>Accuracy rose by 11.2 points (86.50% &rarr; 97.71%) while answers became about 15&times; shorter (1,565 &rarr; 107 mean output tokens).</li>
        <li>The biggest gain is on <code>chase_rate</code> questions: 68.1% &rarr; 94.6%. <code>run_rate</code> went from 91.75% to 99.00%.</li>
        <li><code>milestone</code> questions are essentially unchanged (99.62% &rarr; 99.50%).</li>
        <li>The base model lost 9.4% of its answers to the 4,096-token limit; the tuned model lost none.</li>
        <li>The model largely stopped writing out long reasoning. The length term in the reward pushed in this direction:
          a correct answer scores more when it uses fewer tokens.</li>
        <li>Steps got faster as answers shortened: 233 s at step 1, about 40 s or less at the end.</li>
      </ul>
    </div>
    <div>
      <h3>Caveats</h3>
      <ul class="clean">
        <li>Single run, single seed. Differences of a point or less (for example the milestone change) are not significant.</li>
        <li>Only the adapter (95 MB) is produced. The base weights are unchanged, and the adapter must be applied to or
          merged into the base model at inference.</li>
        <li>The reward only checks the final number and the answer length. The evaluation measures the same narrow task
          (T20 rate and milestone questions); nothing here measures behaviour on other tasks.</li>
        <li>Shorter answers mean less visible working. If written-out reasoning matters for a use case, this adapter
          trades it away for brevity.</li>
        <li>Training rewards (sampled at temperature 1.0 during training) and evaluation accuracy are different
          measurements and should not be compared directly.</li>
      </ul>
    </div>
  </div>

  <h3>Vera Rubin notes</h3>
  <ul class="clean">
    <li>Builds needed <code>TORCH_CUDA_ARCH_LIST="9.0 10.0"</code>.</li>
    <li>The pip-installed cuDNN had to come first on the library path.</li>
    <li>Stock Triton cannot target Rubin; a Triton patch is loaded through <code>sitecustomize.py</code>.</li>
    <li>Refit uses <code>lora_refit_mode: merged</code>: after each step the LoRA weights are merged and pushed into the vLLM engines.</li>
    <li>The resumed job logs to a second W&amp;B run, so the training curve is split over two runs (links below).</li>
  </ul>

  <h3>How to use the adapter</h3>
  <p>The repository contains the LoRA adapter only. Load the base model and apply the adapter, optionally merging it.
  All three ways were checked on 2026-11-24: PEFT apply and merge with transformers 5.22.2 and peft 0.21.2, and vLLM 0.31 serving the adapter as a LoRA module (used for the evaluation above). The adapter file holds 198 tensors; 12 belong to two <code>mtp</code> layers that the transformers version of the model does not have, so PEFT loads the other 186.</p>
  <p class="small"><b>PEFT</b> (apply with <code>PeftModel.from_pretrained</code>, optionally merge with <code>merge_and_unload()</code>):</p>
<pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
ADAPTER = "Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-NeMo-RL-Rhea"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)   # apply the adapter
model = model.merge_and_unload()                    # optional: merge into the base weights</code></pre>
  <p class="small"><b>vLLM</b> (serve the adapter as a LoRA module, as in the evaluation):</p>
<pre><code>vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --enable-lora \
  --lora-modules cricket=Balab2021/Nemotron-3.5-Lightning-30B-A3B-Cricket-NeMo-RL-Rhea
# then request model "cricket" for the tuned behaviour, or the base model name for the base</code></pre>

  <h3>Links</h3>
  <ul class="clean">
    <li>W&amp;B training run, steps 1&ndash;70 (job 131698): <a href="__WANDB_RUN_1__">__WANDB_RUN_1__</a></li>
    <li>W&amp;B training run, steps 66&ndash;100 (job 131714): <a href="__WANDB_RUN_2__">__WANDB_RUN_2__</a></li>
    <li>NeMo Gym evaluation, W&amp;B run <code>zovlu0p2</code> in the same project (job 132709): <a href="__WANDB_EVAL__">__WANDB_EVAL__</a></li>
    <li>Hugging Face adapter repository (public, adapter only) <span class="badge __HF_CLASS__">__HF_STATUS__</span>:
      <a href="__HF_URL__">__HF_URL__</a></li>
  </ul>
</section>

<footer>
  Generated by <code>report/build_nemorl.py</code> from <code>report/data/nemorl_raw.txt</code>. Training curves and GPU
  telemetry are computed from that file; evaluation figures are from the NeMo Gym evaluation job 132709. Times are PST.
</footer>
</main>

<script type="application/json" id="report-data">__DATA__</script>
<script>
(function(){
'use strict';
var D = JSON.parse(document.getElementById('report-data').textContent);
var RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var NS = 'http://www.w3.org/2000/svg';
function $(s){ return document.querySelector(s); }
function el(tag, attrs, parent, text){
  var e = document.createElementNS(NS, tag);
  if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, cls, parent, html){
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  if (parent) parent.appendChild(e);
  return e;
}
function fmt(v, d){ d = d || 0; return Number(v).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d}); }
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }
function ticks(lo, hi, n){
  var span = (hi - lo) || 1, raw = span / n, p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
  var f = raw / p, st = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  var a = Math.floor(lo / st + 1e-9) * st, b = Math.ceil(hi / st - 1e-9) * st, out = [];
  for (var v = a; v <= b + st * 1e-6; v += st) out.push(+v.toFixed(8));
  return out;
}
function clear(node){ while (node.firstChild) node.removeChild(node.firstChild); }
function onVisible(node, cb){
  if (!('IntersectionObserver' in window)) { cb(); return; }
  var io = new IntersectionObserver(function(es){
    for (var i = 0; i < es.length; i++) if (es[i].isIntersecting) { io.disconnect(); cb(); return; }
  }, {threshold: 0.25});
  io.observe(node);
}
function onResize(node, cb){
  var raf = 0, lastW = -1;
  function go(){ if (raf) return; raf = requestAnimationFrame(function(){ raf = 0; var w = node.clientWidth; if (w !== lastW) { lastW = w; cb(); } }); }
  if ('ResizeObserver' in window) new ResizeObserver(go).observe(node);
  window.addEventListener('resize', go);
}
function placeTip(tip, wrap, px, py){
  tip.style.display = 'block';
  var ww = wrap.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
  var left = px + 14;
  if (left + tw > ww) left = px - 14 - tw;
  left = clamp(left, 0, Math.max(0, ww - tw));
  var top = clamp(py - th / 2, 0, Math.max(0, wrap.clientHeight - th));
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}
function setPlayLabel(btn, playing){
  btn.textContent = playing ? 'Pause' : 'Play';
  btn.setAttribute('aria-pressed', playing ? 'true' : 'false');
}

/* ====================================================================== */
/* 2. Architecture diagram                                                */
/* ====================================================================== */
(function(){
  var svg = $('#arch-svg'), info = $('#arch-info'), cap = $('#arch-cap'), phaseLab = $('#arch-phase');
  var playBtn = $('#arch-play'), nextBtn = $('#arch-next');
  var DESC = {
    slurm: ['Slurm job', 'One Slurm allocation of 4 nodes, each with 4 NVIDIA Vera Rubin GPUs (286 GB each, aarch64). Training ran as two jobs with a 90-minute limit: 131698 (steps 1 to 70) and 131714 (resumed from the step-65 checkpoint, steps 66 to 100).'],
    node: ['Compute node', 'Each node has 4 Vera Rubin GPUs. Node 1 hosts the Ray head, the driver script and the NeMo Gym servers; nodes 2 to 4 each run a Ray worker.'],
    driver: ['Driver script: run_grpo_nemo_gym.py', 'Runs the GRPO loop in NeMo RL: samples 64 questions, requests 8 answers for each, collects rewards from NeMo Gym, starts the policy update, triggers the refit, and logs metrics and checkpoints.'],
    rayhead: ['Ray head (node 1)', 'Ray orchestration. The head schedules generation and policy-training work across the GPUs of all four nodes.'],
    rayworker: ['Ray worker (nodes 2 to 4)', 'Joins the Ray cluster and runs the vLLM generation workers and Automodel policy workers on that node.'],
    gymgroup: ['NeMo Gym servers (head node)', 'Three servers on the head node: the agent, the resources server and the model proxy. Together they run the cricket_qa environment that turns answers into rewards.'],
    proxy: ['NeMo Gym model proxy', 'The model endpoint the agent talks to. It forwards generation requests to the vLLM generation workers on the training GPUs.'],
    agent: ['NeMo Gym agent', 'Runs each rollout: sends the question to the model through the model proxy, receives the answer, and passes it to the resources server for verification.'],
    resources: ['NeMo Gym resources server', 'Verifies each answer for the cricket_qa environment. A finished, exactly-correct final number scores 1 - 0.4 x output_tokens / 4096; anything else scores 0.'],
    gpu: ['Vera Rubin GPU', 'One of 16 GPUs with 286,524 MiB of memory each. A vLLM generation worker and an Automodel policy worker share each GPU.'],
    vllm: ['vLLM generation worker', 'vLLM 0.33 colocated on the training GPU. Samples answers at temperature 1.0, top-p 0.95, up to 4,096 new tokens. After every step it receives the merged LoRA weights (refit).'],
    automodel: ['Automodel policy worker', 'NeMo Automodel backend (Transformer Engine attention, grouped-GEMM experts, expert parallel size 4). Computes the GRPO loss and updates the LoRA weights (rank 32, alpha 64) with AdamW at learning rate 1e-4.']
  };
  var groups = {}, sel = null;
  function showInfo(key){
    var d = DESC[key];
    info.innerHTML = '';
    h('b', null, info).textContent = d[0];
    info.appendChild(document.createTextNode(d[1]));
  }
  function comp(key, x, y, w, hgt, lines, opt){
    opt = opt || {};
    var g = el('g', {'class': 'comp ' + (opt.cls || ''), 'data-k': key}, svg);
    el('rect', {x: x, y: y, width: w, height: hgt, rx: opt.rx || 6}, g);
    if (opt.top) {
      el('text', {x: x + 9, y: y + 16, 'class': 'a-sub'}, g, lines[0]);
    } else {
      for (var i = 0; i < lines.length; i++) {
        el('text', {x: x + w / 2, y: y + hgt / 2 + (i - (lines.length - 1) / 2) * 14 + 4, 'text-anchor': 'middle',
          'class': i ? (opt.mono ? 'a-mono' : 'a-sub') : 'a-lab'}, g, lines[i]);
      }
    }
    if (opt.focus) {
      g.setAttribute('tabindex', '0');
      g.setAttribute('role', 'button');
      g.setAttribute('aria-label', DESC[key][0]);
    }
    g.addEventListener('mouseenter', function(){ showInfo(key); });
    g.addEventListener('focus', function(){ showInfo(key); });
    g.addEventListener('click', function(){ select(g, key); });
    g.addEventListener('keydown', function(e){
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(g, key); }
    });
    (groups[key] = groups[key] || []).push(g);
    return {x: x + w / 2, y: y + hgt / 2};
  }
  function select(g, key){
    if (sel) sel.classList.remove('sel');
    sel = g; g.classList.add('sel'); showInfo(key);
  }

  // containers
  comp('slurm', 8, 8, 944, 514, ['Slurm job: 4 nodes x 4 NVIDIA Vera Rubin GPUs (aarch64)'], {top: true, cls: 'box', rx: 10, focus: true});
  var nodeX = [25, 375, 570, 765], nodeW = [335, 170, 170, 170];
  for (var n = 0; n < 4; n++) {
    comp('node', nodeX[n], 34, nodeW[n], 478, [n === 0 ? 'Node 1 (head node)' : 'Node ' + (n + 1)], {top: true, cls: 'node', rx: 8, focus: n === 0});
  }
  // link layer (under leaf components)
  var linkLayer = el('g', null, svg);
  // leaves
  var C = {};
  C.driver = comp('driver', 35, 60, 155, 44, ['Driver script', 'run_grpo_nemo_gym.py'], {focus: true, mono: true});
  C.rayhead = comp('rayhead', 200, 60, 150, 44, ['Ray head'], {focus: true});
  comp('gymgroup', 35, 122, 155, 380, ['NeMo Gym servers'], {top: true, cls: 'box gym', focus: true});
  C.proxy = comp('proxy', 45, 148, 135, 100, ['Model proxy'], {cls: 'gym', focus: true});
  C.agent = comp('agent', 45, 262, 135, 100, ['Agent'], {cls: 'gym', focus: true});
  C.resources = comp('resources', 45, 376, 135, 114, ['Resources server', 'cricket_qa'], {cls: 'gym', focus: true, mono: true});
  C.workers = [];
  for (n = 1; n < 4; n++) C.workers.push(comp('rayworker', nodeX[n] + 10, 60, 150, 44, ['Ray worker'], {focus: n === 1}));
  C.vllm = []; C.am = [];
  for (n = 0; n < 4; n++) {
    var gx = n === 0 ? 200 : nodeX[n] + 10;
    for (var g = 0; g < 4; g++) {
      var gy = 122 + g * 96;
      comp('gpu', gx, gy, 150, 86, ['GPU ' + g + ' (286 GB)'], {top: true, cls: 'box', focus: n === 0 && g === 0});
      C.vllm.push(comp('vllm', gx + 6, gy + 24, 66, 54, ['vLLM'], {cls: 'vllm', focus: n === 0 && g === 0}));
      C.am.push(comp('automodel', gx + 78, gy + 24, 66, 54, ['Auto-', 'model'], {cls: 'am', focus: n === 0 && g === 0}));
    }
  }
  // fix second line style of the Automodel label (same weight as first line)
  groups.automodel.forEach(function(gr){ var t = gr.querySelectorAll('text'); if (t[1]) t[1].setAttribute('class', 'a-lab'); });
  // static links
  [[190, 200], [350, nodeX[1] + 10], [nodeX[1] + 160, nodeX[2] + 10], [nodeX[2] + 160, nodeX[3] + 10]].forEach(function(sg){
    el('line', {x1: sg[0], y1: C.rayhead.y, x2: sg[1], y2: C.rayhead.y, 'class': 'link'}, linkLayer);
  });
  el('line', {x1: C.driver.x, y1: 104, x2: C.driver.x, y2: 122, 'class': 'link'}, linkLayer);

  var COL = {ray: 'var(--muted)', gen: 'var(--blue)', gym: 'var(--green)', train: 'var(--orange)'};
  var PH = [
    {title: 'Dispatch', text: 'The driver script asks Ray for the next batch. The Ray head on node 1 hands generation work to the Ray workers on nodes 2 to 4.',
      on: ['driver', 'rayhead', 'rayworker'],
      flows: [[C.driver, C.rayhead, COL.ray, 0]].concat(C.workers.map(function(w){ return [C.rayhead, w, COL.ray, 0]; }))},
    {title: 'Generate', text: 'The agent sends 64 questions x 8 samples through the model proxy to the vLLM generation workers on all 16 GPUs; the 512 answers come back.',
      on: ['proxy', 'agent', 'vllm'],
      flows: [[C.agent, C.proxy, COL.gen, 1]].concat(C.vllm.map(function(v){ return [C.proxy, v, COL.gen, 1]; }))},
    {title: 'Verify and reward', text: 'The agent passes each answer to the resources server, which checks the final number and returns a reward. The rewards go back to the driver.',
      on: ['agent', 'resources', 'driver'],
      flows: [[C.agent, C.resources, COL.gym, 1], [C.agent, C.driver, COL.gym, 0]]},
    {title: 'Update the policy', text: 'The Automodel policy workers on all 16 GPUs compute the GRPO loss from the rollouts and rewards and update the LoRA weights.',
      on: ['driver', 'rayhead', 'rayworker', 'automodel'],
      flows: C.am.map(function(a){ return [C.driver, a, COL.train, 0]; })},
    {title: 'Refit', text: 'On each GPU the updated LoRA weights are merged and pushed into the vLLM engine that shares the GPU, so the next step samples from the new policy.',
      on: ['automodel', 'vllm'],
      flows: C.am.map(function(a, i){ return [a, C.vllm[i], COL.train, 0]; })}
  ];
  var pktLayer = el('g', {'aria-hidden': 'true'}, svg);
  PH.forEach(function(p){
    p.pk = p.flows.map(function(f){
      return el('circle', {r: p.flows.length > 8 ? 3.5 : 5, fill: f[2], 'class': 'pkt', visibility: 'hidden'}, pktLayer);
    });
  });

  var DUR = 2800, t = 0, last = null, playing = !RM, shown = -1, visible = true, rafId = 0;
  function render(){
    var p = Math.floor(t / DUR) % PH.length, f = (t % DUR) / DUR;
    if (p !== shown) {
      if (shown >= 0) {
        PH[shown].pk.forEach(function(c){ c.setAttribute('visibility', 'hidden'); });
        PH[shown].on.forEach(function(k){ groups[k].forEach(function(g){ g.classList.remove('on'); }); });
      }
      PH[p].pk.forEach(function(c){ c.setAttribute('visibility', 'visible'); });
      PH[p].on.forEach(function(k){ groups[k].forEach(function(g){ g.classList.add('on'); }); });
      cap.innerHTML = '';
      h('b', null, cap).textContent = 'Phase ' + (p + 1) + ' of ' + PH.length + ': ' + PH[p].title;
      cap.appendChild(document.createTextNode(PH[p].text));
      phaseLab.textContent = 'Phase ' + (p + 1) + ' of ' + PH.length;
      shown = p;
    }
    var e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
    PH[p].flows.forEach(function(fl, i){
      var q = fl[3] ? (f < 0.5 ? f * 2 : 2 - f * 2) : e;
      PH[p].pk[i].setAttribute('cx', (fl[0].x + (fl[1].x - fl[0].x) * q).toFixed(1));
      PH[p].pk[i].setAttribute('cy', (fl[0].y + (fl[1].y - fl[0].y) * q).toFixed(1));
    });
  }
  function frame(ts){
    rafId = 0;
    if (!playing || !visible) { last = null; return; }
    if (last != null) t += Math.min(100, ts - last);
    last = ts;
    render();
    rafId = requestAnimationFrame(frame);
  }
  function kick(){ if (playing && visible && !rafId) { last = null; rafId = requestAnimationFrame(frame); } }
  function setPlaying(v){ playing = v; setPlayLabel(playBtn, playing); kick(); }
  playBtn.addEventListener('click', function(){ setPlaying(!playing); });
  nextBtn.addEventListener('click', function(){
    var p = (Math.floor(t / DUR) + 1) % PH.length;
    t = p * DUR + DUR * 0.5;
    setPlaying(false);
    render();
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function(es){ visible = es[es.length - 1].isIntersecting; kick(); }, {threshold: 0.05}).observe(svg);
  }
  if (RM) t = DUR * 0.5;
  setPlayLabel(playBtn, playing);
  render();
  kick();
})();

/* ====================================================================== */
/* 3. Data flow of one GRPO step                                          */
/* ====================================================================== */
(function(){
  var list = $('#flow-list'), cap = $('#flow-cap'), pos = $('#flow-pos');
  var playBtn = $('#flow-play'), prevBtn = $('#flow-prev'), nextBtn = $('#flow-next');
  var ST = [
    {t: 'Dataset', d: '458,702 training questions', c: 'The training set holds 458,702 T20 cricket rate and milestone questions.', side: []},
    {t: 'Sample questions', d: '64 questions per step', c: 'The driver samples 64 questions for this step.', side: []},
    {t: 'Generate answers', d: '8 answers each = 512 rollouts', c: 'vLLM 0.33, colocated on the training GPUs, samples 8 answers for each question (temperature 1.0, top-p 0.95, up to 4,096 new tokens).', side: []},
    {t: 'Verify', d: 'NeMo Gym agent + resources server', c: 'The NeMo Gym agent and the resources server of the cricket_qa environment check every answer: is it finished, and is the final number exactly correct?', side: []},
    {t: 'Rewards', d: '1 \u2212 0.4 \u00d7 tokens / 4096, or 0', c: 'A finished, exactly-correct final number scores 1 \u2212 0.4 \u00d7 output_tokens / 4096. Anything else scores 0. Shorter correct answers therefore score higher.', side: ['wandb']},
    {t: 'GRPO loss', d: 'advantages vs leave-one-out baseline', c: 'Each answer\u2019s advantage is its reward compared with a leave-one-out baseline over the other answers to the same question. Rewards are normalized; ratio clip 0.2, KL penalty 0.', side: []},
    {t: 'Update LoRA', d: 'Automodel on 16 GPUs', c: 'The NeMo Automodel backend updates the LoRA weights (rank 32, alpha 64) on all 16 GPUs with AdamW, learning rate 1e-4 after a 10-step warm-up.', side: ['wandb', 'ckpt']},
    {t: 'Refit vLLM', d: 'merged weights \u2192 next step', c: 'The LoRA weights are merged and pushed into the vLLM engines (refit), so the next step generates with the updated policy. Then the loop starts again.', side: []}
  ];
  var items = ST.map(function(s, i){
    var li = h('li', null, list);
    h('span', 'num', li).textContent = String(i + 1);
    h('span', 'ttl', li).textContent = s.t;
    h('span', 'det', li).textContent = s.d;
    h('span', 'prog', li);
    li.addEventListener('click', function(){ stop(); go(i); });
    return li;
  });
  var sides = Array.prototype.slice.call(document.querySelectorAll('#flow-side div'));
  var DUR = 2600, cur = 0, playing = !RM, timer = 0, visible = true;
  list.style.setProperty('--dur', (DUR / 1000) + 's');
  function go(i){
    cur = (i + ST.length) % ST.length;
    items.forEach(function(li, k){
      li.classList.remove('run');
      li.classList.toggle('active', k === cur);
      li.classList.toggle('done', k < cur);
      if (k === cur) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
    });
    if (playing) { void items[cur].offsetWidth; items[cur].classList.add('run'); }
    sides.forEach(function(d){
      var k = d.getAttribute('data-side');
      d.classList.toggle('lit', k === 'gpu' || ST[cur].side.indexOf(k) >= 0);
    });
    cap.innerHTML = '';
    h('b', null, cap).textContent = 'Stage ' + (cur + 1) + ' of ' + ST.length + ': ' + ST[cur].t;
    cap.appendChild(document.createTextNode(ST[cur].c));
    pos.textContent = 'Stage ' + (cur + 1) + ' of ' + ST.length;
  }
  function schedule(){
    clearTimeout(timer);
    if (playing && visible) timer = setTimeout(function(){ go(cur + 1); schedule(); }, DUR);
  }
  function stop(){ playing = false; clearTimeout(timer); setPlayLabel(playBtn, false); }
  playBtn.addEventListener('click', function(){
    if (playing) { stop(); go(cur); }
    else { playing = true; setPlayLabel(playBtn, true); go(cur); schedule(); }
  });
  prevBtn.addEventListener('click', function(){ stop(); go(cur - 1); });
  nextBtn.addEventListener('click', function(){ stop(); go(cur + 1); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function(es){
      visible = es[es.length - 1].isIntersecting;
      if (visible && playing) { go(cur); schedule(); } else clearTimeout(timer);
    }, {threshold: 0.05}).observe(list);
  }
  setPlayLabel(playBtn, playing);
  go(0);
  schedule();
})();

/* ====================================================================== */
/* 4. Training replay                                                     */
/* ====================================================================== */
(function(){
  var S = D.steps, OLD = D.superseded, RES = D.resumeStep + 0.5, N = S.length;
  var svg = $('#tr-svg'), wrap = $('#tr-wrap'), tip = $('#tr-tip');
  var slider = $('#tr-slider'), playBtn = $('#tr-play'), readout = $('#tr-readout');
  var cum = [], acc = 0;
  S.forEach(function(r){ acc += r.sec; cum.push(acc); });
  var oldSec = OLD.reduce(function(a, r){ return a + r.sec; }, 0);
  $('#tr-total').textContent = fmt(acc / 60, 1) + ' minutes for the 100 steps of the final curve, plus ' +
    fmt(oldSec / 60, 1) + ' minutes for the superseded steps 66 to 70 of job ' + D.jobs[0];
  var P = [
    {k: 'r', title: 'Average reward per rollout (0 to 1)', color: 'var(--green)', d: 3},
    {k: 't', title: 'Mean answer length (output tokens)', color: 'var(--orange)', d: 0, zero: true},
    {k: 'sec', title: 'Step time (seconds)', color: 'var(--blue)', d: 0, zero: true}
  ];
  P.forEach(function(p){
    var vals = S.map(function(r){ return r[p.k]; }).concat(OLD.map(function(r){ return r[p.k]; }));
    var lo = p.zero ? 0 : Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    p.ticks = ticks(lo, p.k === 'r' ? 1 : hi, 4);
    p.lo = p.ticks[0]; p.hi = p.ticks[p.ticks.length - 1];
  });
  var cur = N, playing = false, timer = 0, hover = null, geo = null;

  function draw(){
    var W = Math.max(280, wrap.clientWidth), ML = 46, MR = 12, PHt = W < 520 ? 120 : 150, GAP = 46, TOP = 24;
    var H = TOP + P.length * (PHt + GAP) + 10, pw = W - ML - MR;
    geo = {W: W, ML: ML, pw: pw};
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    clear(svg);
    function X(s){ return ML + (s - 1) / (N - 1) * pw; }
    var xt = W < 420 ? [1, 25, 50, 75, 100] : [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    P.forEach(function(p, i){
      var y0 = TOP + i * (PHt + GAP), y1 = y0 + PHt;
      function Y(v){ return y1 - (v - p.lo) / (p.hi - p.lo) * PHt; }
      el('text', {x: ML, y: y0 - 8, 'class': 'ptitle'}, svg, p.title);
      p.ticks.forEach(function(v){
        el('line', {x1: ML, x2: ML + pw, y1: Y(v), y2: Y(v), 'class': 'gridl'}, svg);
        el('text', {x: ML - 6, y: Y(v) + 4, 'text-anchor': 'end'}, svg, p.k === 'r' ? v.toFixed(1) : fmt(v));
      });
      el('line', {x1: ML, x2: ML + pw, y1: y1, y2: y1, 'class': 'axisl'}, svg);
      xt.forEach(function(s){
        el('line', {x1: X(s), x2: X(s), y1: y1, y2: y1 + 4, 'class': 'axisl'}, svg);
        el('text', {x: X(s), y: y1 + 16, 'text-anchor': s === 100 ? 'end' : s === 1 ? 'start' : 'middle'}, svg, String(s));
      });
      // resume marker
      el('line', {x1: X(RES), x2: X(RES), y1: y0, y2: y1, stroke: 'var(--muted)', 'stroke-width': 1.3, 'stroke-dasharray': '5 4'}, svg);
      // ghost of the whole run, then progressive line
      var full = '', part = '';
      S.forEach(function(r, j){
        var seg = (j ? 'L' : 'M') + X(r.s).toFixed(1) + ' ' + Y(r[p.k]).toFixed(1);
        full += seg;
        if (r.s <= cur) part += seg;
      });
      el('path', {d: full, fill: 'none', stroke: p.color, 'stroke-width': 1.5, opacity: 0.18}, svg);
      if (part) el('path', {d: part, fill: 'none', stroke: p.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);
      OLD.forEach(function(r){
        if (r.s <= cur) el('circle', {cx: X(r.s), cy: Y(r[p.k]), r: 2.6, fill: 'var(--surface)', stroke: 'var(--muted)', 'stroke-width': 1.3}, svg);
      });
      var c = S[cur - 1];
      el('circle', {cx: X(c.s), cy: Y(c[p.k]), r: 4.5, fill: p.color, stroke: 'var(--surface)', 'stroke-width': 1.5}, svg);
      if (hover != null) {
        var hr = S[hover - 1];
        el('line', {x1: X(hover), x2: X(hover), y1: y0, y2: y1, stroke: 'var(--text)', 'stroke-width': 1, opacity: 0.5}, svg);
        el('circle', {cx: X(hover), cy: Y(hr[p.k]), r: 3.5, fill: 'var(--surface)', stroke: p.color, 'stroke-width': 2}, svg);
      }
    });
    var yb = TOP + P.length * (PHt + GAP) - GAP + 32;
    el('text', {x: ML, y: yb}, svg, 'Training step');
    el('text', {x: X(RES), y: yb, 'text-anchor': 'middle'}, svg, 'resume (65 | 66)');
  }
  function cell(lab, val){
    var d = h('div', null, readout);
    h('div', 'lab', d).textContent = lab;
    h('div', 'val', d).textContent = val;
  }
  function updateReadout(){
    var r = S[cur - 1];
    readout.innerHTML = '';
    cell('Step', r.s + ' of ' + N);
    cell('Slurm job', String(r.j));
    cell('Average reward', r.r.toFixed(3));
    cell('Mean answer length', fmt(r.t) + ' tokens');
    cell('Step time', fmt(r.sec, 1) + ' s');
    cell('Cumulative step time', fmt(cum[cur - 1] / 60, 1) + ' min');
    slider.setAttribute('aria-valuetext', 'Step ' + r.s + ', reward ' + r.r.toFixed(3) + ', ' + fmt(r.t) + ' tokens, ' + fmt(r.sec, 1) + ' seconds');
  }
  function set(s){ cur = clamp(s, 1, N); slider.value = cur; updateReadout(); draw(); }
  function stop(){ playing = false; clearInterval(timer); setPlayLabel(playBtn, false); }
  function play(){
    if (cur >= N) set(1);
    playing = true; setPlayLabel(playBtn, true);
    clearInterval(timer);
    timer = setInterval(function(){ if (cur >= N) { stop(); return; } set(cur + 1); }, 110);
  }
  playBtn.addEventListener('click', function(){ if (playing) stop(); else play(); });
  slider.addEventListener('input', function(){ if (playing) stop(); set(+slider.value); });
  function stepAt(e){
    var rc = svg.getBoundingClientRect(), px = (e.clientX - rc.left) * (geo.W / rc.width);
    return {s: clamp(Math.round((px - geo.ML) / geo.pw * (N - 1)) + 1, 1, N), px: e.clientX - rc.left, py: e.clientY - rc.top};
  }
  svg.addEventListener('pointermove', function(e){
    var q = stepAt(e), r = S[q.s - 1];
    hover = q.s; draw();
    var old = null;
    OLD.forEach(function(o){ if (o.s === q.s) old = o; });
    tip.innerHTML = '<b>Step ' + r.s + ' (job ' + r.j + ')</b>' +
      'Average reward: ' + r.r.toFixed(4) + '<br>Mean answer length: ' + fmt(r.t, 1) + ' tokens<br>Step time: ' + fmt(r.sec, 1) + ' s' +
      (old ? '<br><span class="muted">Job ' + old.j + ' (superseded): reward ' + old.r.toFixed(4) + ', ' + fmt(old.t, 1) + ' tokens, ' + fmt(old.sec, 1) + ' s</span>' : '');
    placeTip(tip, wrap, q.px, q.py);
  });
  svg.addEventListener('pointerleave', function(){ hover = null; tip.style.display = 'none'; draw(); });
  svg.addEventListener('click', function(e){ if (playing) stop(); set(stepAt(e).s); });
  onResize(wrap, draw);
  set(N);
  if (!RM) onVisible(wrap, function(){ if (!playing && cur === N) play(); });
})();

/* ====================================================================== */
/* 5. GPU telemetry                                                       */
/* ====================================================================== */
(function(){
  var G = D.gpu, NG = D.nGpus, CAP = D.gpuMemMiB;
  var svg = $('#gpu-svg'), wrap = $('#gpu-wrap'), tip = $('#gpu-tip'), slider = $('#gpu-slider'), tiles = $('#gpu-tiles');
  function gb(mib){ return mib / 1024; }
  // segments: break where consecutive rows are more than one minute apart
  var segs = [[0]], gaps = [];
  for (var i = 1; i < G.length; i++) {
    if (G[i].m - G[i - 1].m > 1) { gaps.push([G[i - 1].m, G[i].m, G[i - 1].t, G[i].t]); segs.push([]); }
    segs[segs.length - 1].push(i);
  }
  var mMax = G[G.length - 1].m;
  function mean(f){ var s = 0; G.forEach(function(r){ s += f(r); }); return s / G.length; }
  var peak = G.reduce(function(a, r){ return r.mp > a.mp ? r : a; }, G[0]);
  var avgU = mean(function(r){ return r.u; }), avgW = mean(function(r){ return r.w; });
  var kwh = G.reduce(function(a, r){ return a + r.w * NG / 60 / 1000; }, 0);
  var maxU = G.reduce(function(a, r){ return r.u > a.u ? r : a; }, G[0]);
  function tile(lab, val, sub){
    var d = h('div', 'tile', tiles);
    h('div', 'lab', d).textContent = lab;
    h('div', 'val', d).textContent = val;
    h('div', 'sub', d).textContent = sub;
  }
  tile('Peak memory on one GPU', fmt(gb(peak.mp), 1) + ' GB', fmt(peak.mp) + ' MiB at ' + peak.t + ', ' + fmt(peak.mp / CAP * 100, 0) + '% of ' + fmt(CAP) + ' MiB');
  tile('Average utilisation', fmt(avgU, 1) + '%', 'mean of ' + G.length + ' logged minutes; highest minute ' + fmt(maxU.u, 1) + '% at ' + maxU.t);
  tile('Average power', fmt(avgW, 0) + ' W per GPU', fmt(avgW * NG / 1000, 2) + ' kW for all 16 GPUs');
  tile('Estimated GPU energy', fmt(kwh, 1) + ' kWh', 'sum over ' + G.length + ' logged minutes, both jobs');

  var P = [
    {title: 'GPU utilisation (%)', ser: [{f: function(r){ return r.u; }, color: 'var(--blue)'}], lo: 0, hi: 100, n: 4},
    {title: 'GPU memory in use (GB per GPU): average and peak', ser: [{f: function(r){ return gb(r.ma); }, color: 'var(--blue)'}, {f: function(r){ return gb(r.mp); }, color: 'var(--orange)', dash: '5 3'}], lo: 0, n: 4},
    {title: 'Power: W per GPU (left), total kW for 16 GPUs (right)', ser: [{f: function(r){ return r.w; }, color: 'var(--blue)'}], n: 4, right: function(v){ return (v * NG / 1000).toFixed(1); }},
    {title: 'GPU temperature (\u00b0C)', ser: [{f: function(r){ return r.c; }, color: 'var(--blue)'}], n: 4}
  ];
  P.forEach(function(p){
    var lo = Infinity, hi = -Infinity;
    p.ser.forEach(function(s){ G.forEach(function(r){ var v = s.f(r); if (v < lo) lo = v; if (v > hi) hi = v; }); });
    if (p.lo != null) lo = p.lo;
    if (p.hi != null) hi = p.hi;
    p.ticks = ticks(lo, hi, p.n);
    p.a = p.ticks[0]; p.b = p.ticks[p.ticks.length - 1];
  });
  var hover = null, geo = null;
  slider.max = String(G.length - 1);

  function draw(){
    var W = Math.max(280, wrap.clientWidth), ML = 46, MR = 44, PHt = W < 520 ? 110 : 130, GAP = 46, TOP = 24;
    var H = TOP + P.length * (PHt + GAP) + 10, pw = W - ML - MR;
    geo = {W: W, ML: ML, pw: pw};
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    clear(svg);
    function X(m){ return ML + m / mMax * pw; }
    var stepMin = W < 520 ? 30 : 15, xt = [];
    G.forEach(function(r){ var mm = +r.t.slice(3); if (mm % stepMin === 0) xt.push(r); });
    // ticks that fall inside a gap have no row; add them from the clock
    var startClock = (+G[0].t.slice(0, 2)) * 60 + (+G[0].t.slice(3));
    var tickList = [];
    for (var m = 0; m <= mMax; m++) {
      var clock = startClock + m;
      if (clock % stepMin === 0) tickList.push({m: m, t: ('0' + Math.floor(clock / 60)).slice(-2) + ':' + ('0' + clock % 60).slice(-2)});
    }
    P.forEach(function(p, i){
      var y0 = TOP + i * (PHt + GAP), y1 = y0 + PHt;
      function Y(v){ return y1 - (v - p.a) / (p.b - p.a) * PHt; }
      el('text', {x: ML, y: y0 - 8, 'class': 'ptitle'}, svg, W < 420 && p.title.length > 40 ? p.title.replace(': average and peak', '').replace(', total kW for 16 GPUs (right)', ', total kW (right)') : p.title);
      gaps.forEach(function(g){
        el('rect', {x: X(g[0]), y: y0, width: X(g[1]) - X(g[0]), height: PHt, fill: 'var(--surface2)'}, svg);
        if (i === 0 && X(g[1]) - X(g[0]) > 44) el('text', {x: (X(g[0]) + X(g[1])) / 2, y: y0 + 14, 'text-anchor': 'middle'}, svg, 'no job');
      });
      p.ticks.forEach(function(v){
        el('line', {x1: ML, x2: ML + pw, y1: Y(v), y2: Y(v), 'class': 'gridl'}, svg);
        el('text', {x: ML - 6, y: Y(v) + 4, 'text-anchor': 'end'}, svg, fmt(v));
        if (p.right) el('text', {x: ML + pw + 6, y: Y(v) + 4}, svg, p.right(v));
      });
      el('line', {x1: ML, x2: ML + pw, y1: y1, y2: y1, 'class': 'axisl'}, svg);
      tickList.forEach(function(tk){
        el('line', {x1: X(tk.m), x2: X(tk.m), y1: y1, y2: y1 + 4, 'class': 'axisl'}, svg);
        el('text', {x: X(tk.m), y: y1 + 16, 'text-anchor': 'middle'}, svg, tk.t);
      });
      p.ser.forEach(function(s){
        segs.forEach(function(seg){
          var d = '';
          seg.forEach(function(idx, j){ d += (j ? 'L' : 'M') + X(G[idx].m).toFixed(1) + ' ' + Y(s.f(G[idx])).toFixed(1); });
          var a = {d: d, fill: 'none', stroke: s.color, 'stroke-width': 1.8, 'stroke-linejoin': 'round'};
          if (s.dash) a['stroke-dasharray'] = s.dash;
          el('path', a, svg);
        });
      });
      if (hover != null) {
        var r = G[hover];
        el('line', {x1: X(r.m), x2: X(r.m), y1: y0, y2: y1, stroke: 'var(--text)', 'stroke-width': 1, opacity: 0.55}, svg);
        p.ser.forEach(function(s){
          el('circle', {cx: X(r.m), cy: Y(s.f(r)), r: 3.5, fill: 'var(--surface)', stroke: s.color, 'stroke-width': 2}, svg);
        });
      }
    });
    var yb = TOP + P.length * (PHt + GAP) - GAP + 32;
    el('text', {x: ML, y: yb}, svg, 'Time of day (PST), 2026-11-24');
  }
  function tipHtml(r){
    return '<b>' + r.t + ' PST</b>' +
      'Utilisation: ' + fmt(r.u, 1) + '%<br>' +
      'Memory, average: ' + fmt(gb(r.ma), 1) + ' GB<br>' +
      'Memory, peak: ' + fmt(gb(r.mp), 1) + ' GB<br>' +
      'Power: ' + fmt(r.w, 0) + ' W per GPU, ' + fmt(r.w * NG / 1000, 2) + ' kW total<br>' +
      'Temperature: ' + fmt(r.c, 1) + ' \u00b0C<br>' +
      '<span class="muted">' + r.n + ' samples in this minute</span>';
  }
  function show(idx, px, py){
    hover = idx; slider.value = String(idx); draw();
    tip.innerHTML = tipHtml(G[idx]);
    slider.setAttribute('aria-valuetext', tip.textContent);
    placeTip(tip, wrap, px, py);
  }
  svg.addEventListener('pointermove', function(e){
    var rc = svg.getBoundingClientRect(), k = geo.W / rc.width;
    var m = ((e.clientX - rc.left) * k - geo.ML) / geo.pw * mMax, best = 0, bd = Infinity;
    for (var j = 0; j < G.length; j++) { var dd = Math.abs(G[j].m - m); if (dd < bd) { bd = dd; best = j; } }
    if (bd > 1.01) {
      hover = null; draw();
      var gp = gaps[0];
      tip.innerHTML = '<b>No data</b>No job was running' + (gp ? ' between ' + gp[2] + ' and ' + gp[3] : '') + '.';
      placeTip(tip, wrap, e.clientX - rc.left, e.clientY - rc.top);
      return;
    }
    show(best, e.clientX - rc.left, e.clientY - rc.top);
  });
  svg.addEventListener('pointerleave', function(){ hover = null; tip.style.display = 'none'; draw(); });
  function fromSlider(){
    var idx = +slider.value, rc = svg.getBoundingClientRect();
    show(idx, (geo.ML + G[idx].m / mMax * geo.pw) * (rc.width / geo.W), 40);
  }
  slider.addEventListener('input', fromSlider);
  slider.addEventListener('blur', function(){ hover = null; tip.style.display = 'none'; draw(); });
  onResize(wrap, draw);
  draw();
})();

/* ====================================================================== */
/* 6. Before / after                                                      */
/* ====================================================================== */
(function(){
  var B = D.eval.base, T = D.eval.tuned, box = $('#ev-bars'), tbody = $('#ev-table');
  function pct(v){ return v.toFixed(2) + '%'; }
  var rows = [
    {k: 'acc', lab: 'Overall accuracy', max: 100, f: pct, scale: 'scale 0 to 100%', up: true, pts: true},
    {k: 'run_rate', lab: 'Accuracy: run_rate questions', max: 100, f: pct, scale: 'scale 0 to 100%', up: true, pts: true},
    {k: 'chase_rate', lab: 'Accuracy: chase_rate questions', max: 100, f: pct, scale: 'scale 0 to 100%', up: true, pts: true},
    {k: 'milestone', lab: 'Accuracy: milestone questions', max: 100, f: pct, scale: 'scale 0 to 100%', up: true, pts: true},
    {k: 'tokens', lab: 'Mean output tokens per answer', max: 1600, f: function(v){ return fmt(v); }, scale: 'scale 0 to 1,600 tokens', up: false},
    {k: 'cutoff', lab: 'Answers cut off at the token limit', max: 10, f: pct, scale: 'scale 0 to 10%', up: false, pts: true}
  ];
  var bars = [];
  rows.forEach(function(r){
    var g = h('div', 'bg', box);
    h('h3', null, g, r.lab + ' <span class="scale">(' + r.scale + ')</span>');
    [['Base', B[r.k], ''], ['Tuned', T[r.k], ' t']].forEach(function(x){
      var row = h('div', 'brow', g);
      h('span', null, row).textContent = x[0];
      var tr = h('div', 'btrack', row);
      var b = h('div', 'bar' + x[2], tr);
      b.setAttribute('data-w', String(clamp(x[1] / r.max * 100, 0, 100)));
      bars.push(b);
      h('span', 'bval', row).textContent = r.f(x[1]);
    });
    var diff = T[r.k] - B[r.k], better = r.up ? diff > 0 : diff < 0, small = r.pts && Math.abs(diff) <= 1;
    var txt;
    if (r.pts) txt = (diff >= 0 ? '+' : '\u2212') + Math.abs(diff).toFixed(2) + ' points' + (small ? ' (within noise for a single run)' : '');
    else txt = (B[r.k] / T[r.k]).toFixed(1) + '\u00d7 fewer tokens';
    h('span', 'delta' + (better && !small ? ' good' : ''), g).textContent = txt;
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', r.lab + ': base ' + r.f(B[r.k]) + ', base plus adapter ' + r.f(T[r.k]));
  });
  [['Accuracy', 'acc', pct], ['Mean output tokens', 'tokens', function(v){ return fmt(v); }], ['Cut off at token limit', 'cutoff', pct],
   ['No answer', 'noanswer', pct], ['run_rate accuracy', 'run_rate', pct], ['chase_rate accuracy', 'chase_rate', pct], ['milestone accuracy', 'milestone', pct]
  ].forEach(function(x){
    var tr = h('tr', null, tbody);
    h('td', null, tr).textContent = x[0];
    h('td', null, tr).textContent = x[2](B[x[1]]);
    h('td', null, tr).textContent = x[2](T[x[1]]);
  });
  function grow(){ bars.forEach(function(b){ b.style.width = b.getAttribute('data-w') + '%'; }); }
  function replay(){
    bars.forEach(function(b){ b.style.transition = 'none'; b.style.width = '0'; });
    void box.offsetWidth;
    bars.forEach(function(b){ b.style.transition = ''; });
    requestAnimationFrame(function(){ requestAnimationFrame(grow); });
  }
  $('#ev-replay').addEventListener('click', replay);
  if (RM) grow(); else onVisible(box, function(){ requestAnimationFrame(grow); });
})();

})();
</script>
</body>
</html>
'''

if __name__ == "__main__":
    main()
