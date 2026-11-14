#!/usr/bin/env python
"""Derive rlvr36_template.html from rlvr_template.html: same page, but GPU data covers every GPU (nvidia-smi on all nodes)
instead of the 4 GPUs W&B sampled, axes come from the data, and a comparison with the 27B run is added."""
from pathlib import Path

here = Path(__file__).resolve().parent
s = (here / "rlvr_template.html").read_text(encoding="utf-8")


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:70])
    s = s.replace(a, b)


# ---- static HTML ----
rep("<title>Cricket RLVR Run Report</title>", "<title>__TITLE__</title>")
rep("<h1>RLVR on 32 Vera Rubin GPUs: what the run did and how well the GPUs were used</h1>",
    "<h1>__H1__</h1>")
rep('''<p class="sub">Qwen3.8-27B was trained with reinforcement learning with verifiable rewards on T20 cricket questions (Rhea job 39000, 13 Nov 2026). The model answered questions itself and was rewarded only for a correct final number, with a small bonus for shorter answers.</p>''',
    '''<p class="sub">__SUB__</p>''')
rep('max="100" step="1" value="100"', 'max="__N__" step="1" value="__N__"')
rep('''<p class="sub">Accuracy and length are measured on the training rollouts, which are capped at 1,024 tokens. They are not benchmark numbers.</p>''',
    '''<p class="sub">Accuracy and length are measured on the training rollouts, on training questions with plain temperature-1 sampling. They are not benchmark numbers.</p>''')
rep('''<select id="gsel"><option value="-1" selected>Average of the 4 GPUs</option><option value="0">GPU 0</option><option value="1">GPU 1</option><option value="2">GPU 2</option><option value="3">GPU 3</option></select>''',
    '''<select id="gsel"></select>''')
rep('''<h3>GPU utilisation over the run</h3><p class="note">Percent of time the GPU was busy, one reading every 15 s</p>''',
    '''<h3>GPU utilisation over the run</h3><p class="note">Percent of time the GPU was busy, averaged per minute</p>''')
rep('''<p class="note">Watts per GPU</p>''', '''<p class="note">Watts per GPU, averaged per minute</p>''')
rep('''<h3>Peak memory per GPU</h3>''', '''<h3>Peak memory per node</h3>''')
rep('''<h2>What the model learned, step by step</h2>''',
    '''<div class="card" style="margin-top:12px"><h3>Utilisation of every GPU, minute by minute</h3><p class="note">One row per GPU, one column per minute. Darker means busier. Measured on all 32 GPUs.</p><svg id="uHeat"></svg><div class="legend" id="uHeatLegend"></div></div>

<h2>What the model learned, step by step</h2>''')
rep('''<details><summary>Per-step table</summary>''',
    '''<h2>__CMP_H__</h2>
<p class="sub">__CMP_SUB__</p>
<div class="card"><div class="scroll"><table id="cmp"></table></div></div>

<details><summary>Per-step table</summary>''')

# ---- JS ----
rep('''tile("Measured GPU utilisation", `${fmt(utilAvg)}%`, `average of the 4 sampled GPUs on ${G.sampled_node}`);''',
    '''tile("Measured GPU utilisation", `${fmt(utilAvg)}%`, `average of all ${J.gpus} GPUs over the training loop`);''')
rep('''const utilAvg = G.util_avg.reduce((a, b) => a + b, 0) / G.util_avg.length;''', '''const utilAvg = G.util_all;''')
rep('''$("#gridnote").textContent = `Each GPU tile is split by how that step's time was divided between rollouts and the update. The percentage on ${G.sampled_node} is measured utilisation during that step; W&B sampled only that node, so the other 28 GPUs show no number. All GPUs ran the same workload in lockstep.`;''',
    '''$("#gridnote").textContent = `Each GPU tile is split by how that step's time was divided between rollouts and the update. The percentage is that GPU's measured utilisation during the step, from nvidia-smi readings every ${G.interval_s} s on every node.`;''')
rep('''sampled: n === G.sampled_node, g };''', '''sampled: true, g };''')
rep('''(cell.sampled && s.util[g] != null ? `<br>Measured utilisation ${fmt(s.util[g])}%` : `<br>Utilisation not sampled on this node`)''',
    '''(s.util[cell.idx] != null ? `<br>Measured utilisation ${fmt(s.util[cell.idx])}%` : `<br>No reading in this step`)''')
rep('''x.t.textContent = x.sampled && s.util[x.g] != null ? fmt(s.util[x.g]) + "%" : "";''',
    '''x.t.textContent = s.util[x.idx] != null ? fmt(s.util[x.idx]) + "%" : "";''')
rep('''[1, 20, 40, 60, 80, 100].forEach(s => svg.appendChild''', '''D.axes.steps.forEach(s => svg.appendChild''')
rep('''(nodeUtil(s) != null ? `<br>GPU utilisation ${fmt(nodeUtil(s))}% (4 sampled)` : "")''',
    '''(nodeUtil(s) != null ? `<br>GPU utilisation ${fmt(nodeUtil(s))}% (all ${J.gpus})` : "")''')
rep('''stepChart("#cTok", 210, 0, 700, [0, 175, 350, 525, 700],''', '''stepChart("#cTok", 210, 0, D.axes.tok[D.axes.tok.length - 1], D.axes.tok,''')
rep('''stepChart("#cMix", 210, 0, 70, [0, 20, 40, 60],''', '''stepChart("#cMix", 210, 0, D.axes.mix[D.axes.mix.length - 1], D.axes.mix,''')
rep('''const TMAX = 180;''', '''const TMAX = D.axes.tmax;''')
rep('''stepChart("#cTime", 250, 0, TMAX, [0, 60, 120, 180],''', '''stepChart("#cTime", 250, 0, TMAX, D.axes.tticks,''')
rep('''$("#utilNote").textContent = `Mean of the 4 GPUs on ${G.sampled_node}, one reading every ${G.interval_s} s, averaged per step`;''',
    '''$("#utilNote").textContent = `Mean of all ${J.gpus} GPUs, one reading every ${G.interval_s} s, averaged per step`;''')
rep('''$("#mem").innerHTML = G.mem_peak_gb.map((v, i) => `<span>GPU ${i}</span>''', '''$("#mem").innerHTML = G.mem_peak_gb.map((v, i) => `<span>${G.units[i]}</span>''')
rep('''hover(t, `<b>${G.sampled_node}, GPU ${i}</b><br>Peak ${fmt(G.mem_peak_gb[i], 1)} GB (${pct(G.mem_peak_gb[i] / J.gpu_gb, 0)})<br>Average ${fmt(G.mem_avg_gb[i], 1)} GB`);''',
    '''hover(t, `<b>${G.units[i]}</b><br>Fullest GPU peaked at ${fmt(G.mem_peak_gb[i], 1)} GB (${pct(G.mem_peak_gb[i] / J.gpu_gb, 0)})<br>Node average ${fmt(G.mem_avg_gb[i], 1)} GB`);''')
rep('''arr.map((a, i) => `${i === g ? "<b>" : ""}GPU ${i}: ${a[k] == null''', '''arr.map((a, i) => `${i === g ? "<b>" : ""}${G.units[i]}: ${a[k] == null''')
rep('''$("#gsel").addEventListener("change", drawUsage); drawUsage();''',
    '''$("#gsel").innerHTML = `<option value="-1" selected>Average of all ${J.gpus} GPUs</option>` + G.units.map((u, i) => `<option value="${i}">${u} (4 GPUs)</option>`).join("");
$("#gsel").addEventListener("change", drawUsage); drawUsage();''')
rep('''$("#usageSub").textContent = `Measured by W&B on ${G.sampled_node}, the node where process 0 ran: ${G.samples} readings, one every ${G.interval_s} s, for 4 of the ${J.gpus} GPUs. The other 28 GPUs ran the same workload but were not sampled.`;''',
    '''$("#usageSub").textContent = `Measured with nvidia-smi on all ${J.nodes.length} nodes: one reading every ${G.interval_s} s for each of the ${J.gpus} GPUs (${fmt(G.samples)} readings in total). The charts show each node as the average of its 4 GPUs; the heatmap shows every GPU.`;''')
rep('''`${fmt(Math.min(...G.util_avg))}% to ${fmt(Math.max(...G.util_avg))}% across the 4 GPUs`''', '''`${fmt(G.util_gpu_min)}% to ${fmt(G.util_gpu_max)}% across the ${J.gpus} GPUs`''')
rep('''`${pct(memAvgAll / J.gpu_gb, 0)} of ${J.gpu_gb} GB, mean of the 4 GPUs`''', '''`${pct(memAvgAll / J.gpu_gb, 0)} of ${J.gpu_gb} GB, mean of all ${J.gpus} GPUs`''')
rep('''$("#uMemNote").textContent = `Gigabytes in use out of ${J.gpu_gb} GB. The lowest reading, with only the model loaded, was ${fmt(Math.min(...G.mem_gb.flat().filter(x => x != null)))} GB.`;''',
    '''$("#uMemNote").textContent = `Gigabytes in use out of ${J.gpu_gb} GB, averaged per minute`;''')
rep('''"<tr><th>GPU</th><th>Avg utilisation</th><th>Avg memory</th><th>Peak memory</th><th>Avg power</th><th>Peak temp</th></tr>" +
  G.util_avg.map((u, i) => `<tr><td>GPU ${i}</td>''', '''"<tr><th>Node</th><th>Avg utilisation</th><th>Avg memory</th><th>Peak memory</th><th>Avg power</th><th>Peak temp</th></tr>" +
  G.util_avg.map((u, i) => `<tr><td>${G.units[i]}</td>''')
rep('''$("#memNote").textContent = `Highest reading for each sampled GPU, out of ${J.gpu_gb} GB`;''', '''$("#memNote").textContent = `Highest reading on any of the node's 4 GPUs, out of ${J.gpu_gb} GB`;''')
rep('''dumbbell("#bAcc", "accuracy", 0.92, 1, [0.92, 0.94, 0.96, 0.98, 1],''', '''dumbbell("#bAcc", "accuracy", D.axes.bacc[0], 1, D.axes.bacc,''')
rep('''dumbbell("#bTok", "mean_tokens", 0, 1000, [0, 250, 500, 750, 1000],''', '''dumbbell("#bTok", "mean_tokens", 0, D.axes.btok[D.axes.btok.length - 1], D.axes.btok,''')

# verdict: replace the two lists and the caveat wholesale
i0, i1 = s.index('''$("#good").innerHTML = li(['''), s.index('''/* ---- table ---- */''')
verdict = r'''const lastRo = sum(S.slice(-10), s => s.ro) / 10, lastUp = sum(S.slice(-10), s => s.up) / 10;
const tokFloor = S.find(s => s.tok <= 1.25 * Math.min(...S.map(x => x.tok)));
const afterFloor = tLoop - tokFloor.el, minAccAfter = Math.min(...S.filter(s => s.s > tokFloor.s).map(s => s.acc));
const gpuHFloor = J.gpus * (tokFloor.el + tOther) / 3600, tokCut = 1 - F.overall.mean_tokens / B.overall.mean_tokens;
const QT = TYPES.filter(([t]) => t !== "overall"), best = QT.slice().sort((x, y) => (F[y[0]].accuracy - B[y[0]].accuracy) - (F[x[0]].accuracy - B[x[0]].accuracy))[0];
const memTight = Math.max(...G.mem_peak_gb) / J.gpu_gb > 0.95;
$("#good").innerHTML = li([
  `<b>All ${J.gpus} GPUs worked on every step.</b> Rollouts and gradients both ran on every GPU in parallel, so each update saw ${fmt(perStep)} fresh answers.`,
  `<b>Load was even across GPUs.</b> Average utilisation per GPU ranged from ${fmt(G.util_gpu_min)}% to ${fmt(G.util_gpu_max)}%, measured on all ${J.gpus}.`,
  `<b>Every question type ended at ${fmt(Math.floor(Math.min(...QT.map(([t]) => F[t].accuracy)) * 1000) / 10, 1)}% or higher.</b> The biggest gain was on ${best[1].toLowerCase()} questions: ${fmt(B[best[0]].accuracy * 100, 2)}% to ${fmt(F[best[0]].accuracy * 100, 2)}%.`,
  ...(memTight ? [] : [`<b>The large memory was put to work.</b> Peak memory per node ranged from ${fmt(Math.min(...G.mem_peak_gb))} to ${fmt(Math.max(...G.mem_peak_gb))} GB of ${fmt(J.gpu_gb, 1)} GB on the fullest GPU.`]),
  `<b>Benchmark result.</b> Accuracy went from ${fmt(B.overall.accuracy * 100, 2)}% to ${fmt(F.overall.accuracy * 100, 2)}% (${sign(dAcc)} points), and answers became ${fmt(tokCut * 100)}% shorter (${fmt(B.overall.mean_tokens)} to ${fmt(F.overall.mean_tokens)} tokens).`]);
const slipped = TYPES.filter(([t]) => t !== "overall" && F[t].accuracy < B[t].accuracy);
$("#bad").innerHTML = li([
  ...slipped.map(([t, name]) => `<b>${name} questions slipped.</b> Accuracy on them went from ${fmt(B[t].accuracy * 100, 2)}% to ${fmt(F[t].accuracy * 100, 2)}%, and ${fmt((1 - F[t]["pass@4"]) * F[t].questions)} of ${F[t].questions} are wrong in all 4 samples (${fmt((1 - B[t]["pass@4"]) * B[t].questions)} before).`),
  ...(!memTight ? [] : [`<b>GPU memory was nearly exhausted.</b> Every node had a GPU peak between ${fmt(Math.min(...G.mem_peak_gb))} and ${fmt(Math.max(...G.mem_peak_gb))} GB of ${fmt(J.gpu_gb, 1)} GB, during the long-answer steps at the start. The run did not fail, but it had almost no headroom.`]),
  `<b>The first step was very slow.</b> Step 1 took ${fmt(S[0].st / 60)} minutes, ${fmt(S[0].ro / 60)} of them in rollouts: answers averaged ${fmt(S[0].tok)} tokens and the model's kernels were compiled for the first time.`,
  `<b>A large part of the run came after answers stopped getting shorter.</b> Rollout accuracy passed 98.5% at step ${conv.s} and answer length reached its floor around step ${tokFloor.s}. The ${N - tokFloor.s} steps after that took ${fmt(afterFloor / 60)} minutes (${pct(afterFloor / J.elapsed_s, 0)} of the job) with rollout accuracy at ${pct(minAccAfter)} or higher.`,
  `<b>Updates dominated step time.</b> Updates took ${pct(tUpd / (tRoll + tUpd), 0)} of step time. In the last ten steps a rollout phase took about ${fmt(lastRo)} s and the update about ${fmt(lastUp)} s.${A.train_batch < 8 ? ` The update batch was cut to ${A.train_batch} sequences to stay inside GPU memory, which makes it slow.` : ""}`,
  `<b>The GPUs were not saturated.</b> Utilisation averaged ${fmt(utilAvg)}% across all ${J.gpus} GPUs (about ${fmt(G.util_rollout)}% in rollouts and ${fmt(G.util_update)}% in updates), and power averaged ${fmt(G.power_avg_w)} W against a ${fmt(G.power_peak_w)} W peak.`,
  `<b>A shorter run would likely have cost less.</b> Stopping at step ${tokFloor.s} would have used about ${fmt(gpuHFloor)} GPU-hours instead of ${fmt(gpuH)}.`]);
$("#caveat").textContent = `Limits of this data: GPU readings are ${G.interval_s} s apart, so the rollout/update split of utilisation is approximate. Answers per minute come from step timings, not per-GPU logs. Only the final adapter was benchmarked, so the benchmark score at earlier steps is not known.`;

/* ---- utilisation heatmap, every GPU ---- */
(() => {
  const svg = $("#uHeat"), H32 = G.heat, NBH = H32[0].length, WW = 1040, l = 104, r = 12, t = 4, rh = 10, b = 22, iw = WW - l - r, cw = iw / NBH, H = t + J.gpus * rh + 7 * 4 + b;
  svg.setAttribute("viewBox", `0 0 ${WW} ${H}`); svg.setAttribute("role", "img");
  const rowY = g => t + g * rh + Math.floor(g / 4) * 4;
  J.nodes.forEach((n, i) => svg.appendChild(el("text", { x: l - 8, y: rowY(i * 4) + 2 * rh + 4, "text-anchor": "end" }, `${n} (${i * 4}-${i * 4 + 3})`)));
  H32.forEach((row, g) => row.forEach((v, m) => { if (v != null) svg.appendChild(el("rect", { x: l + m * cw, y: rowY(g), width: Math.max(0.5, cw - 0.6), height: rh - 1, fill: "var(--s1)", opacity: (0.06 + 0.94 * v / 100).toFixed(3) })); }));
  for (let m = 0; m <= NBH; m += 20) svg.appendChild(el("text", { x: l + m * cw, y: H - 6, "text-anchor": "middle" }, m === 0 ? "0 min" : m));
  const hh = rowY(J.gpus - 1) + rh - t, ov = el("rect", { x: l, y: t, width: iw, height: hh, fill: "transparent" }); svg.appendChild(ov);
  ov.addEventListener("mousemove", e => {
    const bb = ov.getBoundingClientRect(), m = Math.max(0, Math.min(NBH - 1, Math.floor((e.clientX - bb.left) / bb.width * NBH))), yy = (e.clientY - bb.top) / bb.height * hh;
    let g = 0; for (let k = 0; k < J.gpus; k++) if (yy >= rowY(k) - t) g = k;
    showTip(e, `<b>${J.nodes[Math.floor(g / 4)]}, GPU ${g % 4}</b> (process ${g})<br>Minute ${m + 1}: ${H32[g][m] == null ? "no reading" : fmt(H32[g][m]) + "% utilisation"}`);
  });
  ov.addEventListener("mouseleave", hideTip);
  $("#uHeatLegend").innerHTML = [0, 25, 50, 75, 100].map(f => `<span><i style="background:var(--s1);opacity:${0.06 + 0.94 * f / 100}"></i>${f}%</span>`).join("");
})();

/* ---- comparison with the 27B run ---- */
(() => {
  const P = D.prev, row = (k, a, b) => `<tr><td>${k}</td><td>${a}</td><td>${b}</td></tr>`;
  $("#cmp").innerHTML = `<tr><th>Measure</th><th>${P.name}</th><th>${D.model}</th></tr>` +
    row("Benchmark accuracy before", fmt(P.base.accuracy * 100, 2) + "%", fmt(B.overall.accuracy * 100, 2) + "%") +
    row("Benchmark accuracy after", fmt(P.ft.accuracy * 100, 2) + "%", fmt(F.overall.accuracy * 100, 2) + "%") +
    row("Change in accuracy", sign((P.ft.accuracy - P.base.accuracy) * 100) + " points", sign(dAcc) + " points") +
    row("Tokens per answer before", fmt(P.base.mean_tokens), fmt(B.overall.mean_tokens)) +
    row("Tokens per answer after", fmt(P.ft.mean_tokens), fmt(F.overall.mean_tokens)) +
    row("Cut-off answers before", pct(P.base.truncated_rate, 2), pct(B.overall.truncated_rate, 2)) +
    row("Cut-off answers after", pct(P.ft.truncated_rate, 2), pct(F.overall.truncated_rate, 2)) +
    row("Training steps", P.steps, N) +
    row("Rollouts", fmt(P.rollouts), fmt(N * perStep)) +
    row("Rollout cap (tokens)", fmt(P.max_new_tokens), fmt(A.max_new_tokens)) +
    row("Trainable LoRA parameters", P.trainable, D.trainable) +
    row("GPUs with usage readings", P.measured, `all ${J.gpus}`) + row("Average GPU utilisation", P.util + "%", fmt(utilAvg) + "%") + row("Peak GPU memory", fmt(P.mem_peak) + " GB", fmt(Math.max(...G.mem_peak_gb)) + " GB") +
    row("Training job time", fmt(P.elapsed_s / 60) + " min", fmt(J.elapsed_s / 60) + " min") +
    row("GPU-hours", fmt(P.elapsed_s * 32 / 3600, 1), fmt(gpuH, 1));
})();

'''
s = s[:i0] + verdict + s[i1:]
rep('''const tile = (k, v, d) =>''', '''const sign = (x, d = 2) => (x >= 0 ? "+" : "") + fmt(x, d);
const tile = (k, v, d) =>''')
rep('''tile("Benchmark accuracy", `${fmt(B.overall.accuracy * 100, 2)}% <small>to</small> ${fmt(F.overall.accuracy * 100, 2)}%`, `+${fmt(dAcc, 2)} points on 600 held-out questions`)''',
    '''tile("Benchmark accuracy", `${fmt(B.overall.accuracy * 100, 2)}% <small>to</small> ${fmt(F.overall.accuracy * 100, 2)}%`, `${sign(dAcc)} points on 600 held-out questions`)''')
rep(" (process ${cell.idx})", "")
rep(" (process ${g})", "", 2)
(here / "rlvr_full_template.html").write_text(s, encoding="utf-8", newline="\n")
print("wrote rlvr_full_template.html", len(s))
