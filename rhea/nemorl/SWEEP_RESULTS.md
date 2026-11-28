# Seed and length-coefficient sweep (2026-11-26 to 2026-11-28)

Nemotron 3.5 Lightning 30B, NeMo RL GRPO + LoRA r32, 4 nodes (16 Vera Rubin GPUs), 60 steps of 64 questions x 8
answers. Reward of a finished, correct answer = `1 - coef * output_tokens / 4096`, otherwise 0.
Evaluation: NeMo Gym, 600 held-out questions x 4 sampled answers, vLLM serving the adapter as a LoRA module.
Submitted with `nemorl/sweep.sh "0 0.1 0.4 1.0" "1 2 3" 60`; outputs on Lustre in `nemo-rl/runs/sweep-c<coef>-s<seed>`
(training) and `nemotron-rhea/runs/sweep-c<coef>-s<seed>/gym` (evaluation).

Base model, same harness: 86.50% and 86.25% accuracy in two evaluations, 1,565 and 1,605 output tokens, 9.42% and
10.12% of answers cut off at 4,096 tokens.

| Coefficient | Seed | Accuracy | Avg output tokens | Cut off | run_rate | chase_rate | milestone |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 99.79% | 845 | 0.04% | 99.75% | 99.75% | 99.88% |
| 0 | 2 | 99.88% | 1,034 | 0.00% | 99.88% | 100.00% | 99.75% |
| 0 | 3 | 99.88% | 896 | 0.00% | 99.88% | 99.75% | 100.00% |
| 0.1 | 1 | 99.88% | 351 | 0.00% | 99.62% | 100.00% | 100.00% |
| 0.1 | 2 | 98.54% | 266 | 0.00% | 99.50% | 97.12% | 99.00% |
| 0.1 | 3 | 99.79% | 155 | 0.00% | 99.88% | 99.62% | 99.88% |
| 0.4 | 1 | 99.62% | 155 | 0.00% | 99.75% | 99.12% | 100.00% |
| 0.4 | 2 | 95.12% | 126 | 0.00% | 99.62% | 99.50% | 86.25% |
| 0.4 | 3 | 96.83% | 90 | 0.04% | 97.25% | 93.50% | 99.75% |
| 1.0 | 1 | 97.50% | 78 | 0.00% | 97.88% | 96.50% | 98.12% |
| 1.0 | 2 | 99.67% | 77 | 0.00% | 99.75% | 99.75% | 99.50% |
| 1.0 | 3 | 98.33% | 67 | 0.00% | 99.00% | 97.00% | 99.00% |

Means over the three seeds:

| Coefficient | Accuracy (min to max) | Avg output tokens (min to max) |
|---|---|---|
| 0 | 99.85% (99.79 to 99.88) | 925 (845 to 1,034) |
| 0.1 | 99.40% (98.54 to 99.88) | 257 (155 to 351) |
| 0.4 | 97.19% (95.12 to 99.62) | 124 (90 to 155) |
| 1.0 | 98.50% (97.50 to 99.67) | 74 (67 to 78) |

The earlier 100-step run with coefficient 0.4 (seed 42) scored 97.71% and 97.58% with 107 and 106 tokens.

Same-harness results for the two Qwen RLVR adapters and the throughput and energy measurements are in the job
outputs `nemotron-rhea/runs/qwen38-rlvr-1114/gym`, `qwen36-rlvr-1113/gym` and `nemorl-cricket-1124/gym-energy`
(`*_summary.json`, `energy.json`).
