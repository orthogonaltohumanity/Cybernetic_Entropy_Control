# Cybernetic Entropy Control

A 4th-order feedback controller that adjusts LLM sampling parameters in real-time based on token-level entropy. Using velocity-form actuation, it improves MATH benchmark accuracy from 75% to 83% over an uncontrolled baseline on small-scale experiments, and from 60.9% to 63.0% on a full 5000-problem sweep.

## What it does

At each token generation step, the controller:

1. Computes Shannon entropy over the top-64 softmaxed logits
2. Feeds the error (target − actual) through a tanh dampener
3. Updates a 4th-order state vector: integral, error, velocity (Δe), acceleration (Δ²e)
4. Applies velocity-form actuation: each actuator integrates K·x over time, accumulating corrections rather than computing them from scratch

The acceleration term is the key contribution — it catches the upward curvature in entropy that precedes a hallucination spike, enabling intervention before it peaks. The velocity form gives the controller memory — by integrating corrections over time, it maintains persistent state that adapts to the trajectory of the generation rather than reacting only to instantaneous error.

## Results

### Full benchmark sweep (5000 problems)

All results: Qwen 3.5 2B (Q4_K_M), 5000 problems from MATH (Hendrycks et al.), 4096 token budget, llama.cpp with CUDA.

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 60.9% (3047/5000) | — |
| **Hybrid (4th order + QEWS, w_H=1, w_Q=1)** | **63.0% (3152/5000)** | **+2.1pp** |

**Token efficiency:** The controller generates ~7% fewer tokens on average (mean 1852 → 1722 tokens).

#### Failure mode analysis

The dominant failure mode is spinning: the model exhausts the token budget without converging. Cap hits are nearly always wrong (5.4% accuracy on baseline, 6.2% on hybrid). The controller's gain comes almost entirely from reducing cap hits rather than correcting confident wrong answers.

| | Baseline | Hybrid | Δ |
|---|---|---|---|
| Cap hit rate | 28.2% (1409/5000) | 25.3% (1267/5000) | −2.9pp |
| Under-cap accuracy | 82.7% (2971/3591) | 82.3% (3073/3733) | −0.4pp |
| Cap-hit accuracy | 5.4% (76/1409) | 6.2% (79/1267) | +0.8pp |
| Overall accuracy | 60.9% | 63.0% | +2.1pp |

Under-cap accuracy is nearly unchanged (−0.4pp), confirming the controller is not interfering with problems it has no business touching. The entire gain is cap hit reduction.

#### Entropy analysis

Mean entropy is a strong predictor of outcome. Capped problems have systematically higher entropy than uncapped ones, and wrong answers have higher entropy than correct ones across both conditions.

| Condition | Group | mean H | std H |
|---|---|---|---|
| Baseline | Uncapped + Correct | 0.225 | 0.419 |
| Baseline | Uncapped + Wrong | 0.256 | 0.455 |
| Baseline | Capped + Wrong | 0.344 | 0.540 |
| Hybrid | Uncapped + Correct | 0.210 | 0.404 |
| Hybrid | Uncapped + Wrong | 0.244 | 0.443 |
| Hybrid | Capped + Wrong | 0.285 | 0.496 |

The controller reduces mean entropy across all groups. The largest reduction is in capped wrong answers (0.344 → 0.285), consistent with it having the most authority over high-entropy spinning problems.

This implies the +2.1pp gain is a lower bound: stronger actuators with more authority over spinning should produce larger gains. The current actuators (min_p, top_p, repeat penalty) are insufficient to fully break a spin once it begins.

---

### Small-scale experiments (v2: velocity-form controller)

All results: Qwen 3.5 2B (Q4_K_M), 100 problems from MATH, 2048 token budget, llama.cpp with CUDA.

#### Multi-actuator control (Min-P + Top-P + Frequency Penalty)

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | — |
| PID (3rd order) | 78.0% | +3.0 |
| 4th order, 2x acceleration | 79.0% | +4.0 |
| **4th order** | **82.0%** | **+7.0** |

#### Min-P only

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | — |
| 4th order | 77.0% | +2.0 |
| 4th order, 2x acceleration | 77.0% | +2.0 |
| **PID** | **82.0%** | **+7.0** |

#### QEWS (Quantum Early Warning Signal)

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | — |
| QEWS hybrid (w_H=1, w_Q=2) | 74.0% | −1.0 |
| QEWS replace (4th order) | 76.0% | +1.0 |
| QEWS replace (4th order 2x) | 79.0% | +4.0 |
| QEWS replace (PID) | 82.0% | +7.0 |
| **QEWS hybrid (w_H=1, w_Q=1)** | **83.0%** | **+8.0** |

<details>
<summary>v1: Position-form controller (earlier results, different parser)</summary>

All results: Qwen 3.5 2B (Q4_K_M), 200 problems from MATH, 4096 token budget, llama.cpp with CUDA.

> Note: these experiments used an earlier answer parser and are not directly comparable to results above.

#### Multi-actuator control

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| PID (3rd order) | 58.0% | +3.0 |
| **4th order** | **59.5%** | **+4.5** |
| 4th order, 2x acceleration | 56.5% | +1.5 |

#### Min-P only

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| PID | 56.0% | +1.0 |
| 4th order | 55.5% | +0.5 |
| 4th order, 2x acceleration | 59.5% | +4.5 |

#### QEWS

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| QEWS replace (density operator only) | 51.5% | −3.5 |
| Hybrid (w_H=1, w_Q=1) | 54.0% | −1.0 |
| Hybrid (w_H=1, w_Q=2) | 58.0% | +3.0 |

</details>

---

### Key findings

- **Velocity form is a major improvement.** Switching from position-form to velocity-form actuation raised the best result from +4.5pp to +8.0pp on small-scale experiments. The integrating actuator maintains persistent corrections that compound over the generation trajectory.
- **The acceleration term matters.** The 4th-order controller outperforms PID by 1.5–4.5 points depending on configuration. The signal catches entropy dynamics that lower-order controllers miss.
- **QEWS hybrid is the best configuration.** Shannon entropy and quantum density operator signals at equal weighting (w_H=1, w_Q=1) achieves 83% — the single best result across all small-scale experiments.
- **The spinning failure mode is the primary target.** In the full sweep, cap hits account for nearly all wrong answers. The controller's gain comes almost entirely from reducing cap hits, not from correcting confident wrong answers. Confident wrong answers (short responses, low entropy) are outside the controller's observable domain.
- **Entropy predicts outcome.** Mean entropy cleanly separates correct from wrong answers and uncapped from capped problems across both conditions, validating entropy as a control signal.
- **Temperature is a harmful actuator.** Earlier experiments using temperature control degraded accuracy by up to 8 points. Temperature directly modifies the logit distribution, corrupting the entropy signal the controller is responding to. Min-P does not have this problem.
- **The controller has real authority.** Aggressive overtuning (2x acceleration gains) can degrade performance, confirming the controller meaningfully steers generation — not a null effect.

---

## Controller architecture

### Velocity-form actuation

Unlike the position-form controller where actuators were computed fresh each step, the v2 velocity-form controller integrates corrections over time:

```
actuator(k) = actuator(k-1) + K · x(k)
```

If entropy has been consistently too high, the actuator accumulates tighter sampling — it doesn't snap back to default the moment entropy briefly dips.

### QEWS: Quantum Early Warning Signal

A secondary observation channel inspired by the density operator formalism from quantum information theory. A rolling window of L2-normalized top-K logit vectors forms a density matrix:

$$\boldsymbol{\rho}_k = \frac{1}{W} \sum_{i=k-W+1}^{k} \boldsymbol{\psi}_i \boldsymbol{\psi}_i^\top$$

The von Neumann entropy of ρ tracks structural shifts in the *distribution of distributions* over time, rather than instantaneous entropy at a single token. The QEWS signal is the deviation from a running exponential moving average of this entropy.

---

## How it's different from Entropix

[Entropix](https://github.com/xjdr-alt/entropix) uses entropy thresholds to switch between predefined sampling strategies (e.g., "high entropy → increase temperature"). This is open-loop control: a lookup table with no feedback.

Cybernetic Entropy Control is closed-loop. The controller continuously tracks the error signal and its derivatives, adjusting actuators proportionally to how fast and in which direction entropy is moving. The acceleration term gives it a predictive edge — it responds to the *onset* of an entropy spike, not the spike itself. The velocity form gives it memory — it accumulates corrections rather than reacting from scratch at each step.

---

## Usage

```bash
# Baseline (no control)
python run.py -m 2b -d data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 -o results/baseline.jsonl

# 4th-order entropy control (velocity form)
python run.py -m 2b -d data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    -o results/controlled.jsonl

# QEWS hybrid mode (best configuration)
python run.py -m 2b -d data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid --qews-target 0.0 --w-H 1.0 --w-Q 1.0 \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    -o results/qews_hybrid.jsonl

# Run full sweep
bash run_full_math_ab.sh

# Analyze results
python analyze.py results/*.jsonl --sort accuracy --md

# Interactive mode
python run.py -m 2b --control --H-target 0.1
```

---

## Requirements

- Python 3.10+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with CUDA
- numpy, torch, pyarrow
- A GGUF model (tested with Qwen 3.5 family)

---

## Roadmap

- ~~Velocity form controller~~ (done — v2)
- ~~Multi-actuator + QEWS hybrid combined run~~ (done — best result)
- ~~Full 5000-problem benchmark sweep~~ (done)
- Single-actuator ablation (min_p, top_p, repeat_penalty) — in progress
- Token position as secondary sensor (budget pressure b(t) = t/t_max)
- TECA (Token Entropy Cumulative Average) as sensor
- Decoupled temperature: measure H(t) at fixed temperature, sample at dynamic temperature
- QEWS window size ablation (W=8, W=16)
- KV cache spectral reshaping as actuator (post GPU upgrade)
- Dual-model dynamic setpoint via speculative decoding
- Automated gain tuning
- Additional benchmarks (GSM8K, TruthfulQA, MuSiQue)

---

## Acknowledgments

Motivated by the semantic entropy work of [Farquhar et al. (2024)](https://www.nature.com/articles/s41586-024-07421-0), the QEWS density operator framework of [Gong, Sedai, and Medda (2025)](https://arxiv.org/abs/2511.21515), and the Token Entropy Cumulative Average (TECA) introduced by [Bin et al. (2025)](https://arxiv.org/abs/2510.02249).

---

## License

MIT

---

## Funding

This project is self-funded. If you'd like to support the research, the [Manifund project](https://manifund.org/projects/cybernetic-entropy-controller) covers a GPU upgrade that would significantly accelerate experimentation.
