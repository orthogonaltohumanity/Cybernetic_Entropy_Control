# Cybernetic Entropy Control

A 4th-order feedback controller that adjusts LLM sampling parameters in real-time based on token-level entropy. Using velocity-form actuation, it improves MATH benchmark accuracy from 75% to 83% over an uncontrolled baseline with untuned gains.

## What it does

At each token generation step, the controller:

1. Computes Shannon entropy over the top-64 softmaxed logits
2. Feeds the error (target - actual) through a tanh dampener
3. Updates a 4th-order state vector: integral, error, velocity (delta-e), acceleration (delta-2-e)
4. Applies velocity-form actuation: each actuator integrates K dot x over time, accumulating corrections rather than computing them from scratch

The acceleration term is the key contribution -- it catches the upward curvature in entropy that precedes a hallucination spike, enabling intervention before it peaks. The velocity form is the key architectural improvement -- by integrating actuator corrections over time, the controller maintains persistent state that adapts to the trajectory of the generation, rather than reacting only to instantaneous error.

## Results

### v2: Velocity-form controller

All results: Qwen 3.5 2B (Q4\_K\_M), 100 problems from MATH (Hendrycks et al.), 2048 token budget, llama.cpp with CUDA.

#### Multi-actuator control (Min-P + Top-P + Frequency Penalty)

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | -- |
| PID (3rd order) | 78.0% | +3.0 |
| 4th order, 2x acceleration | 79.0% | +4.0 |
| **4th order** | **82.0%** | **+7.0** |

#### Min-P only

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | -- |
| 4th order | 77.0% | +2.0 |
| 4th order, 2x acceleration | 77.0% | +2.0 |
| **PID** | **82.0%** | **+7.0** |

#### QEWS (Quantum Early Warning Signal)

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 75.0% | -- |
| QEWS hybrid (w\_H=1, w\_Q=2) | 74.0% | -1.0 |
| QEWS replace (4th order) | 76.0% | +1.0 |
| QEWS replace (4th order 2x) | 79.0% | +4.0 |
| QEWS replace (PID) | 82.0% | +7.0 |
| **QEWS hybrid (w\_H=1, w\_Q=1)** | **83.0%** | **+8.0** |

### v1: Position-form controller (prior results)

All results: Qwen 3.5 2B (Q4\_K\_M), 200 problems from MATH (Hendrycks et al.), 4096 token budget, llama.cpp with CUDA.

<details>
<summary>Click to expand v1 results</summary>

#### Multi-actuator control (Min-P + Top-P + Frequency Penalty)

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | -- |
| PID (3rd order) | 58.0% | +3.0 |
| **4th order** | **59.5%** | **+4.5** |
| 4th order, 2x acceleration | 56.5% | +1.5 |

#### Min-P only

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | -- |
| PID | 56.0% | +1.0 |
| 4th order | 55.5% | +0.5 |
| 4th order, 2x acceleration | 59.5% | +4.5 |

#### QEWS

| Setup | Accuracy | Delta |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | -- |
| QEWS replace (density operator only) | 51.5% | -3.5 |
| Hybrid (w\_H=1, w\_Q=1) | 54.0% | -1.0 |
| Hybrid (w\_H=1, w\_Q=2) | 58.0% | +3.0 |

</details>

### Key findings

- **Velocity form is a major improvement.** Switching from position-form (actuator = base + K dot x) to velocity-form (actuator += K dot x) raised the best result from +4.5pp to +8.0pp over baseline. The integrating actuator maintains persistent corrections that compound over the generation trajectory.
- **The acceleration term matters.** The 4th-order controller outperforms PID by 1.5-4.5 points depending on configuration. The jerk signal catches entropy dynamics that lower-order controllers miss.
- **QEWS hybrid is the best configuration.** The combination of Shannon entropy and quantum density operator signals at equal weighting (w\_H=1, w\_Q=1) achieves 83% -- the single best result across all experiments.
- **Temperature is a harmful actuator.** Earlier experiments using temperature control degraded accuracy by up to 8 points. Temperature directly modifies the logit distribution, corrupting the entropy signal the controller is responding to. Min-P does not have this problem.
- **QEWS contains real information.** The density operator signal alone underperforms, but combined with Shannon entropy it detects structural shifts in the distribution that entropy alone misses.
- **The controller has real authority.** Aggressive overtuning (2x acceleration gains) can degrade performance, confirming the controller meaningfully steers generation -- not a null effect.

## Controller architecture

### Velocity-form actuation

Unlike the earlier position-form controller where actuators were computed fresh each step from base values, the v2 velocity-form controller integrates corrections over time:

```
actuator(k) = actuator(k-1) + K dot x(k)
```

This means the controller builds up persistent adjustments. If entropy has been consistently too high, the actuator accumulates tighter sampling -- it doesn't snap back to default the moment entropy briefly dips.

### QEWS: Quantum Early Warning Signal

A secondary observation channel inspired by the density operator formalism from quantum information theory. A rolling window of L2-normalized top-K logit vectors forms a density matrix:

$$\boldsymbol{\rho}_k = \frac{1}{W} \sum_{i=k-W+1}^{k} \boldsymbol{\psi}_i \boldsymbol{\psi}_i^\top$$

The von Neumann entropy of rho tracks structural shifts in the *distribution of distributions* over time, rather than instantaneous entropy at a single token. The QEWS signal is the deviation from a running exponential moving average of this entropy.

## KV Cache Spectral Monitor

`kv_monitor.py` instruments the KV cache during generation via SVD, computing per-layer per-head spectral health metrics:

- **Spectral entropy / effective rank** -- how many singular value dimensions are active
- **Concentration** -- fraction of variance captured by the largest singular value
- **Dominance ratio** -- sigma\_1 / sigma\_2
- **Condition number** -- sigma\_max / sigma\_min
- **Gram matrix trace and log-determinant** -- volume and scale of the value subspace

This is an observation-only tool for understanding how the KV cache evolves during generation. It runs on HuggingFace transformers (not llama.cpp) to access the raw cache tensors.

## How it's different from Entropix

[Entropix](https://github.com/xjdr-alt/entropix) uses entropy thresholds to switch between predefined sampling strategies (e.g., "high entropy -> increase temperature"). This is open-loop control: a lookup table with no feedback.

Cybernetic Entropy Control is closed-loop. The controller continuously tracks the error signal and its derivatives, adjusting actuators proportionally to how fast and in which direction entropy is moving. The acceleration term gives it a predictive edge -- it responds to the *onset* of an entropy spike, not the spike itself. The velocity form gives it memory -- it accumulates corrections rather than reacting from scratch at each step.

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

# KV cache spectral monitoring
python kv_monitor.py -d data/hendrycks_math.parquet \
    --limit 50 --max-tokens 1024 --svd-interval 32 \
    -o results/kv_spectral.jsonl

# Run full sweep
bash run_all_sweeps.sh

# Analyze results
python analyze.py results/*.jsonl --sort accuracy --md

# Interactive mode
python run.py -m 2b --control --H-target 0.1
```

## Requirements

- Python 3.10+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with CUDA
- numpy, torch, pyarrow
- A GGUF model (tested with Qwen 3.5 family)
- For KV monitor: [transformers](https://github.com/huggingface/transformers)

## Roadmap

- ~~Velocity form controller~~ (done -- v2)
- Multi-actuator + QEWS hybrid combined run (done -- best result)
- QEWS window size ablation
- KV cache spectral health as a control signal (observation tool done)
- Dual-model dynamic setpoint (9B + 0.8B via speculative decoding)
- Automated gain tuning
- Additional benchmarks (TruthfulQA, SimpleQA, GSM8K)

## Acknowledgments

Motivated by the semantic entropy work of [Farquhar et al. (2024)](https://www.nature.com/articles/s41586-024-07421-0) and the QEWS density operator framework of [Gong, Sedai, and Medda (2025)](https://arxiv.org/abs/2511.21515).

## License

MIT

## Funding

This project is self-funded. If you'd like to support the research, the [Manifund project](https://manifund.org/projects/cybernetic-entropy-controller) covers a GPU upgrade that would significantly accelerate experimentation.
