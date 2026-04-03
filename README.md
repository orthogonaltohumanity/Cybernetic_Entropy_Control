# Cybernetic Entropy Control

A 4th-order feedback controller that adjusts LLM sampling parameters in real-time based on token-level entropy. On MATH benchmark, it improves accuracy from 55% to 59.5% over an uncontrolled baseline with untuned gains.

## What it does

At each token generation step, the controller:

1. Computes Shannon entropy over the top-64 softmaxed logits
2. Feeds the error (target − actual) through a tanh dampener
3. Updates a 4th-order state vector: integral, error, velocity (Δe), acceleration (Δ²e)
4. Adjusts Min-P, Top-P, and frequency penalty via independent gain vectors

The acceleration term is the key contribution — it catches the upward curvature in entropy that precedes a hallucination spike, enabling intervention before it peaks.

## Results

All results: Qwen 3.5 2B (Q4\_K\_M), 200 problems from MATH (Hendrycks et al.), 4096 token budget, llama.cpp with CUDA.

### Multi-actuator control (Min-P + Top-P + Frequency Penalty)

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| PID (3rd order) | 58.0% | +3.0 |
| **4th order** | **59.5%** | **+4.5** |
| 4th order, 2x acceleration | 56.5% | +1.5 |

### Min-P only

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| PID | 56.0% | +1.0 |
| 4th order | 55.5% | +0.5 |
| 4th order, 2x acceleration | 59.5% | +4.5 |

### QEWS (Quantum Early Warning Signal)

| Setup | Accuracy | Δ |
|---|---|---|
| Baseline (uncontrolled) | 55.0% | — |
| QEWS replace (density operator only) | 51.5% | −3.5 |
| Hybrid (w\_H=1, w\_Q=1) | 54.0% | −1.0 |
| Hybrid (w\_H=1, w\_Q=2) | 58.0% | +3.0 |

### Key findings

- **The acceleration term matters.** The 4th-order controller outperforms PID by 1.5–4.5 points depending on configuration. The jerk signal catches entropy dynamics that lower-order controllers miss.
- **Temperature is a harmful actuator.** Earlier experiments using temperature control degraded accuracy by up to 8 points. Temperature directly modifies the logit distribution, corrupting the entropy signal the controller is responding to. Min-P does not have this problem.
- **QEWS contains real information.** The density operator signal alone underperforms, but as a supplementary channel at 2x weight it adds 3 points over baseline. It detects structural shifts in the distribution that Shannon entropy misses.
- **The controller has real authority.** Aggressive overtuning can degrade performance significantly, confirming the controller meaningfully steers generation — not a null effect.

## QEWS: Quantum Early Warning Signal

A secondary observation channel inspired by the density operator formalism from quantum information theory. A rolling window of L2-normalized top-K logit vectors forms a density matrix:

$$\boldsymbol{\rho}_k = \frac{1}{W} \sum_{i=k-W+1}^{k} \boldsymbol{\psi}_i \boldsymbol{\psi}_i^\top$$

The von Neumann entropy of ρ tracks structural shifts in the *distribution of distributions* over time, rather than instantaneous entropy at a single token. The QEWS signal is the deviation from a running exponential moving average of this entropy.

## How it's different from Entropix

[Entropix](https://github.com/xjdr-alt/entropix) uses entropy thresholds to switch between predefined sampling strategies (e.g., "high entropy → increase temperature"). This is open-loop control: a lookup table with no feedback.

Cybernetic Entropy Control is closed-loop. The controller continuously tracks the error signal and its derivatives, adjusting actuators proportionally to how fast and in which direction entropy is moving. The acceleration term gives it a predictive edge — it responds to the *onset* of an entropy spike, not the spike itself.

## Usage

```bash
# Baseline (no control)
python testbench/run.py -m 2b -d testbench/data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 -o results/baseline.jsonl

# 4th-order entropy control
python testbench/run.py -m 2b -d testbench/data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    -o results/controlled.jsonl

# QEWS hybrid mode
python testbench/run.py -m 2b -d testbench/data/hendrycks_math.parquet \
    --limit 200 --max-tokens 4096 --control --H-target 0.1 \
    --K-M 0.005 0.03 0.04 0.08 \
    --K-P 0.005 0.02 0.03 0.05 \
    --K-F 0.002 0.01 0.015 0.025 \
    --qews-mode hybrid --qews-target 0.0 --w-H 1.0 --w-Q 2.0 \
    --qews-K-M 0.005 0.03 0.04 0.08 \
    --qews-K-P 0.005 0.02 0.03 0.05 \
    --qews-K-F 0.002 0.01 0.015 0.025 \
    -o results/qews_hybrid.jsonl

# Interactive mode
python testbench/run.py -m 2b --control --H-target 0.1
```

## Requirements

- Python 3.10+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with CUDA
- numpy, torch, pyarrow
- A GGUF model (tested with Qwen 3.5 family)

## Roadmap

- Velocity form controller (accumulating actuator state)
- Multi-actuator + QEWS hybrid combined run
- QEWS window size ablation
- Dual-model dynamic setpoint (9B + 0.8B via speculative decoding)
- Automated gain tuning
- Additional benchmarks (TruthfulQA, SimpleQA, GSM8K)

## Acknowledgments

Motivated by the semantic entropy work of [Farquhar et al. (2024)](https://www.nature.com/articles/s41586-024-07421-0) and the QEWS density operator framework of [Gong, Sedai, and Medda (2025)](https://arxiv.org/abs/2511.21515).

## License

MIT

## Funding

This project is self-funded. If you'd like to support the research, the [Manifund project]([https://manifund.org](https://manifund.org/projects/cybernetic-entropy-controller]) covers a GPU upgrade that would significantly accelerate experimentation.
