#!/usr/bin/env python3
"""Entropy controller testbench — plant model only, entropy as sole control signal.

Usage:
    # Interactive
    python testbench/run.py

    # Single prompt
    python testbench/run.py -p "explain DNS"

    # Dataset (parquet)
    python testbench/run.py -d data.parquet
    python testbench/run.py -d data.parquet --prompt-col question --limit 50

    # A/B comparison:
    python testbench/run.py -d data.parquet -o results/baseline.jsonl           # no control
    python testbench/run.py -d data.parquet -o results/controlled.jsonl --control  # entropy control
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import pyarrow.parquet as pq
from llama_cpp import Llama
from llama_cpp.llama_cpp import GGML_TYPE_Q8_0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "testbench", "results")

MODELS = {
    "9b": ("models/Qwen3.5-9B-UD-Q5_K_XL.gguf", 10240),
    "4b": ("models/Qwen3.5-4B-Q4_K_M.gguf",      10240),
    "2b": ("models/Qwen3.5-2B-Q4_K_M.gguf",      10240),
    "0.8b": ("models/Qwen3.5-0.8B-Q4_K_M.gguf",  10240),
}

# ── ANSI ─────────────────────────────────────────────────────────────────────
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── Entropy ──────────────────────────────────────────────────────────────────

def entropy(logits: np.ndarray, top_k: int = 64) -> float:
    """Shannon entropy over top-K softmaxed logits."""
    idx = np.argpartition(logits, -top_k)[-top_k:]
    v = logits[idx]
    v = v - v.max()
    e = np.exp(v)
    p = e / e.sum()
    mask = p > 0
    return -float(np.sum(p[mask] * np.log(p[mask])))


class ZScore:
    """Rolling z-score, O(1) per update."""
    def __init__(self, window: int = 128, warmup: int = 16):
        self.window = window
        self.warmup = warmup
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, v: float) -> float:
        if len(self._buf) == self.window:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(v)
        self._sum += v
        self._sum_sq += v * v
        if len(self._buf) < self.warmup:
            return 0.0
        n = len(self._buf)
        mu = self._sum / n
        var = self._sum_sq / n - mu * mu
        if var < 1e-16:
            return 0.0
        return (v - mu) / (var ** 0.5)

    def reset(self):
        self._buf.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


# ── 4th-order entropy controller ─────────────────────────────────────────────

class EntropyController:
    """4th-order feedback controller that drives entropy toward a target setpoint.

    State vector x = [integral, error, d_error, d2_error]
        x[0] = ∫ e(t) dt        (trapezoidal integral of error, with anti-windup)
        x[1] = e(k)             (current error = H_target - H)
        x[2] = e(k) - e(k-1)   (first difference / velocity)
        x[3] = Δe(k) - Δe(k-1) (second difference / acceleration)

    Actuators: min-p (M), top-p (P), frequency_penalty (F)
        ΔM = -(K_M · x)   positive error (too confident) → lower min-p (widen)
        ΔP =  K_P · x      positive error → raise top-p (widen)
        ΔF = -(K_F · x)   positive error → lower freq penalty (allow repeats)

    Error convention: e = H_target - H
        e > 0 → entropy is below target (too confident) → widen distribution
        e < 0 → entropy is above target (too uncertain) → tighten distribution
    """
    def __init__(
        self,
        H_target: float = 0.5,
        M_base: float = 0.05,
        P_base: float = 0.95,
        F_base: float = 0.0,
        # Gain vectors: [integral, error, d_error, d2_error]
        K_M: tuple = (0.005, 0.03, 0.04, 0.08),
        K_P: tuple = (0.005, 0.02, 0.03, 0.05),
        K_F: tuple = (0.002, 0.01, 0.015, 0.025),
        sigmoid_alpha: float = 1.0,
    ):
        self.H_target = H_target
        self.M_base = M_base
        self.P_base = P_base
        self.F_base = F_base
        self.K_M = np.array(K_M, dtype=np.float64)
        self.K_P = np.array(K_P, dtype=np.float64)
        self.K_F = np.array(K_F, dtype=np.float64)
        self.sigmoid_alpha = sigmoid_alpha

        # State: [integral, error, d_error, d2_error]
        self._x = np.zeros(4, dtype=np.float64)
        self._prev_error = 0.0
        self._prev_d_error = 0.0

    def step(self, H: float) -> tuple[float, float, float, float, float, float]:
        """Given current entropy H, return (M, P, F, dM, dP, dF)."""
        e_raw = self.H_target - H

        # Sigmoidal dampener: soft-limit large errors via tanh
        e = float(np.tanh(self.sigmoid_alpha * e_raw))

        # Derivatives (on dampened signal)
        d_e = e - self._prev_error
        d2_e = d_e - self._prev_d_error

        # Integral (trapezoidal)
        integral = self._x[0]
        integral += 0.5 * (e + self._prev_error)
        integral = max(-5.0, min(5.0, integral))

        # Update state vector
        self._x[0] = integral
        self._x[1] = e
        self._x[2] = d_e
        self._x[3] = d2_e

        # Actuation
        # Positive error (too confident) → lower M (widen), raise P (widen), lower F (less penalty)
        dM = -float(self.K_M @ self._x)
        dP = float(self.K_P @ self._x)
        dF = -float(self.K_F @ self._x)

        M = max(0.0, self.M_base + dM)
        P = max(0.0, min(1.0, self.P_base + dP))
        F = max(0.0, self.F_base + dF)

        # Save for next step
        self._prev_error = e
        self._prev_d_error = d_e

        return M, P, F, dM, dP, dF

    def reset(self):
        self._x[:] = 0.0
        self._prev_error = 0.0
        self._prev_d_error = 0.0


# ── QEWS (Quantum Early Warning Signal) ─────────────────────────────────────

class QEWSComputer:
    """Von Neumann entropy of a density operator built from rolling logit vectors.

    Maintains a window of W unit-normalized top-K logit vectors. Constructs
    rho = (1/W) * sum(|psi><psi|) and computes S(rho) = -tr(rho log rho).
    The QEWS signal is the deviation from the running mean of S(rho).
    """

    def __init__(self, W: int = 12, K: int = 64, ema_alpha: float = 0.05,
                 device: torch.device | None = None):
        self.W = W
        self.K = K
        self.ema_alpha = ema_alpha
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))

        # Rolling buffer of unit-normalized vectors: (W, K)
        self._buf = torch.zeros(W, K, device=self.device, dtype=torch.float32)
        self._count = 0       # how many vectors inserted so far
        self._ptr = 0         # circular buffer write pointer
        self._mu_S = 0.0      # EMA of S(rho)
        self._warmed_up = False

    def step(self, logits: np.ndarray) -> tuple[float, float, float]:
        """Ingest top-K logits, return (S_raw, qews_signal, mu_S).

        During burn-in (first W tokens), returns (0, 0, 0).
        """
        # Extract top-K logits and normalize to unit vector
        idx = np.argpartition(logits, -self.K)[-self.K:]
        v = torch.tensor(logits[idx], device=self.device, dtype=torch.float32)
        norm = torch.linalg.norm(v)
        if norm > 1e-12:
            v = v / norm

        # Insert into circular buffer
        self._buf[self._ptr] = v
        self._ptr = (self._ptr + 1) % self.W
        self._count += 1

        # Burn-in: need full window
        if self._count < self.W:
            return 0.0, 0.0, 0.0

        # Construct density operator: rho = (1/W) * sum(|psi><psi|)
        # _buf is (W, K), rho is (K, K)
        rho = (self._buf.T @ self._buf) / self.W

        # Von Neumann entropy: S = -sum(lambda * log(lambda))
        eigvals = torch.linalg.eigvalsh(rho)
        eigvals = eigvals.clamp(min=1e-12)
        S_raw = -float(torch.sum(eigvals * torch.log(eigvals)))

        # Update running mean (EMA)
        if not self._warmed_up:
            self._mu_S = S_raw
            self._warmed_up = True
        else:
            self._mu_S += self.ema_alpha * (S_raw - self._mu_S)

        signal = S_raw - self._mu_S
        return S_raw, signal, self._mu_S

    def reset(self):
        self._buf.zero_()
        self._count = 0
        self._ptr = 0
        self._mu_S = 0.0
        self._warmed_up = False


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class TokenLog:
    step: int
    token_id: int
    token_text: str
    H: float            # raw Shannon entropy
    error: float        # H_target - H (0 if no controller)
    M: float            # min-p
    P: float            # top-p
    F: float            # frequency penalty
    delta_M: float
    delta_P: float
    delta_F: float
    tps: float
    qews_raw: float = 0.0
    qews_signal: float = 0.0
    qews_mu: float = 0.0


@dataclass
class SampleResult:
    index: int
    prompt: str
    system: str
    reference: str
    output: str
    extracted_answer: str
    correct: bool | None   # None if no reference
    tokens_generated: int
    wall_time_s: float
    tok_per_sec: float
    mean_H: float
    max_H: float
    std_H: float
    controlled: bool
    token_log: list[TokenLog] = field(default_factory=list)


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    depth = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


import re as _re


def _try_eval_numeric(s: str) -> float | None:
    """Try to evaluate a string as a number or simple fraction."""
    s = s.strip()
    # Strip outer parens from each side of a fraction: (a)/(b) or a/b
    s = _re.sub(r"\(([^()]+)\)", r"\1", s)
    # a/b fraction
    m = _re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    # plain number
    try:
        return float(s)
    except ValueError:
        return None


def normalize_answer(s: str) -> str:
    """Normalize a math answer for comparison."""
    s = s.strip()

    # Strip LaTeX wrappers: \text{}, \mathrm{}, \mbox{}, \textbf{}, etc.
    s = _re.sub(r"\\(?:text|mathrm|mbox|textbf|textit|operatorname)\{", "", s)
    # Remove trailing braces left over
    while s.endswith("}") and s.count("}") > s.count("{"):
        s = s[:-1]

    # Remove cosmetic LaTeX: \left, \right, \!, \;, \:, \quad, \,
    s = _re.sub(r"\\(?:left|right|!|;|:|quad|,)", "", s)

    # Remove $, spaces
    s = s.replace("$", "").replace(" ", "")

    # Normalize all fraction commands → \frac
    s = _re.sub(r"\\[dtc]frac", r"\\frac", s)

    # \frac{a}{b} → a/b (simple, non-nested)
    s = _re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    # Comma-formatted numbers: 12{,}000 or 12,\!000 or 12,000 (if clearly a number)
    s = s.replace("{,}", "")
    s = _re.sub(r"(\d),(?=\d{3})", r"\1", s)

    # \sqrt{x} → sqrt(x), \sqrt[n]{x} → nrt(x)
    s = _re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r"\1rt(\2)", s)
    s = _re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\sqrt", "sqrt")

    # Common constants
    s = s.replace("\\infty", "inf")
    s = s.replace("\\pi", "pi")

    # \cdot → *
    s = s.replace("\\cdot", "*")

    # Escaped braces in sets: \{ → {, \} → }
    s = s.replace("\\{", "{").replace("\\}", "}")

    s = s.lower()
    return s


def _is_unordered_sequence(s: str) -> bool:
    """Check if s looks like a comma-separated list (not a tuple/interval)."""
    # Tuples (a,b) and intervals [a,b] are ordered — don't sort those
    if s.startswith("(") and s.endswith(")"):
        return False
    if s.startswith("[") and s.endswith("]"):
        return False
    if s.startswith("[") and s.endswith(")"):
        return False
    if s.startswith("(") and s.endswith("]"):
        return False
    return "," in s


def _sort_elements(s: str) -> str:
    """Sort comma-separated elements for order-independent comparison."""
    parts = [p.strip() for p in s.split(",")]
    return ",".join(sorted(parts))


def _numeric_equal(a: str, b: str) -> bool:
    """Try numeric comparison of two normalized strings."""
    va = _try_eval_numeric(a)
    vb = _try_eval_numeric(b)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-6
    return False


def check_answer(output: str, reference: str) -> tuple[str, bool | None]:
    """Extract answer from output and compare to reference."""
    extracted = extract_boxed(output)
    if not reference:
        return extracted, None
    if not extracted:
        return "", False

    n_ext = normalize_answer(extracted)
    n_ref = normalize_answer(reference)

    # Direct string match
    if n_ext == n_ref:
        return extracted, True

    # Numeric match (handles 0.5 vs 1/2 etc.)
    if _numeric_equal(n_ext, n_ref):
        return extracted, True

    # Unordered sequence match: "2,-3,4" == "-3,2,4"
    if _is_unordered_sequence(n_ext) and _is_unordered_sequence(n_ref):
        ext_parts = sorted(p.strip() for p in n_ext.split(","))
        ref_parts = sorted(p.strip() for p in n_ref.split(","))
        if len(ext_parts) == len(ref_parts):
            # Try element-wise: string match or numeric match
            if all(a == b or _numeric_equal(a, b)
                   for a, b in zip(ext_parts, ref_parts)):
                return extracted, True

    return extracted, False


# ── Generation ───────────────────────────────────────────────────────────────

def format_prompt(system: str, user: str, thinking: bool) -> str:
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    s += f"<|im_start|>user\n{user}<|im_end|>\n"
    s += "<|im_start|>assistant\n<think>\n" if thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return s


def generate(
    plant: Llama,
    system: str,
    prompt: str,
    *,
    controller: EntropyController | None = None,
    qews_controller: EntropyController | None = None,
    qews: QEWSComputer | None = None,
    qews_mode: str = "off",          # "off", "replace", "hybrid"
    w_H: float = 1.0,                # entropy controller weight (hybrid)
    w_Q: float = 1.0,                # QEWS controller weight (hybrid)
    max_tokens: int = 2048,
    temperature: float = 0.7,
    min_p: float = 0.05,
    top_k: int = 20,
    top_p: float = 0.95,
    repeat_penalty: float = 1.05,
    thinking_budget: int = 0,
    live: bool = True,
) -> tuple[str, list[TokenLog]]:
    """Generate a response. Returns (output_text, token_log)."""

    thinking = thinking_budget > 0
    prompt_text = format_prompt(system, prompt, thinking)
    toks = plant.tokenize(prompt_text.encode("utf-8"), add_bos=True, special=True)

    ctx_size = plant.n_ctx()
    max_tokens = min(max_tokens, ctx_size - 128)
    avail = max(128, ctx_size - max_tokens - 16)
    if len(toks) > avail:
        toks = toks[:avail]

    plant.reset()
    plant.eval(toks)

    eos = plant.token_eos()
    im_end = plant.tokenize("<|im_end|>".encode(), add_bos=False, special=True)
    im_end_id = im_end[-1] if im_end else None
    think_end = plant.tokenize("</think>\n\n".encode(), add_bos=False, special=True)

    if controller is not None:
        controller.reset()
    if qews_controller is not None:
        qews_controller.reset()
    if qews is not None:
        qews.reset()

    in_thinking = thinking
    think_count = 0
    out: list[str] = []
    log: list[TokenLog] = []
    t0 = time.monotonic()

    if live:
        w = 90
        print(f"\n{DIM}{'─' * w}{RESET}")
        hdr = f"{'tok':>5}  {'H':>7}  {'err':>7}  {'∫err':>7}  {'M':>6}  {'P':>5}  {'F':>5}  {'dM':>6}  {'dP':>6}  {'dF':>6}  {'tok/s':>6}  token"
        print(f"{DIM}{hdr}{RESET}")
        print(f"{DIM}{'─' * w}{RESET}")

    for step in range(max_tokens):
        # Entropy from current logits
        logits = np.ctypeslib.as_array(
            plant._ctx.get_logits(), shape=(plant._n_vocab,)).copy()
        H = entropy(logits, 64)

        # QEWS — always compute if present (for logging), regardless of mode
        qews_raw, qews_sig, qews_mu = 0.0, 0.0, 0.0
        if qews is not None:
            qews_raw, qews_sig, qews_mu = qews.step(logits)

        # Control
        M, P, F = min_p, top_p, 0.0
        dM, dP, dF = 0.0, 0.0, 0.0
        err = 0.0
        integral = 0.0

        if qews_mode == "replace" and qews_controller is not None:
            M, P, F, dM, dP, dF = qews_controller.step(qews_sig)
            err = qews_controller._x[1]
            integral = qews_controller._x[0]
        elif qews_mode == "hybrid" and controller is not None and qews_controller is not None:
            M_h, P_h, F_h, dM_h, dP_h, dF_h = controller.step(H)
            M_q, P_q, F_q, dM_q, dP_q, dF_q = qews_controller.step(qews_sig)
            dM = w_H * dM_h + w_Q * dM_q
            dP = w_H * dP_h + w_Q * dP_q
            dF = w_H * dF_h + w_Q * dF_q
            M = max(0.0, controller.M_base + dM)
            P = max(0.0, min(1.0, controller.P_base + dP))
            F = max(0.0, controller.F_base + dF)
            err = controller._x[1]
            integral = controller._x[0]
        elif controller is not None:
            M, P, F, dM, dP, dF = controller.step(H)
            err = controller._x[1]
            integral = controller._x[0]

        # Sample
        token_id = plant.sample(
            top_k=top_k, top_p=P, min_p=M, temp=temperature,
            frequency_penalty=F,
        )

        if token_id == eos or token_id == im_end_id:
            break

        text = plant.detokenize([token_id]).decode("utf-8", errors="replace")
        plant.eval([token_id])

        # Thinking boundary
        is_content = not in_thinking
        if in_thinking:
            think_count += 1
            if thinking_budget > 0 and think_count >= thinking_budget:
                in_thinking = False
                if think_end:
                    plant.eval(think_end)
            elif "</think>" in text:
                in_thinking = False
        else:
            out.append(text)

        elapsed = time.monotonic() - t0
        tps = (step + 1) / elapsed if elapsed > 0 else 0

        tl = TokenLog(step=step, token_id=token_id, token_text=text,
                      H=H, error=err, M=M, P=P, F=F,
                      delta_M=dM, delta_P=dP, delta_F=dF, tps=tps,
                      qews_raw=qews_raw, qews_signal=qews_sig, qews_mu=qews_mu)
        log.append(tl)

        if live:
            disp = repr(text)
            if not is_content:
                disp = f"{DIM}{disp}{RESET}"
            ec = RED if abs(err) > 1.0 else (YELLOW if abs(err) > 0.5 else RESET)
            print(f"{step:5d}  {ec}{H:7.3f}{RESET}  {ec}{err:+7.3f}{RESET}  {integral:+7.3f}  "
                  f"{M:6.3f}  {P:5.2f}  {F:5.3f}  {dM:+6.3f}  {dP:+6.3f}  {dF:+6.3f}  {tps:6.1f}  {disp}")

    if live:
        elapsed = time.monotonic() - t0
        n = len(log)
        print(f"{DIM}{'─' * 90}{RESET}")
        print(f"{BOLD}{n} tokens in {elapsed:.2f}s = {n/elapsed:.1f} tok/s{RESET}")

    return "".join(out), log


# ── Dataset ──────────────────────────────────────────────────────────────────

def detect_col(columns: list[str], hints: list[str]) -> str | None:
    for h in hints:
        for c in columns:
            if h in c.lower():
                return c
    return None


def load_dataset(path: str, prompt_col: str | None, system_col: str | None,
                 reference_col: str | None, limit: int | None, offset: int) -> list[dict]:
    table = pq.read_table(path)
    cols = table.column_names

    if prompt_col is None:
        prompt_col = detect_col(cols, ["prompt", "question", "input", "text",
                                       "instruction", "query", "content", "problem"])
    if prompt_col is None or prompt_col not in cols:
        print(f"{RED}Cannot find prompt column. Columns: {cols}{RESET}")
        print(f"Use --prompt-col to specify.")
        sys.exit(1)

    if reference_col is None:
        reference_col = detect_col(cols, ["answer", "response", "output", "target",
                                          "completion", "reference", "gold", "label", "solution"])
    if system_col and system_col not in cols:
        print(f"{RED}System column '{system_col}' not found.{RESET}")
        sys.exit(1)

    end = min(offset + limit, table.num_rows) if limit else table.num_rows
    rows = []
    for i in range(offset, end):
        p = str(table.column(prompt_col)[i].as_py())
        s = str(table.column(system_col)[i].as_py()) if system_col else ""
        r = str(table.column(reference_col)[i].as_py()) if reference_col and table.column(reference_col)[i].as_py() is not None else ""
        rows.append({"prompt": p, "system": s, "reference": r})

    print(f"{CYAN}Dataset: {os.path.basename(path)} ({table.num_rows} rows){RESET}")
    print(f"{CYAN}  prompt_col={prompt_col}" +
          (f"  system_col={system_col}" if system_col else "") +
          (f"  reference_col={reference_col}" if reference_col else "") + RESET)
    print(f"{CYAN}  using {len(rows)} rows (offset={offset}){RESET}")
    return rows


def run_dataset(plant: Llama, dataset: list[dict], default_system: str,
                output_path: str, **gen_kwargs):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_tokens = 0
    total_time = 0.0
    all_H: list[float] = []
    n_correct = 0
    n_scored = 0

    with open(output_path, "w") as f:
        for i, row in enumerate(dataset):
            sys_prompt = row["system"] or default_system
            prompt = row["prompt"]

            print(f"\n{CYAN}[{i+1}/{len(dataset)}]{RESET} {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            t0 = time.monotonic()
            output, token_log = generate(
                plant, sys_prompt, prompt, live=False, **gen_kwargs)
            elapsed = time.monotonic() - t0

            Hs = [t.H for t in token_log]
            n = len(token_log)
            tps = n / elapsed if elapsed > 0 else 0

            extracted, correct = check_answer(output, row.get("reference", ""))
            if correct is not None:
                n_scored += 1
                if correct:
                    n_correct += 1

            result = SampleResult(
                index=i, prompt=prompt, system=sys_prompt,
                reference=row.get("reference", ""), output=output,
                extracted_answer=extracted, correct=correct,
                tokens_generated=n, wall_time_s=round(elapsed, 3),
                tok_per_sec=round(tps, 1),
                mean_H=round(float(np.mean(Hs)), 4) if Hs else 0,
                max_H=round(float(np.max(Hs)), 4) if Hs else 0,
                std_H=round(float(np.std(Hs)), 4) if Hs else 0,
                controlled=gen_kwargs.get("controller") is not None or gen_kwargs.get("qews_mode", "off") != "off",
                token_log=token_log,
            )
            rec = asdict(result)
            rec["token_log"] = [asdict(t) for t in token_log]
            f.write(json.dumps(rec) + "\n")
            f.flush()

            total_tokens += n
            total_time += elapsed
            all_H.extend(Hs)

            # Status line
            acc_str = ""
            if n_scored > 0:
                acc_str = f"  acc={n_correct}/{n_scored} ({100*n_correct/n_scored:.1f}%)"
            mark = ""
            if correct is True:
                mark = f"{GREEN}✓{RESET}"
            elif correct is False:
                mark = f"{RED}✗{RESET} got={extracted} want={row.get('reference','')}"

            print(f"  {DIM}{n} tok, {tps:.0f} tok/s, H={result.mean_H:.3f}±{result.std_H:.3f}{RESET}  {mark}")
            print(f"  {DIM}{output[:120]}{'...' if len(output) > 120 else ''}{RESET}")
            if acc_str:
                print(f"  {DIM}running{acc_str}{RESET}")

    print(f"\n{DIM}{'═' * 78}{RESET}")
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"{BOLD}Done: {len(dataset)} samples, {total_tokens} tokens, "
          f"{total_time:.1f}s, {avg_tps:.1f} tok/s{RESET}")
    if all_H:
        print(f"{BOLD}Entropy: mean={np.mean(all_H):.4f}  std={np.std(all_H):.4f}  "
              f"max={np.max(all_H):.4f}{RESET}")
    if n_scored > 0:
        print(f"{BOLD}Accuracy: {n_correct}/{n_scored} = {100*n_correct/n_scored:.1f}%{RESET}")
    print(f"{BOLD}Results: {output_path}{RESET}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Entropy controller testbench")

    p.add_argument("--prompt", "-p", type=str, help="Single prompt")
    p.add_argument("--dataset", "-d", type=str, help="Parquet dataset path")
    p.add_argument("--prompt-col", type=str)
    p.add_argument("--system-col", type=str)
    p.add_argument("--reference-col", type=str)
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output", "-o", type=str)

    MATH_SYSTEM = ("Solve the following math problem step by step. "
                   "Put your final answer in \\boxed{}.")
    p.add_argument("--system", "-s", type=str, default=None,
                   help="System prompt (defaults to math prompt for math datasets)")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", "-t", type=float, default=0.7)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repeat-penalty", type=float, default=1.05)
    p.add_argument("--thinking", type=int, default=0)

    p.add_argument("--model", "-m", type=str, default="9b",
                   choices=list(MODELS.keys()),
                   help="Model size (default: 9b)")
    p.add_argument("--control", action="store_true",
                   help="Enable 4th-order entropy control")
    p.add_argument("--H-target", type=float, default=0.5,
                   help="Target entropy setpoint (default 0.5)")
    p.add_argument("--K-M", type=float, nargs=4, default=[0.005, 0.03, 0.04, 0.08],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="Min-P gain vector [integral, error, d_error, d2_error]")
    p.add_argument("--K-P", type=float, nargs=4, default=[0.005, 0.02, 0.03, 0.05],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="Top-P gain vector [integral, error, d_error, d2_error]")
    p.add_argument("--K-F", type=float, nargs=4, default=[0.002, 0.01, 0.015, 0.025],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="Frequency penalty gain vector [integral, error, d_error, d2_error]")
    p.add_argument("--sigmoid-alpha", type=float, default=1.0,
                   help="Sigmoidal dampener steepness (default 1.0)")

    # QEWS args
    p.add_argument("--qews-mode", choices=["off", "replace", "hybrid"], default="off",
                   help="QEWS mode: off, replace (QEWS replaces entropy), hybrid (weighted sum)")
    p.add_argument("--qews-target", type=float, default=0.0,
                   help="QEWS target setpoint (default 0.0 = track running mean)")
    p.add_argument("--qews-K-M", type=float, nargs=4, default=[0.005, 0.03, 0.04, 0.08],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="QEWS controller Min-P gain vector")
    p.add_argument("--qews-K-P", type=float, nargs=4, default=[0.005, 0.02, 0.03, 0.05],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="QEWS controller Top-P gain vector")
    p.add_argument("--qews-K-F", type=float, nargs=4, default=[0.002, 0.01, 0.015, 0.025],
                   metavar=("Ki", "Kp", "Kd", "Kdd"),
                   help="QEWS controller Freq penalty gain vector")
    p.add_argument("--qews-alpha", type=float, default=1.0,
                   help="QEWS controller tanh dampener steepness")
    p.add_argument("--w-H", type=float, default=1.0,
                   help="Entropy controller weight in hybrid mode")
    p.add_argument("--w-Q", type=float, default=1.0,
                   help="QEWS controller weight in hybrid mode")

    args = p.parse_args()

    # Auto-detect system prompt for math datasets
    if args.system is None:
        if args.dataset and "math" in os.path.basename(args.dataset).lower():
            args.system = MATH_SYSTEM
        else:
            args.system = "You are a helpful assistant."

    # ── Load model ───────────────────────────────────────────────────
    model_rel, model_ctx = MODELS[args.model]
    model_path = os.path.join(ROOT, model_rel)
    if not os.path.exists(model_path):
        print(f"{RED}Model not found: {model_path}{RESET}")
        sys.exit(1)
    print(f"{CYAN}Loading {args.model} ({os.path.basename(model_rel)})...{RESET}")
    plant = Llama(
        model_path=model_path, n_gpu_layers=99, n_ctx=model_ctx,
        flash_attn=True, verbose=False,
        type_k=GGML_TYPE_Q8_0, type_v=GGML_TYPE_Q8_0,
    )

    # ── Controller ───────────────────────────────────────────────────
    ctrl = None
    if args.control or args.qews_mode == "hybrid":
        ctrl = EntropyController(
            H_target=args.H_target,
            M_base=args.min_p, P_base=args.top_p, F_base=0.0,
            K_M=tuple(args.K_M), K_P=tuple(args.K_P), K_F=tuple(args.K_F),
            sigmoid_alpha=args.sigmoid_alpha,
        )
        print(f"{CYAN}Entropy control: ON  H_target={args.H_target}  "
              f"K_M={args.K_M}  K_P={args.K_P}  K_F={args.K_F}{RESET}")
    else:
        print(f"{CYAN}Entropy control: OFF{RESET}")

    # ── QEWS ────────────────────────────────────────────────────────
    qews_comp = None
    qews_ctrl = None
    if args.qews_mode != "off":
        qews_comp = QEWSComputer()
        qews_ctrl = EntropyController(
            H_target=args.qews_target,
            M_base=args.min_p, P_base=args.top_p, F_base=0.0,
            K_M=tuple(args.qews_K_M), K_P=tuple(args.qews_K_P), K_F=tuple(args.qews_K_F),
            sigmoid_alpha=args.qews_alpha,
        )
        print(f"{CYAN}QEWS: {args.qews_mode}  target={args.qews_target}  "
              f"K_M={args.qews_K_M}  K_P={args.qews_K_P}  K_F={args.qews_K_F}{RESET}")
        if args.qews_mode == "hybrid":
            print(f"{CYAN}  w_H={args.w_H}  w_Q={args.w_Q}{RESET}")

    gen_kwargs = dict(
        controller=ctrl,
        qews_controller=qews_ctrl,
        qews=qews_comp,
        qews_mode=args.qews_mode,
        w_H=args.w_H,
        w_Q=args.w_Q,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        thinking_budget=args.thinking,
    )

    # ── Dataset mode ─────────────────────────────────────────────────
    if args.dataset:
        dataset = load_dataset(args.dataset, args.prompt_col, args.system_col,
                               args.reference_col, args.limit, args.offset)
        if args.output:
            out = args.output
        else:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.dataset))[0]
            tag = "controlled" if (args.control or args.qews_mode != "off") else "baseline"
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(RESULTS_DIR, f"{base}_{args.model}_{tag}_{ts}.jsonl")
        run_dataset(plant, dataset, args.system, out, **gen_kwargs)

    # ── Single prompt ────────────────────────────────────────────────
    elif args.prompt:
        generate(plant, args.system, args.prompt, live=True, **gen_kwargs)

    # ── Interactive ──────────────────────────────────────────────────
    else:
        print(f"{YELLOW}Interactive. Type a message, /quit to exit, /control /off to toggle.{RESET}\n")
        while True:
            try:
                inp = input(f"{GREEN}> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not inp or inp == "/quit":
                break
            if inp == "/control":
                ctrl = EntropyController(T_base=args.temperature, M_base=args.min_p,
                                         gain_T=args.gain_T, gain_M=args.gain_M)
                gen_kwargs["controller"] = ctrl
                print(f"{CYAN}Control: ON{RESET}")
                continue
            if inp == "/off":
                ctrl = None
                gen_kwargs["controller"] = None
                print(f"{CYAN}Control: OFF{RESET}")
                continue
            generate(plant, args.system, inp, live=True, **gen_kwargs)

    plant.close()
    print(f"\n{DIM}Done.{RESET}")


if __name__ == "__main__":
    main()
