#!/usr/bin/env python3
"""KV Cache Spectral Health Monitor — observation only.

Runs Qwen 3.5 2B via HuggingFace transformers (float16), extracts V matrices
from the KV cache at regular intervals during generation, computes spectral
health metrics via SVD, and logs everything to JSONL.

Usage:
    python testbench/kv_monitor.py \
        -d testbench/data/hendrycks_math.parquet \
        --limit 50 --offset 3000 \
        --max-tokens 1024 --svd-interval 32 \
        -o testbench/results/kv_spectral_2b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import answer-checking logic from existing testbench
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import extract_boxed, normalize_answer, check_answer

# ── ANSI ─────────────────────────────────────────────────────────────────────
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MATH_SYSTEM = ("Solve the following math problem step by step. "
               "Put your final answer in \\boxed{}.")


# ── Spectral metrics ────────────────────────────────────────────────────────

def compute_spectral_metrics(sigmas: torch.Tensor, gram: torch.Tensor) -> dict:
    """Given singular values and Gram matrix (V^T V), compute health metrics."""
    sigmas = sigmas.clamp(min=1e-12)
    total = sigmas.sum()
    normalized = sigmas / total

    spectral_entropy = -torch.sum(normalized * torch.log(normalized)).item()
    effective_rank = torch.exp(torch.tensor(spectral_entropy)).item()
    concentration = (sigmas[0] / total).item()
    dominance_ratio = (sigmas[0] / sigmas[1]).item() if len(sigmas) > 1 else float('inf')
    top5 = sigmas[:5].tolist()

    # Condition number: σ_max / σ_min
    condition_number = (sigmas[0] / sigmas[-1]).item()

    # Gram matrix metrics: tr(V^T V) and det(V^T V)
    gram_trace = torch.trace(gram).item()
    # Log-det for numerical stability (det can be astronomically large or tiny)
    gram_logdet = torch.linalg.slogdet(gram)
    gram_logdet_sign = gram_logdet.sign.item()
    gram_logdet_val = gram_logdet.logabsdet.item()

    return {
        "spectral_entropy": round(spectral_entropy, 6),
        "effective_rank": round(effective_rank, 4),
        "concentration": round(concentration, 6),
        "dominance_ratio": round(dominance_ratio, 4),
        "condition_number": round(condition_number, 4),
        "top5_singular_values": [round(v, 4) for v in top5],
        "num_singular_values": len(sigmas),
        "gram_trace": round(gram_trace, 4),
        "gram_logdet_sign": gram_logdet_sign,
        "gram_logdet": round(gram_logdet_val, 4),
    }


def compute_spectral_snapshot(past_key_values, step: int) -> dict:
    """Run SVD on V/recurrent matrices for all layers and heads.

    Qwen3.5 is a hybrid architecture:
      - KV layers (every 4th): DynamicLayer with .keys/.values
        V shape: (batch, num_kv_heads, seq_len, head_dim)
      - Linear attention layers: LinearAttentionLayer with .recurrent_states
        Recurrent shape: (batch, num_heads, state_dim, state_dim)
    """
    snapshot = {"step": step, "layers": []}

    num_layers = len(past_key_values.layers)
    for layer_idx in range(num_layers):
        layer = past_key_values.layers[layer_idx]
        is_kv = hasattr(layer, 'values') and hasattr(layer, 'is_initialized') and layer.is_initialized

        layer_metrics = {"layer": layer_idx, "type": "kv" if is_kv else "linear", "heads": []}

        if is_kv:
            V = layer.values.squeeze(0)  # (num_kv_heads, seq_len, head_dim)
            for head_idx in range(V.shape[0]):
                v_head = V[head_idx].float()  # (seq_len, head_dim)
                gram = v_head.T @ v_head
                sigmas = torch.linalg.svdvals(v_head)
                metrics = compute_spectral_metrics(sigmas, gram)
                metrics["head"] = head_idx
                layer_metrics["heads"].append(metrics)
        else:
            # Linear attention: SVD of recurrent state matrix per head
            rs = layer.recurrent_states  # (batch, num_heads, d, d)
            if rs is not None:
                rs = rs.squeeze(0)  # (num_heads, d, d)
                for head_idx in range(rs.shape[0]):
                    state = rs[head_idx].float()  # (d, d)
                    gram = state.T @ state
                    sigmas = torch.linalg.svdvals(state)
                    metrics = compute_spectral_metrics(sigmas, gram)
                    metrics["head"] = head_idx
                    layer_metrics["heads"].append(metrics)

        snapshot["layers"].append(layer_metrics)

    return snapshot


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(path: str, limit: int | None, offset: int) -> list[dict]:
    table = pq.read_table(path)
    cols = table.column_names

    # Auto-detect columns
    prompt_col = None
    for hint in ["problem", "prompt", "question", "input", "text"]:
        for c in cols:
            if hint in c.lower():
                prompt_col = c
                break
        if prompt_col:
            break

    ref_col = None
    for hint in ["answer", "response", "output", "target", "solution"]:
        for c in cols:
            if hint in c.lower():
                ref_col = c
                break
        if ref_col:
            break

    if prompt_col is None:
        print(f"{RED}Cannot find prompt column. Columns: {cols}{RESET}")
        sys.exit(1)

    end = min(offset + limit, table.num_rows) if limit else table.num_rows
    rows = []
    for i in range(offset, end):
        p = str(table.column(prompt_col)[i].as_py())
        r = ""
        if ref_col:
            val = table.column(ref_col)[i].as_py()
            if val is not None:
                r = str(val)
        rows.append({"prompt": p, "reference": r})

    print(f"{CYAN}Dataset: {os.path.basename(path)} ({table.num_rows} rows){RESET}")
    print(f"{CYAN}  prompt_col={prompt_col}  ref_col={ref_col}{RESET}")
    print(f"{CYAN}  using {len(rows)} rows (offset={offset}){RESET}")
    return rows


# ── Generation ───────────────────────────────────────────────────────────────

def generate_with_kv_monitoring(
    model, tokenizer, prompt: str, system: str,
    max_tokens: int = 1024, svd_interval: int = 32,
) -> tuple[str, int, list[dict]]:
    """Generate tokens with periodic KV cache SVD snapshots.

    Returns: (output_text, tokens_generated, spectral_snapshots)
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    past_key_values = None
    generated_ids = []
    spectral_log = []

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # Greedy sampling
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)
        input_ids = next_token

        # Check EOS
        if token_id == tokenizer.eos_token_id:
            break

        # Periodic SVD
        if (step + 1) % svd_interval == 0:
            snapshot = compute_spectral_snapshot(past_key_values, step + 1)
            spectral_log.append(snapshot)
            torch.cuda.empty_cache()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), spectral_log


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="KV Cache Spectral Health Monitor")
    p.add_argument("--dataset", "-d", type=str, required=True, help="Parquet dataset path")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--svd-interval", type=int, default=32,
                   help="Run SVD every N tokens (default 32)")
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--system", "-s", type=str, default=None)
    args = p.parse_args()

    if args.system is None:
        if "math" in os.path.basename(args.dataset).lower():
            args.system = MATH_SYSTEM
        else:
            args.system = "You are a helpful assistant."

    # ── Load model ──────────────────────────────────────────────────
    print(f"{CYAN}Loading {MODEL_NAME} (float16)...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"{CYAN}Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB{RESET}")

    # ── Load dataset ────────────────────────────────────────────────
    dataset = load_dataset(args.dataset, args.limit, args.offset)

    # ── Output path ─────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"kv_spectral_2b_{ts}.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # ── Run ─────────────────────────────────────────────────────────
    n_correct = 0
    n_scored = 0
    total_tokens = 0
    total_time = 0.0

    with open(out_path, "w") as f:
        for i, row in enumerate(dataset):
            print(f"\n{CYAN}[{i+1}/{len(dataset)}]{RESET} {row['prompt'][:80]}{'...' if len(row['prompt']) > 80 else ''}")

            t0 = time.monotonic()
            output, n_tokens, spectral_log = generate_with_kv_monitoring(
                model, tokenizer, row["prompt"], args.system,
                max_tokens=args.max_tokens, svd_interval=args.svd_interval,
            )
            elapsed = time.monotonic() - t0
            tps = n_tokens / elapsed if elapsed > 0 else 0

            extracted, correct = check_answer(output, row.get("reference", ""))
            if correct is not None:
                n_scored += 1
                if correct:
                    n_correct += 1

            total_tokens += n_tokens
            total_time += elapsed

            record = {
                "index": i,
                "prompt": row["prompt"],
                "reference": row.get("reference", ""),
                "output": output,
                "extracted_answer": extracted,
                "correct": correct,
                "tokens_generated": n_tokens,
                "wall_time_s": round(elapsed, 3),
                "tok_per_sec": round(tps, 1),
                "svd_interval": args.svd_interval,
                "spectral_snapshots": spectral_log,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            # Status
            mark = ""
            if correct is True:
                mark = f"{GREEN}✓{RESET}"
            elif correct is False:
                mark = f"{RED}✗{RESET} got={extracted} want={row.get('reference','')}"

            n_snapshots = len(spectral_log)
            print(f"  {DIM}{n_tokens} tok, {tps:.0f} tok/s, {n_snapshots} SVD snapshots{RESET}  {mark}")
            print(f"  {DIM}{output[:120]}{'...' if len(output) > 120 else ''}{RESET}")
            if n_scored > 0:
                print(f"  {DIM}running acc={n_correct}/{n_scored} ({100*n_correct/n_scored:.1f}%){RESET}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{DIM}{'═' * 78}{RESET}")
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"{BOLD}Done: {len(dataset)} samples, {total_tokens} tokens, "
          f"{total_time:.1f}s, {avg_tps:.1f} tok/s{RESET}")
    if n_scored > 0:
        print(f"{BOLD}Accuracy: {n_correct}/{n_scored} = {100*n_correct/n_scored:.1f}%{RESET}")
    print(f"{BOLD}Results: {out_path}{RESET}")


if __name__ == "__main__":
    main()
