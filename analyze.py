#!/usr/bin/env python3
"""Compare testbench result files: accuracy, speed, entropy stats."""

import json, sys, os, argparse
from pathlib import Path

def load_results(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    n = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    incorrect = sum(1 for r in records if r.get("correct") is False)
    skipped = n - correct - incorrect
    tps = [r["tok_per_sec"] for r in records if "tok_per_sec" in r]
    mean_h = [r["mean_H"] for r in records if "mean_H" in r]
    tokens = [r["tokens_generated"] for r in records if "tokens_generated" in r]
    return {
        "n": n,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "accuracy": correct / max(1, correct + incorrect) * 100,
        "avg_tps": sum(tps) / len(tps) if tps else 0,
        "avg_H": sum(mean_h) / len(mean_h) if mean_h else 0,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "controlled": records[0].get("controlled", False),
    }

def fmt(val, unit=""):
    if isinstance(val, float):
        return f"{val:.1f}{unit}"
    return f"{val}{unit}"

def print_table(summaries: list[tuple[str, dict]], md: bool = False):
    cols = ["File", "N", "Correct", "Accuracy", "Avg tok/s", "Avg H", "Avg tokens", "Control"]

    def make_row(name, s):
        tag = "YES" if s["controlled"] else "no"
        return [name, str(s["n"]),
                f"{s['correct']}/{s['correct'] + s['incorrect']}",
                fmt(s["accuracy"], "%"), fmt(s["avg_tps"]),
                fmt(s["avg_H"]), fmt(s["avg_tokens"]), tag]

    if md:
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join(["---"] * len(cols)) + " |")
        for name, s in summaries:
            print("| " + " | ".join(make_row(name, s)) + " |")
    else:
        widths = [max(40, max(len(name) for name, _ in summaries) + 2),
                  5, 9, 9, 10, 7, 11, 8]
        def row(vals):
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
        print()
        print(row(cols))
        print(row(["-" * w for w in widths]))
        for name, s in summaries:
            print(row(make_row(name, s)))
        print()

def print_comparison(summaries: list[tuple[str, dict]]):
    """Print pairwise accuracy deltas."""
    if len(summaries) < 2:
        return
    print("── Pairwise accuracy comparison ──")
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            a_name, a = summaries[i]
            b_name, b = summaries[j]
            delta = b["accuracy"] - a["accuracy"]
            sign = "+" if delta >= 0 else ""
            print(f"  {b_name} vs {a_name}: {sign}{delta:.1f}pp")
    print()

def main():
    p = argparse.ArgumentParser(description="Compare testbench result files")
    p.add_argument("files", nargs="*", help="Result JSONL files (default: all in testbench/results/)")
    p.add_argument("--sort", "-s", choices=["accuracy", "tps", "name", "H"], default="name")
    p.add_argument("--md", action="store_true", help="Output as markdown table")
    args = p.parse_args()

    if args.files:
        paths = args.files
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        paths = sorted(str(p) for p in Path(results_dir).glob("*.jsonl"))

    if not paths:
        print("No result files found.")
        sys.exit(1)

    summaries = []
    for path in paths:
        records = load_results(path)
        s = summarize(records)
        if s:
            name = os.path.basename(path).replace(".jsonl", "")
            summaries.append((name, s))

    sort_keys = {
        "accuracy": lambda x: x[1]["accuracy"],
        "tps": lambda x: x[1]["avg_tps"],
        "name": lambda x: x[0],
        "H": lambda x: x[1]["avg_H"],
    }
    summaries.sort(key=sort_keys[args.sort])

    print_table(summaries, md=args.md)
    # print_comparison(summaries)

if __name__ == "__main__":
    main()
