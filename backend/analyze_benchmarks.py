#!/usr/bin/env python
"""Analysis script for paired A/B benchmark evaluation.

Reads benchmark results from JSONL files, performs paired comparisons
(per seed), computes paired t-statistics and p-values, and validates
the hard-constraint and LIFO zero-violation gates.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple


def t_distribution_cdf(t: float, df: int) -> float:
    """Approximate two-tailed p-value for Student's t distribution with df degrees of freedom."""
    # Using incomplete beta function approximation or standard numerical integration
    # For df=7 (8 pairs), Simpson's rule numerical integration is exact and fast.
    if df <= 0 or math.isnan(t):
        return 1.0
    
    # PDF of Student's t
    def pdf(x):
        coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        return coeff * (1 + (x * x) / df) ** (-(df + 1) / 2)

    # Integrate from 0 to |t|
    limit = abs(t)
    n_steps = 1000
    h = limit / n_steps
    integral = 0.0
    for i in range(n_steps + 1):
        x = i * h
        w = 1 if (i == 0 or i == n_steps) else (4 if i % 2 == 1 else 2)
        integral += w * pdf(x)
    integral *= (h / 3.0)

    # Two-tailed p-value = 2 * (0.5 - P(0 <= X <= |t|)) = 1 - 2 * integral
    p_val = max(0.0, min(1.0, 1.0 - 2.0 * integral))
    return p_val


def paired_t_test(deltas: List[float]) -> Tuple[float, float, float, float]:
    """Compute (mean_delta, sd_delta, t_stat, p_val) for a list of paired deltas."""
    n = len(deltas)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    sd_d = math.sqrt(var_d)
    se_d = sd_d / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 1e-12 else 0.0
    p_val = t_distribution_cdf(t_stat, df=n - 1)
    return mean_d, sd_d, t_stat, p_val


def load_runs(jsonl_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    runs = {}
    if not jsonl_path.exists():
        return runs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            # key: (dataset, seed)
            key = (entry["dataset"], entry["seed"])
            runs[key] = entry
    return runs


def compare_phases(baseline_file: str, test_file: str, test_phase_name: str = "Test Phase"):
    base_runs = load_runs(Path(baseline_file))
    test_runs = load_runs(Path(test_file))

    datasets = sorted(list(set(k[0] for k in test_runs.keys())))
    if not datasets:
        print(f"No runs found in {test_file}")
        return

    print("=" * 80)
    print(f"PAIRED A/B COMPARISON: Baseline ({baseline_file}) vs {test_phase_name} ({test_file})")
    print("=" * 80)

    for ds in datasets:
        ds_keys = sorted([k for k in test_runs.keys() if k[0] == ds], key=lambda k: k[1])
        paired_keys = [k for k in ds_keys if k in base_runs]

        print(f"\n-------------------------------------------------------------")
        print(f"Dataset: {ds} | Paired Seeds: {len(paired_keys)}")
        print(f"-------------------------------------------------------------")
        print(f"{'Seed':>4} | {'Base Fill':>9} -> {'New Fill':>9} ({'Delta':>7}) | {'Base Placed':>11} -> {'New Placed':>11} | {'LIFO Base':>9} -> {'New':>4} | {'Hard Err':>8}")
        print("-" * 80)

        delta_fills = []
        delta_placed = []
        total_lifo_base = 0
        total_lifo_new = 0
        total_hard_errs = 0

        for k in paired_keys:
            b = base_runs[k]
            t = test_runs[k]
            s = k[1]

            d_fill = (t["physical_fill"] - b["physical_fill"]) * 100.0
            d_plc = t["placed_cartons"] - b["placed_cartons"]

            delta_fills.append(d_fill)
            delta_placed.append(float(d_plc))

            total_lifo_base += b.get("lifo_violations", 0)
            total_lifo_new += t.get("lifo_violations", 0)
            total_hard_errs += t.get("hard_errors", 0)

            print(
                f"{s:4d} | {b['physical_fill']*100:8.2f}% -> {t['physical_fill']*100:8.2f}% ({d_fill:+6.2f}%) | "
                f"{b['placed_cartons']:5d}/{b['total_cartons']:<5d} -> {t['placed_cartons']:5d}/{t['total_cartons']:<5d} | "
                f"{b.get('lifo_violations', 0):9d} -> {t.get('lifo_violations', 0):4d} | "
                f"{t.get('hard_errors', 0):8d}"
            )

        print("-" * 80)

        # Statistics
        if paired_keys:
            mean_df, sd_df, t_fill, p_fill = paired_t_test(delta_fills)
            mean_dp, sd_dp, t_plc, p_plc = paired_t_test(delta_placed)

            print(f"Summary for {ds}:")
            print(f"  Physical Fill Delta: mean = {mean_df:+.2f}%, sd = {sd_df:.2f}% | t = {t_fill:.3f}, p = {p_fill:.4f} "
                  f"{'(*stat. sig. p<0.05)' if p_fill < 0.05 else '(not stat. sig.)'}")
            print(f"  Placed Cartons Delta: mean = {mean_dp:+.2f}, sd = {sd_dp:.2f} | t = {t_plc:.3f}, p = {p_plc:.4f}")
            print(f"  LIFO Violations: baseline total = {total_lifo_base} -> new total = {total_lifo_new} "
                  f"({'GATE PASSED (0 violations)' if total_lifo_new == 0 else 'GATE FAILED'})")
            print(f"  Hard Constraint Errors: {total_hard_errs} "
                  f"({'GATE PASSED (0 errors)' if total_hard_errs == 0 else 'GATE FAILED'})")


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_benchmarks.py <baseline_jsonl> <test_jsonl> [phase_name]")
        return
    base_file = sys.argv[1]
    test_file = sys.argv[2]
    phase_name = sys.argv[3] if len(sys.argv) > 3 else "Test Phase"
    compare_phases(base_file, test_file, phase_name)


if __name__ == "__main__":
    main()
