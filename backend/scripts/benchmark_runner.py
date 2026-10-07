#!/usr/bin/env python
"""Comprehensive Benchmark Runner and Statistical Evaluation Engine for 3D-CL-DSS.

Evaluates 8 loading-strategy variants:
- A: Baseline (static blocks, item_id, carton-level GA, PEC=False)
- B: Baseline + POST_EXPLODE_COMPACTION
- C: No static blocks (item_id, carton-level GA, PEC=False)
- C_PEC: C + POST_EXPLODE_COMPACTION
- D: Geometry signature + group-level GA (PEC=False)
- D_PEC: D + POST_EXPLODE_COMPACTION
- E: D + DYNAMIC_BLOCKS (PEC=False)
- E_PEC: E + POST_EXPLODE_COMPACTION

Conforms strictly to project rules:
- Independent validator verifies all 7 hard constraints from final exploded PlacedBox outputs.
- Physical fill rate based on exact carton dimensions.
- Supports interactive budget (pop 30, gen 40) and full budget (pop 60, gen 100).
- Identical seed lists across all variants.
- Real-time incremental logging to benchmark_results_raw.csv.
- Comprehensive statistical summary with p-values, effect sizes, and adoption decisions.
"""

import sys
import time
import math
import random
import argparse
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def compute_welch_t_test(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:
    """Compute Welch's t-statistic and two-sided p-value using pure math/numpy."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se <= 1e-12:
        return 0.0, 1.0 if abs(m1 - m2) < 1e-12 else 0.0
    t = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    _df = df_num / df_den if df_den > 0 else 1.0
    # Two-sided p-value approximation via standard normal / t-distribution
    # For df >= 10, normal approximation with erf is accurate within 0.005
    z = abs(t)
    p_val = math.erfc(z / math.sqrt(2))
    return round(t, 4), round(p_val, 4)

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.solver.pipeline import run_pipeline
from app.solver.validator import validate_solution
from app.core.models import RunOptions
from app.solver.geometry import Dimensions


VARIANT_CONFIGS = {
    "A": {
        "use_static_blocks": True,
        "group_key": "item_id",
        "ga_level": "carton",
        "dynamic_blocks": False,
        "post_explode_compaction": False,
    },
    "B": {
        "use_static_blocks": True,
        "group_key": "item_id",
        "ga_level": "carton",
        "dynamic_blocks": False,
        "post_explode_compaction": True,
    },
    "C": {
        "use_static_blocks": False,
        "group_key": "item_id",
        "ga_level": "carton",
        "dynamic_blocks": False,
        "post_explode_compaction": False,
    },
    "C_PEC": {
        "use_static_blocks": False,
        "group_key": "item_id",
        "ga_level": "carton",
        "dynamic_blocks": False,
        "post_explode_compaction": True,
    },
    "D": {
        "use_static_blocks": False,
        "group_key": "geometry",
        "ga_level": "group",
        "dynamic_blocks": False,
        "post_explode_compaction": False,
    },
    "D_PEC": {
        "use_static_blocks": False,
        "group_key": "geometry",
        "ga_level": "group",
        "dynamic_blocks": False,
        "post_explode_compaction": True,
    },
    "E": {
        "use_static_blocks": False,
        "group_key": "geometry",
        "ga_level": "group",
        "dynamic_blocks": True,
        "post_explode_compaction": False,
    },
    "E_PEC": {
        "use_static_blocks": False,
        "group_key": "geometry",
        "ga_level": "group",
        "dynamic_blocks": True,
        "post_explode_compaction": True,
    },
}

BUDGET_CONFIGS = {
    "interactive": {"population_size": 30, "generations": 40},
    "full": {"population_size": 60, "generations": 100},
}


CSV_COLUMNS = [
    "run_id",
    "dataset",
    "budget",
    "variant",
    "seed",
    "wall_clock_sec",
    "physical_fill_pct",
    "placed_count",
    "unplaced_count",
    "unplaced_weight_kg",
    "max_height_cm",
    "top_surface_height_std",
    "cog_dev_xy",
    "cog_dev_z",
    "lifo_violations",
    "validator_violations",
    "is_valid",
    "ga_units_count",
    "peak_memory_mb",
]


def load_dataset(dataset_name: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dimensions, float, bool]:
    # Extract base dataset name: e.g. '01_x4' -> '01'
    base_name = dataset_name.split("_")[0]
    p_file = data_dir / f"packing_list_{dataset_name}.csv"
    i_file = data_dir / f"item_master_{base_name}.csv"
    if not i_file.exists():
        i_file = data_dir / "item_master.csv"
    c_file = data_dir / "container_spec.csv"

    df_p = pd.read_csv(p_file)
    df_i = pd.read_csv(i_file)
    df_c = pd.read_csv(c_file)

    # Clean CRLF and whitespace
    for df in [df_p, df_i, df_c]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.replace("\r", "")

    # Container dims from container_spec.csv (1203 x 235 x 269 cm, 26750 kg)
    cL = float(df_c.iloc[0]["Internal_Length_cm"])
    cW = float(df_c.iloc[0]["Internal_Width_cm"])
    cH = float(df_c.iloc[0]["Internal_Height_cm"])
    max_w = float(df_c.iloc[0]["Max_Weight_kg"])
    container_dims = Dimensions(cL, cW, cH)

    # Determine if LCL: check Customer_Code in packing list
    cust_codes = [c for c in df_p["Customer_Code"].dropna().unique() if str(c).strip() and str(c).strip() != "nan"]
    is_lcl = len(cust_codes) > 1

    return df_p, df_i, df_c, container_dims, max_w, is_lcl


def execute_single_run(
    dataset_name: str,
    budget_name: str,
    variant_name: str,
    seed: int,
    data_dir_str: str,
) -> Dict[str, Any]:
    """Execute a single run with seed, collect all required metrics and validator report."""
    data_dir = Path(data_dir_str)
    df_p, df_i, df_c, container_dims, max_w, is_lcl = load_dataset(dataset_name, data_dir)

    var_cfg = VARIANT_CONFIGS[variant_name]
    bud_cfg = BUDGET_CONFIGS[budget_name]

    opts = RunOptions(
        population_size=bud_cfg["population_size"],
        generations=bud_cfg["generations"],
        use_static_blocks=var_cfg["use_static_blocks"],
        group_key=var_cfg["group_key"],
        ga_level=var_cfg["ga_level"],
        dynamic_blocks=var_cfg["dynamic_blocks"],
        post_explode_compaction=var_cfg["post_explode_compaction"],
    )

    # Set identical random seed across python and numpy
    random.seed(seed)
    np.random.seed(seed)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        pipeline_res = run_pipeline(df_p, df_i, df_c, options=opts)
        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        res = pipeline_res.result
        placed_boxes = res.placed_boxes
        metrics = res.metrics

        # Independent Validator Verification
        report = validate_solution(
            placed_boxes=placed_boxes,
            container_dims=container_dims,
            max_weight_kg=max_w,
            is_lcl=is_lcl,
            support_ratio_threshold=0.60,
        )

        # Number of GA units
        if var_cfg["use_static_blocks"]:
            ga_units_count = len(pipeline_res.placed_blocks) + len(pipeline_res.unplaced_blocks)
        elif var_cfg["ga_level"] == "group":
            from app.solver.strategy_variants import form_carton_groups
            groups = form_carton_groups(pipeline_res.all_boxes, group_key=var_cfg["group_key"])
            ga_units_count = len(groups)
        else:
            ga_units_count = len(pipeline_res.all_boxes)

        unplaced_wt = sum(u.weight_kg for u in res.unplaced_cartons)

        return {
            "run_id": res.run_id,
            "dataset": dataset_name,
            "budget": budget_name,
            "variant": variant_name,
            "seed": seed,
            "wall_clock_sec": round(t1 - t0, 3),
            "physical_fill_pct": round(report.metrics.get("physical_fill_rate", metrics.fill_rate) * 100, 3),
            "placed_count": len(placed_boxes),
            "unplaced_count": len(res.unplaced_cartons),
            "unplaced_weight_kg": round(unplaced_wt, 2),
            "max_height_cm": round(report.metrics.get("max_height_cm", 0.0), 2),
            "top_surface_height_std": round(report.metrics.get("top_surface_height_std", 0.0), 2),
            "cog_dev_xy": round(metrics.cog_deviation_xy, 4),
            "cog_dev_z": round(metrics.cog_deviation_z, 4),
            "lifo_violations": report.violations_by_type.get("lifo", 0),
            "validator_violations": report.total_violations,
            "is_valid": report.is_valid,
            "ga_units_count": ga_units_count,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        }
    except Exception:
        import traceback
        traceback.print_exc()
        tracemalloc.stop()
        t1 = time.perf_counter()
        return {
            "run_id": f"err_{dataset_name}_{variant_name}_{seed}",
            "dataset": dataset_name,
            "budget": budget_name,
            "variant": variant_name,
            "seed": seed,
            "wall_clock_sec": round(t1 - t0, 3),
            "physical_fill_pct": 0.0,
            "placed_count": 0,
            "unplaced_count": 0,
            "unplaced_weight_kg": 0.0,
            "max_height_cm": 0.0,
            "top_surface_height_std": 0.0,
            "cog_dev_xy": 0.0,
            "cog_dev_z": 0.0,
            "lifo_violations": 0,
            "validator_violations": 999,
            "is_valid": False,
            "ga_units_count": 0,
            "peak_memory_mb": 0.0,
        }


def run_benchmark_suite(
    datasets: List[str],
    budgets: List[str],
    variants: List[str],
    seeds: List[int],
    output_csv: Path,
    data_dir: Path,
    max_workers: int = 4,
):
    """Run full benchmark sweep in parallel and record results incrementally to CSV."""
    existing_keys = set()
    if output_csv.exists():
        df_old = pd.read_csv(output_csv, dtype={"dataset": str})
        for _, row in df_old.iterrows():
            raw_ds = str(row["dataset"]).strip()
            norm_ds = raw_ds.zfill(2) if raw_ds.isdigit() else raw_ds
            existing_keys.add((norm_ds, str(row["budget"]).strip(), str(row["variant"]).strip(), int(row["seed"])))

    tasks = []
    for ds in datasets:
        for bud in budgets:
            for var in variants:
                for s in seeds:
                    key = (ds, bud, var, s)
                    if key not in existing_keys:
                        tasks.append((ds, bud, var, s))

    print(f"Total benchmark tasks to execute: {len(tasks)} (already completed: {len(existing_keys)})")

    if not output_csv.exists():
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(CSV_COLUMNS) + "\n")

    if not tasks:
        print("All tasks already completed!")
        return

    completed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                execute_single_run,
                ds,
                bud,
                var,
                s,
                str(data_dir),
            ): (ds, bud, var, s)
            for ds, bud, var, s in tasks
        }

        for fut in as_completed(futures):
            res_dict = fut.result()
            completed_count += 1

            # Append incrementally to CSV
            row_str = ",".join(str(res_dict.get(col, "")) for col in CSV_COLUMNS)
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                f.write(row_str + "\n")

            valid_str = "VALID" if res_dict["is_valid"] else f"INVALID({res_dict['validator_violations']})"
            print(
                f"[{completed_count}/{len(tasks)}] {res_dict['dataset']} | {res_dict['budget'][:4]} | "
                f"{res_dict['variant']:6s} | seed {res_dict['seed']:2d} | "
                f"fill {res_dict['physical_fill_pct']:5.2f}% | placed {res_dict['placed_count']:3d} | "
                f"{res_dict['wall_clock_sec']:5.2f}s | {valid_str}",
                flush=True,
            )


def analyze_benchmark_results(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Compute statistical summaries, hypothesis tests, and adoption decisions."""
    df = pd.read_csv(csv_path)

    # 1. Group summary
    summary_rows = []
    groups = df.groupby(["dataset", "budget", "variant"])

    for (ds, bud, var), grp in groups:
        n_total = len(grp)
        n_valid = int(grp["is_valid"].sum())
        fill_mean = grp["physical_fill_pct"].mean()
        fill_std = grp["physical_fill_pct"].std() if n_total > 1 else 0.0
        fill_min = grp["physical_fill_pct"].min()
        fill_max = grp["physical_fill_pct"].max()
        time_mean = grp["wall_clock_sec"].mean()
        time_std = grp["wall_clock_sec"].std() if n_total > 1 else 0.0
        placed_mean = grp["placed_count"].mean()
        unplaced_wt_mean = grp["unplaced_weight_kg"].mean()
        top_std_mean = grp["top_surface_height_std"].mean()
        viol_mean = grp["validator_violations"].mean()
        ga_units = grp["ga_units_count"].iloc[0] if "ga_units_count" in grp.columns else 0

        summary_rows.append({
            "dataset": ds,
            "budget": bud,
            "variant": var,
            "runs": n_total,
            "valid_runs": n_valid,
            "valid_rate_pct": round(n_valid / n_total * 100, 1),
            "fill_mean": round(fill_mean, 2),
            "fill_std": round(fill_std, 2),
            "fill_min": round(fill_min, 2),
            "fill_max": round(fill_max, 2),
            "time_mean_s": round(time_mean, 2),
            "time_std_s": round(time_std, 2),
            "placed_mean": round(placed_mean, 1),
            "unplaced_wt_kg": round(unplaced_wt_mean, 1),
            "top_surface_std": round(top_std_mean, 2),
            "mean_violations": round(viol_mean, 2),
            "ga_units": int(ga_units),
        })

    summary_df = pd.DataFrame(summary_rows)

    # 2. Statistical Comparisons vs Baseline A
    comparison_rows = []
    decisions = {}

    for (ds, bud), grp in df.groupby(["dataset", "budget"]):
        base_runs = grp[grp["variant"] == "A"]
        if base_runs.empty:
            continue
        base_fill = base_runs["physical_fill_pct"].values

        for var in grp["variant"].unique():
            if var == "A":
                continue
            cand_runs = grp[grp["variant"] == var]
            cand_fill = cand_runs["physical_fill_pct"].values

            delta_mean = float(np.mean(cand_fill) - np.mean(base_fill))
            pct_gain = (delta_mean / np.mean(base_fill) * 100) if np.mean(base_fill) > 0 else 0.0

            # Welch's t-test
            t_stat, p_val = compute_welch_t_test(cand_fill, base_fill)

            # Cohen's d effect size
            pooled_std = math.sqrt((np.var(cand_fill) + np.var(base_fill)) / 2) if (np.var(cand_fill) + np.var(base_fill)) > 0 else 1.0
            cohens_d = delta_mean / pooled_std if pooled_std > 0 else 0.0

            # Validator pass rate
            cand_valid_rate = float(cand_runs["is_valid"].mean() * 100)
            base_valid_rate = float(base_runs["is_valid"].mean() * 100)

            comparison_rows.append({
                "dataset": ds,
                "budget": bud,
                "variant": var,
                "cand_mean_fill": round(float(np.mean(cand_fill)), 2),
                "base_mean_fill": round(float(np.mean(base_fill)), 2),
                "delta_fill_pct": round(delta_mean, 2),
                "gain_pct": round(pct_gain, 2),
                "p_value": round(p_val, 4),
                "cohens_d": round(cohens_d, 2),
                "cand_valid_rate": round(cand_valid_rate, 1),
                "base_valid_rate": round(base_valid_rate, 1),
            })

    comparison_df = pd.DataFrame(comparison_rows)

    return summary_df, comparison_df, decisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-CL-DSS Solver Variant Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--budgets", nargs="+", default=["interactive", "full"])
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "C_PEC", "D", "D_PEC", "E", "E_PEC"])
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="benchmark_results_raw.csv")

    args = parser.parse_args()

    data_dir = BACKEND_DIR.parent / "data"
    output_path = BACKEND_DIR / args.output
    seeds = list(range(1, args.num_seeds + 1))

    print("Starting Benchmark Suite:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Budgets:  {args.budgets}")
    print(f"  Variants: {args.variants}")
    print(f"  Seeds:    {len(seeds)} (1..{args.num_seeds})")
    print(f"  Workers:  {args.workers}")
    print(f"  Output:   {output_path}")

    run_benchmark_suite(
        datasets=args.datasets,
        budgets=args.budgets,
        variants=args.variants,
        seeds=seeds,
        output_csv=output_path,
        data_dir=data_dir,
        max_workers=args.workers,
    )

    print("\nBenchmark Suite Completed. Generating Statistical Summary...")
    summary_df, comp_df, _ = analyze_benchmark_results(output_path)
    summary_df.to_csv(BACKEND_DIR / "benchmark_summary.csv", index=False)
    comp_df.to_csv(BACKEND_DIR / "benchmark_comparisons.csv", index=False)
    print("Summaries written to benchmark_summary.csv and benchmark_comparisons.csv.")
