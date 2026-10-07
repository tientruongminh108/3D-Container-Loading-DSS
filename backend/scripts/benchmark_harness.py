#!/usr/bin/env python
"""Benchmark harness for 3D-CL-DSS solver evaluation across phases.

Conforms to PROMPT_coordinate_convention.md:
- Uses production app/config.py
- Evaluates LCL and FCL datasets
- Computes physical fill = sum(l * w * h of placed cartons) / container_volume
- Validates hard-constraint gate (0 errors): overlap, bounds, max weight, duplicate box_id
- Computes LIFO violations
- Reports mean, sd, min, max for >= 8 seeds
- Incremental JSONL logging for resilient detached execution
"""

import sys
import time
import math
import random
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings
from app.solver.pipeline import run_pipeline


def compute_physical_fill(placed_boxes: List[Any], container_dims: Any) -> float:
    """Compute physical fill = sum(carton actual l*w*h) / container volume."""
    vol = sum(b.actual_length * b.actual_width * b.actual_height for b in placed_boxes)
    c_vol = container_dims.length * container_dims.width * container_dims.height
    return vol / c_vol if c_vol > 0 else 0.0


def check_hard_constraints(placed_boxes: List[Any], container_dims: Any, max_weight: float) -> Tuple[int, List[str]]:
    """Check hard constraints:
    1. No two cartons overlap in 3D (tolerance 1e-4)
    2. Every carton inside container (tolerance 1e-4)
    3. Total weight <= max weight (tolerance 1e-4)
    4. No duplicate box_id
    Returns (error_count, list of error messages).
    """
    errors = []
    
    # 1. Duplicate box_id
    box_ids = [b.box_id for b in placed_boxes]
    if len(box_ids) != len(set(box_ids)):
        dup = len(box_ids) - len(set(box_ids))
        errors.append(f"Duplicate box_ids: {dup} duplicates found")

    # 2. Total weight
    total_weight = sum(b.weight_kg for b in placed_boxes)
    if total_weight > max_weight + 1e-4:
        errors.append(f"Weight limit exceeded: {total_weight:.2f} kg > {max_weight:.2f} kg")

    # 3. Inside container
    cL, cW, cH = container_dims.length, container_dims.width, container_dims.height
    for b in placed_boxes:
        bx2 = b.x + b.actual_length
        by2 = b.y + b.actual_width
        bz2 = b.z + b.actual_height
        if b.x < -1e-4 or b.y < -1e-4 or b.z < -1e-4 or bx2 > cL + 1e-4 or by2 > cW + 1e-4 or bz2 > cH + 1e-4:
            errors.append(
                f"Box {b.box_id} out of bounds: [{b.x:.1f}, {bx2:.1f}]x[{b.y:.1f}, {by2:.1f}]x[{b.z:.1f}, {bz2:.1f}] "
                f"vs container [{cL:.1f}, {cW:.1f}, {cH:.1f}]"
            )

    # 4. Overlaps
    n = len(placed_boxes)
    overlap_count = 0
    for i in range(n):
        b1 = placed_boxes[i]
        b1_x2, b1_y2, b1_z2 = b1.x + b1.actual_length, b1.y + b1.actual_width, b1.z + b1.actual_height
        for j in range(i + 1, n):
            b2 = placed_boxes[j]
            b2_x2, b2_y2, b2_z2 = b2.x + b2.actual_length, b2.y + b2.actual_width, b2.z + b2.actual_height
            
            dx = min(b1_x2, b2_x2) - max(b1.x, b2.x)
            dy = min(b1_y2, b2_y2) - max(b1.y, b2.y)
            dz = min(b1_z2, b2_z2) - max(b1.z, b2.z)
            if dx > 1e-4 and dy > 1e-4 and dz > 1e-4:
                overlap_count += 1
                if overlap_count <= 5:
                    errors.append(f"Overlap between {b1.box_id} and {b2.box_id}: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

    if overlap_count > 5:
        errors.append(f"Total overlapping pairs: {overlap_count}")

    return len(errors), errors


def count_lifo_violations(placed_boxes: List[Any], container_length: float, convention: str = "new") -> int:
    """Count LIFO violations between pairs of placed cartons.
    
    In the NEW convention:
      - Rear wall is x = 0, Door is x = L.
      - Cartons for earlier drop-off (smaller customer_sequence) are near the door (larger x).
      - Cartons for later drop-off (larger customer_sequence) are deeper (smaller x).
      - If a unloads first (a.seq < b.seq) and they overlap in Y-Z:
        Violation if b.max_x > a.min_x + 1e-6.
        
    In the OLD convention:
      - Door is x = 0, Rear wall is x = L.
      - If a unloads first (a.seq < b.seq) and they overlap in Y-Z:
        Violation if b.min_x < a.max_x - 1e-6.
        (Which is equivalent to mapping x_new = L - x_old - l and applying the new check).
    """
    n = len(placed_boxes)
    violations = 0
    for i in range(n):
        a = placed_boxes[i]
        seq_a = getattr(a, 'customer_sequence', 0)
        a_y2 = a.y + a.actual_width
        a_z2 = a.z + a.actual_height
        
        for j in range(n):
            if i == j:
                continue
            b = placed_boxes[j]
            seq_b = getattr(b, 'customer_sequence', 0)
            
            if seq_a >= seq_b:
                continue  # Only consider pairs where a unloads before b
                
            # Overlap in Y and Z
            b_y2 = b.y + b.actual_width
            b_z2 = b.z + b.actual_height
            overlap_y = min(a_y2, b_y2) - max(a.y, b.y) > 1e-4
            overlap_z = min(a_z2, b_z2) - max(a.z, b.z) > 1e-4
            
            if overlap_y and overlap_z:
                if convention == "new":
                    a_min_x = a.x
                    b_max_x = b.x + b.actual_length
                    if b_max_x > a_min_x + 1e-6:
                        violations += 1
                else:  # old convention
                    a_max_x = a.x + a.actual_length
                    b_min_x = b.x
                    if b_min_x < a_max_x - 1e-6:
                        violations += 1

    return violations


def run_single_seed(
    seed: int,
    dataset_type: str,
    convention: str = "new",
    wall_rule: str = "A",
    wall_weight: float = 1.0,
    wall_secondary: str = "ZY",
) -> Dict[str, Any]:
    """Execute a single run_pipeline call for a given seed and dataset."""
    root_data = BACKEND_DIR.parent / "data"
    
    if dataset_type == "LCL":
        packing_list_df = pd.read_csv(root_data / "packing_list.csv")
        item_master_df = pd.read_csv(root_data / "item_master.csv")
        container_df = pd.read_csv(root_data / "container_spec.csv")
    elif dataset_type == "FCL":
        packing_list_df = pd.read_csv(root_data / "packing_list_02.csv")
        item_master_df = pd.read_csv(root_data / "item_master.csv")
        container_df = pd.read_csv(root_data / "container_spec.csv")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Set seed
    random.seed(seed)
    
    settings = get_settings()
    settings.WALL_FIRST_RULE = wall_rule
    settings.WALL_FIRST_WEIGHT = wall_weight
    settings.WALL_FIRST_SECONDARY_ORDER = wall_secondary

    start_time = time.time()
    pipeline_res = run_pipeline(
        packing_list_df=packing_list_df,
        item_master_df=item_master_df,
        container_df=container_df,
        options=None,  # Use production settings from app/config.py
    )
    elapsed = time.time() - start_time
    
    run_res = pipeline_res.result
    placed_boxes = run_res.placed_boxes
    c_dims = pipeline_res.container_spec
    
    # Compute metrics
    from app.solver.geometry import Dimensions
    container_dims = Dimensions(c_dims.usable_length, c_dims.usable_width, c_dims.usable_height)
    
    phys_fill = compute_physical_fill(placed_boxes, container_dims)
    reported_fill = run_res.metrics.fill_rate
    placed_count = len(placed_boxes)
    total_count = run_res.metrics.total_cartons
    
    hard_error_count, hard_errors = check_hard_constraints(placed_boxes, container_dims, c_dims.max_weight_kg)
    lifo_violations = count_lifo_violations(placed_boxes, container_dims.length, convention=convention)
    
    return {
        "seed": seed,
        "dataset": dataset_type,
        "convention": convention,
        "wall_rule": wall_rule,
        "wall_weight": wall_weight,
        "wall_secondary": wall_secondary,
        "elapsed_sec": round(elapsed, 2),
        "physical_fill": round(phys_fill, 4),
        "reported_fill": round(reported_fill, 4),
        "placed_cartons": placed_count,
        "total_cartons": total_count,
        "lifo_violations": lifo_violations,
        "hard_errors": hard_error_count,
        "hard_error_msgs": hard_errors[:3],
    }


def stats(values: List[float]) -> Dict[str, float]:
    """Calculate mean, sd, min, max."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(variance)
    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def run_benchmark(
    phase: str,
    dataset: str,
    seeds: List[int],
    workers: int = 3,
    convention: str = "new",
    wall_rule: str = "A",
    wall_weight: float = 1.0,
    wall_secondary: str = "ZY",
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    print("\n=======================================================")
    print(f"RUNNING BENCHMARK: {phase} | Dataset: {dataset} | Convention: {convention}")
    print(f"Wall Rule: {wall_rule} (weight={wall_weight}, secondary={wall_secondary})")
    print(f"Seeds: {seeds} | Workers: {workers}")
    print("=======================================================\n")
    
    datasets = ["LCL", "FCL"] if dataset == "both" else [dataset]
    all_results = {}
    
    for ds in datasets:
        ds_results = []
        print(f"\n--- Starting {ds} (Seeds: {seeds}) ---")
        
        existing = {}
        if log_file and log_file.exists():
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if (
                                item.get("phase") == phase
                                and item.get("dataset") == ds
                                and item.get("wall_rule", "A") == wall_rule
                                and item.get("wall_weight", 1.0) == wall_weight
                                and item.get("wall_secondary", "ZY") == wall_secondary
                            ):
                                existing[item["seed"]] = item
                        except Exception:
                            pass

        for s in seeds:
            if s in existing:
                res = existing[s]
                ds_results.append(res)
                print(
                    f"[{ds} Seed {s:2d}] (Resumed) Fill: {res['physical_fill']*100:.2f}% | "
                    f"Placed: {res['placed_cartons']}/{res['total_cartons']} | "
                    f"LIFO Violations: {res['lifo_violations']:3d} | "
                    f"Hard Errors: {res['hard_errors']} | Time: {res['elapsed_sec']:.1f}s",
                    flush=True,
                )

        needed_seeds = [s for s in seeds if s not in existing]

        # Parallel execution using ProcessPoolExecutor
        if needed_seeds:
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_to_seed = {
                        executor.submit(run_single_seed, s, ds, convention, wall_rule, wall_weight, wall_secondary): s
                        for s in needed_seeds
                    }
                    for future in as_completed(future_to_seed):
                        s = future_to_seed[future]
                        try:
                            res = future.result()
                            ds_results.append(res)
                            print(
                                f"[{ds} Seed {s:2d}] Fill: {res['physical_fill']*100:.2f}% | "
                                f"Placed: {res['placed_cartons']}/{res['total_cartons']} | "
                                f"LIFO Violations: {res['lifo_violations']:3d} | "
                                f"Hard Errors: {res['hard_errors']} | Time: {res['elapsed_sec']:.1f}s",
                                flush=True,
                            )
                            if log_file:
                                with open(log_file, "a") as f:
                                    f.write(json.dumps({"phase": phase, **res}) + "\n")
                        except Exception as e:
                            print(f"[{ds} Seed {s:2d}] ERROR: {e}", flush=True)
            else:
                for s in needed_seeds:
                    res = run_single_seed(s, ds, convention, wall_rule, wall_weight, wall_secondary)
                    ds_results.append(res)
                    print(
                        f"[{ds} Seed {s:2d}] Fill: {res['physical_fill']*100:.2f}% | "
                        f"Placed: {res['placed_cartons']}/{res['total_cartons']} | "
                        f"LIFO Violations: {res['lifo_violations']:3d} | "
                        f"Hard Errors: {res['hard_errors']} | Time: {res['elapsed_sec']:.1f}s",
                        flush=True,
                    )
                    if log_file:
                        with open(log_file, "a") as f:
                            f.write(json.dumps({"phase": phase, **res}) + "\n")
                        
        ds_results.sort(key=lambda r: r["seed"])
        
        # Aggregate statistics
        fills = [r["physical_fill"] for r in ds_results]
        placed = [r["placed_cartons"] for r in ds_results]
        lifos = [r["lifo_violations"] for r in ds_results]
        hard_errs = [r["hard_errors"] for r in ds_results]
        times = [r["elapsed_sec"] for r in ds_results]
        
        summary = {
            "dataset": ds,
            "convention": convention,
            "wall_rule": wall_rule,
            "wall_weight": wall_weight,
            "wall_secondary": wall_secondary,
            "seed_count": len(ds_results),
            "physical_fill": stats(fills),
            "placed_cartons": stats(placed),
            "lifo_violations": stats(lifos),
            "hard_errors_total": sum(hard_errs),
            "runtime_sec": stats(times),
            "raw_runs": ds_results,
        }
        all_results[ds] = summary
        
        print(f"\n--- Summary for {ds} ({phase}, Rule {wall_rule}) ---")
        print(f"Physical Fill:   mean={summary['physical_fill']['mean']*100:.2f}%, sd={summary['physical_fill']['sd']*100:.2f}%, min={summary['physical_fill']['min']*100:.2f}%, max={summary['physical_fill']['max']*100:.2f}%")
        print(f"Placed Cartons:  mean={summary['placed_cartons']['mean']:.1f}, min={summary['placed_cartons']['min']:.0f}, max={summary['placed_cartons']['max']:.0f} (total: {ds_results[0]['total_cartons']})")
        print(f"LIFO Violations: mean={summary['lifo_violations']['mean']:.1f}, sd={summary['lifo_violations']['sd']:.1f}, min={summary['lifo_violations']['min']:.0f}, max={summary['lifo_violations']['max']:.0f}")
        print(f"Hard Constraint Gate: {summary['hard_errors_total']} errors across all runs")
        print(f"Runtime (sec):   mean={summary['runtime_sec']['mean']:.1f}s, min={summary['runtime_sec']['min']:.1f}s, max={summary['runtime_sec']['max']:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="3D-CL-DSS Solver Benchmark Harness")
    parser.add_argument("--phase", type=str, default="phase0_baseline", help="Phase name")
    parser.add_argument("--dataset", choices=["LCL", "FCL", "both"], default="both", help="Dataset to run")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8", help="Comma-separated seed list")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel worker processes")
    parser.add_argument("--convention", choices=["old", "new"], default="new", help="Coordinate convention for LIFO check")
    parser.add_argument("--wall-rule", choices=["A", "B", "C"], default="A", help="Wall-first placement rule (A, B, C)")
    parser.add_argument("--wall-weight", type=float, default=1.0, help="Weight w for Rule C")
    parser.add_argument("--wall-secondary", choices=["ZY", "YZ"], default="ZY", help="Secondary tie-break inside wall")
    parser.add_argument("--output", type=str, default="benchmark_results.jsonl", help="JSONL output file")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    log_path = BACKEND_DIR / args.output
    
    run_benchmark(
        phase=args.phase,
        dataset=args.dataset,
        seeds=seeds,
        workers=args.workers,
        convention=args.convention,
        wall_rule=args.wall_rule,
        wall_weight=args.wall_weight,
        wall_secondary=args.wall_secondary,
        log_file=log_path,
    )


if __name__ == "__main__":
    main()
