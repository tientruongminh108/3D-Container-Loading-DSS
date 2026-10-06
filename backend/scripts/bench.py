#!/usr/bin/env python
"""Verification protocol for solver changes (see PR notes, section 5).

Runs the default pipeline (group GA + dynamic grids + compaction) on

* the six real instances ``data/packing_list_0{1..6}.csv`` (+ item masters), and
* the committed held-out instances produced by ``scripts/gen_instances.py``,

for several seeds, and reports per instance:

    physical fill mean/min/max, % of the volume ceiling, placed/total,
    wall time, validator violations by type (must all be zero) and the real
    LIFO conflict count (soft metric, see validator.validate_solution).

PHYSICAL fill = sum(actual l*w*h of placed cartons) / NOMINAL internal container
volume (L*W*H from container_spec.csv) - never the shrunk "usable" volume.
Ceiling = total carton volume / nominal container volume.

Usage (from ``backend/``)::

    python scripts/bench.py                         # 10 instances, seeds 1 2 3, 30x40, gap 2.0
    python scripts/bench.py --instances 01 06 --pop 60 --gen 100 --seeds 1
    python scripts/bench.py --tag after_a1 --seeds 1 2 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR.parent / "data"
RESULTS_DIR = BACKEND_DIR / "benchmarks" / "results"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REAL_INSTANCES = ["01", "02", "03", "04", "05", "06"]


def _held_out_names() -> List[str]:
    from gen_instances import HELD_OUT

    return list(HELD_OUT)


def _load_instance(name: str):
    import pandas as pd

    container_df = pd.read_csv(DATA_DIR / "container_spec.csv")
    if name in REAL_INSTANCES:
        pack_df = pd.read_csv(DATA_DIR / f"packing_list_{name}.csv")
        im_file = DATA_DIR / f"item_master_{name}.csv"
        item_df = pd.read_csv(im_file if im_file.exists() else DATA_DIR / "item_master.csv")
    else:
        from gen_instances import HELD_OUT, generate_instance

        pack_df, item_df = generate_instance(HELD_OUT[name])
    for df in (pack_df, item_df, container_df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.replace("\r", "")
    return pack_df, item_df, container_df


def run_one(name: str, seed: int, pop: int, gen: int, gap: float, ga_workers: int) -> Dict[str, Any]:
    """Run one (instance, seed) and return a flat result row. Executed in a worker process."""
    os.environ["GA_WORKERS"] = str(ga_workers)  # before app.config is imported / cached
    from app.core.models import RunOptions
    from app.solver.geometry import Dimensions
    from app.solver.pipeline import run_pipeline
    from app.solver.validator import validate_solution

    pack_df, item_df, container_df = _load_instance(name)
    cL = float(container_df.iloc[0]["Internal_Length_cm"])
    cW = float(container_df.iloc[0]["Internal_Width_cm"])
    cH = float(container_df.iloc[0]["Internal_Height_cm"])
    max_w = float(container_df.iloc[0]["Max_Weight_kg"])
    nominal_volume = cL * cW * cH

    kw: Dict[str, Any] = dict(population_size=pop, generations=gen, tolerance_gap_cm=gap)
    if "seed" in RunOptions.model_fields:
        kw["seed"] = seed
    random.seed(seed)  # legacy global RNG (pre-B4 code paths)
    opts = RunOptions(**kw)

    t0 = time.perf_counter()
    res = run_pipeline(pack_df, item_df, container_df, options=opts)
    wall = time.perf_counter() - t0

    placed = res.result.placed_boxes
    is_lcl = res.is_lcl
    report = validate_solution(
        placed_boxes=placed,
        container_dims=Dimensions(cL, cW, cH),
        max_weight_kg=max_w,
        is_lcl=is_lcl,
        support_ratio_threshold=0.60,
    )

    placed_vol = sum(b.actual_length * b.actual_width * b.actual_height for b in placed)
    total_boxes = len(res.all_boxes)
    total_vol = sum(b.length_cm * b.width_cm * b.height_cm for b in res.all_boxes)
    total_wt = sum(b.weight_kg for b in res.all_boxes)
    return {
        "instance": name,
        "seed": seed,
        "fill_pct": 100.0 * placed_vol / nominal_volume,
        "ceiling_pct": 100.0 * min(total_vol, nominal_volume) / nominal_volume,
        "placed": len(placed),
        "total": total_boxes,
        "wall_s": wall,
        "weight_ratio": total_wt / max_w,
        "violations": dict(report.violations_by_type),
        "n_violations": sum(v for k, v in report.violations_by_type.items() if k != "lifo"),
        "is_valid": all(v == 0 for k, v in report.violations_by_type.items() if k != "lifo"),
        "lifo_conflicts": report.metrics.get("lifo_conflicts"),
        "lifo_blocked": report.metrics.get("lifo_blocked_cartons"),
        "lcl": bool(is_lcl),
        "budget": f"{pop}x{gen}",
        "gap": gap,
    }


def _summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    names = sorted({r["instance"] for r in rows}, key=lambda n: (n not in REAL_INSTANCES, n))
    for n in names:
        rs = [r for r in rows if r["instance"] == n]
        fills = [r["fill_pct"] for r in rs]
        mean = sum(fills) / len(fills)
        viol: Dict[str, int] = {}
        for r in rs:
            for k, v in r["violations"].items():
                if k != "lifo" and v:
                    viol[k] = viol.get(k, 0) + v
        lifo = [r["lifo_conflicts"] for r in rs if r["lifo_conflicts"] is not None]
        out.append({
            "instance": n,
            "n": len(rs),
            "mean": mean,
            "min": min(fills),
            "max": max(fills),
            "ceiling": rs[0]["ceiling_pct"],
            "pct_ceiling": 100.0 * mean / rs[0]["ceiling_pct"] if rs[0]["ceiling_pct"] else 0.0,
            "placed": sum(r["placed"] for r in rs) / len(rs),
            "total": rs[0]["total"],
            "wall": sum(r["wall_s"] for r in rs) / len(rs),
            "violations": viol,
            "lifo": (sum(lifo) / len(lifo)) if lifo else None,
            "lcl": rs[0]["lcl"],
        })
    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    summ = _summarise(rows)
    hdr = (f"{'inst':12s} {'n':>2s} {'fill mean':>9s} {'min':>6s} {'max':>6s} {'ceil':>6s} {'%ceil':>6s} "
           f"{'placed':>11s} {'wall s':>7s} {'violations':>12s} {'LIFO confl':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for s in summ:
        v = "0" if not s["violations"] else ",".join(f"{k}:{c}" for k, c in s["violations"].items())
        lifo = "n/a" if s["lifo"] is None else (f"{s['lifo']:.0f}" if s["lcl"] else "0 (FCL)")
        print(f"{s['instance']:12s} {s['n']:2d} {s['mean']:9.2f} {s['min']:6.1f} {s['max']:6.1f} "
              f"{s['ceiling']:6.1f} {s['pct_ceiling']:6.1f} {s['placed']:5.1f}/{s['total']:<5d} "
              f"{s['wall']:7.1f} {v:>12s} {lifo:>10s}")
    mean_all = sum(s["mean"] for s in summ) / len(summ)
    bad = sum(sum(s["violations"].values()) for s in summ)
    print("-" * len(hdr))
    print(f"mean over {len(summ)} instances: {mean_all:.2f}   total validator violations: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=["all"],
                    help="'all', 'real', 'held', or names (01..06, gen_*)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4, help="parallel (instance, seed) runs")
    ap.add_argument("--ga-workers", type=int, default=None,
                    help="process workers INSIDE one GA run (default 1 when --workers>1, else auto=0)")
    ap.add_argument("--tag", default=None, help="write benchmarks/results/<tag>.json")
    args = ap.parse_args()

    names: List[str] = []
    for tok in args.instances:
        if tok == "all":
            names += REAL_INSTANCES + _held_out_names()
        elif tok == "real":
            names += REAL_INSTANCES
        elif tok == "held":
            names += _held_out_names()
        else:
            names.append(tok)
    names = list(dict.fromkeys(names))
    ga_workers = args.ga_workers if args.ga_workers is not None else (1 if args.workers > 1 else 0)

    tasks: List[Tuple] = [(n, s, args.pop, args.gen, args.gap, ga_workers) for n in names for s in args.seeds]
    print(f"{len(tasks)} runs  budget {args.pop}x{args.gen}  gap {args.gap}  "
          f"workers {args.workers}  ga-workers {ga_workers}", flush=True)

    rows: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for t in tasks:
            r = run_one(*t)
            rows.append(r)
            print(f"  {r['instance']:12s} seed {r['seed']} fill {r['fill_pct']:.2f} "
                  f"{r['wall_s']:.1f}s valid={r['is_valid']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, *t): t for t in tasks}
            for f in as_completed(futs):
                r = f.result()
                rows.append(r)
                print(f"  {r['instance']:12s} seed {r['seed']} fill {r['fill_pct']:.2f} "
                      f"{r['wall_s']:.1f}s valid={r['is_valid']}", flush=True)

    print()
    print_table(rows)
    if args.tag:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{args.tag}.json"
        path.write_text(json.dumps(sorted(rows, key=lambda r: (r["instance"], r["seed"])), indent=1))
        print(f"wrote {path}")
    return 0 if all(r["is_valid"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
