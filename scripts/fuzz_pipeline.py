"""Randomised end-to-end fuzz of run_pipeline (usage: python scripts/fuzz_pipeline.py [repo_root] [n_cases] [start_seed]).
Every result must pass the independent validator, account for every carton exactly once, stay inside the
wall clearance, and report a fill rate that matches the placed volume."""
import os, sys, random, json, traceback
from pathlib import Path

os.environ["GA_WORKERS"] = "1"
if len(sys.argv) > 1:
    ROOT = sys.argv[1]
else:
    # Resolve repo root whether run from backend/scripts/ or scripts/
    cur = Path(__file__).resolve()
    if (cur.parents[1] / "data").exists():
        ROOT = str(cur.parents[1])
    else:
        ROOT = str(cur.parents[2])

sys.path.insert(0, ROOT + "/backend")
import numpy as np, pandas as pd
from app.solver.pipeline import run_pipeline
from app.core.models import RunOptions
from app.solver.validator import validate_solution
from app.solver.geometry import Dimensions

N = int(sys.argv[2]) if len(sys.argv) > 2 else 36; START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
cs = pd.read_csv(ROOT + "/data/container_spec.csv")
L, W, H, MW = [float(cs.iloc[0][k]) for k in ("Internal_Length_cm", "Internal_Width_cm", "Internal_Height_cm", "Max_Weight_kg")]
bad = []
for k in range(START, START + N):
    rng = random.Random(k)
    n_items = rng.randint(2, 7)
    lcl = rng.random() < 0.4
    items, rows = [], []
    for i in range(n_items):
        d = [rng.choice([rng.uniform(15, 60), rng.uniform(40, 140), rng.uniform(100, 230)]) for _ in range(3)]
        items.append(dict(Item_ID=f"F{k}-{i}", Description=f"fuzz {i}", Length_cm=round(d[0], 1), Width_cm=round(d[1], 1),
                          Height_cm=round(min(d[2], H - 20), 1), Weight_kg=round(rng.uniform(3, 140), 1),
                          This_Way_Up=rng.random() < 0.3))
        rows.append(dict(Item_ID=f"F{k}-{i}", PO_No=f"PO{k}", Customer_Code=(f"C{rng.randint(1,3)}" if lcl else ""),
                         Description=f"fuzz {i}", Qty_Pcs=1, Qty_Cartons=rng.randint(1, 40)))
    im, pl = pd.DataFrame(items), pd.DataFrame(rows)
    gap = rng.choice([0.0, 0.0, 1.0, 2.0]); wall = rng.choice([0.0, 0.0, 1.5, 2.0])
    opt = RunOptions(population_size=10, generations=10, tolerance_gap_cm=gap, container_wall_clearance_cm=wall, seed=k)
    tag = dict(k=k, gap=gap, wall=wall, lcl=lcl)
    try:
        res = run_pipeline(pl, im, cs, options=opt)
        r = res.result
        total = int(pl.Qty_Cartons.sum())
        problems = []
        if len(r.placed_boxes) + len(r.unplaced_cartons) != total:
            problems.append(f"accounting placed {len(r.placed_boxes)} + unplaced {len(r.unplaced_cartons)} != {total}")
        ids = [b.box_id for b in r.placed_boxes] + [u.box_id for u in r.unplaced_cartons]
        if len(set(ids)) != len(ids): problems.append("duplicate ids across placed/unplaced")
        rep = validate_solution(r.placed_boxes, Dimensions(L, W, H), MW, is_lcl=lcl)
        v = {a: b for a, b in rep.violations_by_type.items() if b and a != "lifo"}
        if v: problems.append(f"validator {v}")
        # geometry vs wall clearance
        if r.placed_boxes:
            minx = min(b.x for b in r.placed_boxes); miny = min(b.y for b in r.placed_boxes)
            maxx = max(b.x + b.actual_length for b in r.placed_boxes); maxy = max(b.y + b.actual_width for b in r.placed_boxes)
            if minx < wall - 1e-6 or miny < wall - 1e-6: problems.append(f"min x/y {minx:.2f},{miny:.2f} < wall {wall}")
            if maxx > L - wall + 1e-6 or maxy > W - wall + 1e-6: problems.append(f"max x/y {maxx:.2f},{maxy:.2f} > L-wall/W-wall")
        m = r.metrics
        pv = sum(b.actual_length * b.actual_width * b.actual_height for b in r.placed_boxes)
        if abs(m.fill_rate - pv / (L * W * H)) > 5e-4 and abs(m.fill_rate * 100 - pv / (L * W * H) * 100) > 0.06:
            problems.append(f"fill_rate {m.fill_rate} vs recomputed {pv/(L*W*H):.4f}")
        tw = sum(b.weight_kg for b in r.placed_boxes)
        if tw > MW + 1e-6: problems.append(f"weight {tw} > {MW}")
        if problems: bad.append((tag, problems))
    except Exception as e:
        bad.append((tag, ["EXC " + type(e).__name__ + ": " + str(e)[:200]]))
print(json.dumps(dict(n=N, failures=len(bad)), indent=0))
for t, p in bad: print(t, p)
