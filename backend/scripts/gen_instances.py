"""Deterministic generator for held-out benchmark instances.

The six real instances (data/packing_list_0{1..6}.csv) are what the solver was
tuned against, so every behavioural change must also be confirmed on instances
that were never used for tuning.  The instances below are produced with a fixed
seed per instance (``random.Random(seed)``, no global state) so they are
bit-identical on every machine.

Each generator returns ``(packing_list_df, item_master_df)`` using exactly the
column layout of the real CSV files.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

# Nominal container used by data/container_spec.csv (40HC).
CONTAINER_L, CONTAINER_W, CONTAINER_H, CONTAINER_MAX_KG = 1203.0, 235.0, 269.0, 26750.0
CONTAINER_VOLUME = CONTAINER_L * CONTAINER_W * CONTAINER_H


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    seed: int
    n_skus: int
    this_way_up_ratio: float  # fraction of SKUs flagged This_Way_Up
    n_customers: int = 1  # >1 => LCL
    target_volume_ratio: float = 0.90  # total carton volume / container volume
    target_weight_ratio: float = 0.0  # >0 => scale density so total kg ~ ratio * max weight
    density_kg_m3: Tuple[float, float] = (60.0, 160.0)
    description: str = ""


# Fixed, committed held-out set (seeds never changed).
HELD_OUT: Dict[str, InstanceSpec] = {
    spec.name: spec
    for spec in [
        InstanceSpec("gen_free", 1101, n_skus=18, this_way_up_ratio=0.0, target_volume_ratio=0.90,
                     description="18 SKUs, all rotatable"),
        InstanceSpec("gen_upright", 1102, n_skus=9, this_way_up_ratio=1.0, target_volume_ratio=0.89,
                     description="9 SKUs, all This_Way_Up"),
        InstanceSpec("gen_lcl4", 1103, n_skus=15, this_way_up_ratio=0.4, n_customers=4,
                     target_volume_ratio=0.88, description="LCL, 4 stops, 15 SKUs, 40% upright"),
        InstanceSpec("gen_heavy", 1104, n_skus=12, this_way_up_ratio=0.3, target_volume_ratio=0.90,
                     target_weight_ratio=1.25, density_kg_m3=(250.0, 700.0),
                     description="weight-bound: total kg ~125% of container limit"),
    ]
}


def _random_dims(rng: random.Random) -> Tuple[float, float, float]:
    """Furniture/appliance-like carton sizes (cm), always fitting the container."""
    kind = rng.random()
    if kind < 0.30:  # small boxes
        dims = (rng.uniform(25, 70), rng.uniform(20, 55), rng.uniform(15, 50))
    elif kind < 0.70:  # medium
        dims = (rng.uniform(50, 120), rng.uniform(35, 90), rng.uniform(20, 90))
    else:  # long / flat-pack
        dims = (rng.uniform(110, 220), rng.uniform(25, 95), rng.uniform(12, 80))
    return tuple(round(d, 1) for d in dims)  # type: ignore[return-value]


def generate_instance(spec: InstanceSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(spec.seed)

    skus: List[dict] = []
    for i in range(spec.n_skus):
        l, w, h = _random_dims(rng)
        vol = l * w * h
        dens = rng.uniform(*spec.density_kg_m3)  # kg per m3
        weight = max(0.5, round(vol / 1_000_000.0 * dens * rng.uniform(0.7, 1.3), 1))
        skus.append(dict(
            Item_ID=f"G{spec.seed % 100:02d}-{i + 1:03d}",
            Description=f"Generated SKU {i + 1}",
            Length_cm=l, Width_cm=w, Height_cm=h, Weight_kg=weight,
            This_Way_Up=(i < round(spec.this_way_up_ratio * spec.n_skus)),
            _vol=vol,
        ))
    rng.shuffle(skus)  # decouple SKU order from the This_Way_Up flag

    # Relative popularity -> carton quantity, scaled to the target total volume.
    pop = [rng.uniform(0.3, 3.0) for _ in skus]
    target_vol = spec.target_volume_ratio * CONTAINER_VOLUME
    scale = target_vol / sum(p * s["_vol"] for p, s in zip(pop, skus))
    qty = [max(1, round(p * scale)) for p in pop]

    if spec.target_weight_ratio > 0:
        tot_w = sum(q * s["Weight_kg"] for q, s in zip(qty, skus))
        k = spec.target_weight_ratio * CONTAINER_MAX_KG / tot_w
        for s in skus:
            s["Weight_kg"] = round(max(0.5, s["Weight_kg"] * k), 1)

    po_pool = [f"PO-{spec.seed % 100:02d}{j:03d}" for j in range(1, 8)]
    customers = [f"CUST-{chr(ord('A') + c)}" for c in range(spec.n_customers)]
    rows = []
    for s, q in zip(skus, qty):
        # split a SKU over up to two customers in LCL so geometry groups span stops
        owners = [rng.choice(customers)]
        if spec.n_customers > 1 and q >= 4 and rng.random() < 0.35:
            owners.append(rng.choice(customers))
        for o_idx, owner in enumerate(owners):
            part = q // len(owners) if o_idx < len(owners) - 1 else q - (q // len(owners)) * (len(owners) - 1)
            if part <= 0:
                continue
            rows.append(dict(
                Item_ID=s["Item_ID"], PO_No=rng.choice(po_pool),
                Customer_Code=owner if spec.n_customers > 1 else "CUST-A",
                Description=s["Description"], Qty_Pcs=part, Qty_Cartons=part,
            ))

    # LCL: customers appear in stop order (first appearance = customer_sequence)
    if spec.n_customers > 1:
        rows.sort(key=lambda r: r["Customer_Code"])

    item_df = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in skus])
    item_df["This_Way_Up"] = item_df["This_Way_Up"].map(lambda b: "TRUE" if b else "FALSE")
    pack_df = pd.DataFrame(rows)
    return pack_df, item_df


if __name__ == "__main__":
    for nm, sp in HELD_OUT.items():
        p, it = generate_instance(sp)
        vol = sum(r.Length_cm * r.Width_cm * r.Height_cm * 1 for r in it.itertuples())
        n = int(p["Qty_Cartons"].sum())
        print(f"{nm:12s} skus={len(it):2d} cartons={n:4d} customers={p['Customer_Code'].nunique()}")
