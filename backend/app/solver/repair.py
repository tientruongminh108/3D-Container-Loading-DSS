"""Post-compaction repair: swap an unplaced unit for a placed one that is in its way.

``rescan_and_insert`` only tries to drop an unplaced unit into free extreme points. When the
remaining free space is unusable for it (wrong shape / support / weight rule) nothing more happens.
This pass tries a *remove-and-reinsert* move: take a placed unit P that nothing rests on, remove it,
try to put the unplaced unit U into the freed region, then try to put P back anywhere. The move is
accepted only if the total placed volume strictly increases, and every placement goes through the
same ``find_best_placement`` constraints as the main decoder.
"""
import logging
import time
from typing import Any, List, Optional, Tuple

from app.config import get_settings
from app.solver.compaction import _check_support_for_unit, _unit_volume as _env_volume
from app.solver.geometry import (
    BoundingBox,
    Dimensions,
    FLOOR_EPSILON,
    Posture,
    generate_extreme_points,
    sort_extreme_points,
)
from app.solver.placement import find_best_placement


logger = logging.getLogger(__name__)


def _unit_volume(u: Any) -> float:
    """Real carton volume: a block's contents, not its (possibly looser) envelope."""
    contents = getattr(u, "contents", None)
    if contents:
        return float(sum(c.length_cm * c.width_cm * c.height_cm for c in contents))
    return _env_volume(u)


def _carries_nothing(i: int, bboxes: List[BoundingBox]) -> bool:
    b = bboxes[i]
    for j, o in enumerate(bboxes):
        if j != i and abs(o.min_z - b.max_z) < 1e-4 and b.contact_area(o) > 1e-4:
            return False
    return True


def _new_rests_ok(idx: int, bboxes: List[BoundingBox], data: List[Any], postures: List[Posture],
                  min_support_ratio: float) -> bool:
    """Boxes that end up resting on the newly added box ``idx`` must satisfy support + weight rule."""
    nb = bboxes[idx]
    for j, o in enumerate(bboxes):
        if j != idx and o.min_z > FLOOR_EPSILON and abs(o.min_z - nb.max_z) < 1e-4 and nb.contact_area(o) > 1e-4:
            if not _check_support_for_unit(j, o, bboxes, min_support_ratio, data, postures):
                return False
    return True


def _try_place(unit, bboxes, data, postures, dims, weight, max_weight, is_lcl, last_seq, eps, min_support_ratio):
    res, _ = find_best_placement(
        box=unit, placed_boxes=bboxes, placed_boxes_data=data, container_dims=dims,
        current_weight=weight, max_weight=max_weight, is_lcl=is_lcl,
        extreme_points=eps, last_customer_sequence=last_seq, placed_postures=postures,
    )
    if not res:
        return None
    nb = BoundingBox.from_position_and_dims(res.position, res.dims)
    bboxes.append(nb)
    data.append(unit)
    postures.append(res.posture)
    if not _new_rests_ok(len(bboxes) - 1, bboxes, data, postures, min_support_ratio):
        bboxes.pop(); data.pop(); postures.pop()
        return None
    return nb


def _tower(i: int, bboxes: List[BoundingBox], max_size: int) -> Optional[List[int]]:
    """Indices of box ``i`` plus everything (transitively) resting on it; None if larger than max_size."""
    out, stack = [i], [i]
    seen = {i}
    while stack:
        k = stack.pop()
        b = bboxes[k]
        for j, o in enumerate(bboxes):
            if j not in seen and abs(o.min_z - b.max_z) < 1e-4 and b.contact_area(o) > 1e-4:
                seen.add(j)
                out.append(j)
                stack.append(j)
                if len(out) > max_size:
                    return None
    return out


def repair_swap(
    placed_bboxes: List[BoundingBox],
    placed_data: List[Any],
    placed_postures: List[Posture],
    unplaced: List[Tuple[Any, str]],
    container_dims: Dimensions,
    current_weight: float,
    max_weight: float,
    is_lcl: bool = False,
    min_support_ratio: float = 0.6,
    max_trials: Optional[int] = None,
    max_candidates: Optional[int] = None,
):
    """Returns (bboxes, data, postures, unplaced, weight, n_swaps).

    Move: remove a placed unit P together with the tower resting on it, put the unplaced unit U into
    the freed region, then re-place the tower members one by one (bottom-up). Accepted only when the
    total placed volume strictly increases. The work bound is a TRIAL COUNT, not wall-clock time,
    so the same seed always gives the same layout on any machine.
    """
    from collections import Counter

    cfg = get_settings()
    max_trials = cfg.REPAIR_MAX_TRIALS if max_trials is None else max_trials
    max_candidates = cfg.REPAIR_MAX_CANDIDATES if max_candidates is None else max_candidates
    max_tower = cfg.REPAIR_MAX_TOWER
    t0 = time.perf_counter()  # diagnostics only: never used in a decision (keeps runs reproducible)
    trials = 0
    bboxes, data, postures = list(placed_bboxes), list(placed_data), list(placed_postures)
    pending = list(unplaced)
    weight = current_weight
    n_swaps = 0
    stats = Counter()
    last_seq = max((getattr(u, "customer_sequence", 0) for u in data), default=0)

    for U, reason in sorted(list(pending), key=lambda p: -_unit_volume(p[0])):
        if trials >= max_trials:
            break
        uvol = _unit_volume(U)
        uw = getattr(U, "weight_kg", 0.0)
        margin = max(getattr(U, "length_cm", 0.0), getattr(U, "width_cm", 0.0), getattr(U, "height_cm", 0.0)) or 100.0
        towers = []
        for i in range(len(bboxes)):
            t = _tower(i, bboxes, max_tower)
            if t is None:
                continue
            tv = sum(_unit_volume(data[k]) for k in t)
            towers.append((abs(tv - uvol), i, t, tv))
        towers.sort(key=lambda x: x[0])
        for _, i, T, tvol in towers[:max_candidates]:
            if trials >= max_trials:
                break
            trials += 1
            Tset = set(T)
            region = (min(bboxes[k].min_x for k in T), max(bboxes[k].max_x for k in T),
                      min(bboxes[k].min_y for k in T), max(bboxes[k].max_y for k in T),
                      min(bboxes[k].min_z for k in T), max(bboxes[k].max_z for k in T))
            keep = [k for k in range(len(bboxes)) if k not in Tset]
            tb = [bboxes[k] for k in keep]
            td = [data[k] for k in keep]
            tp = [postures[k] for k in keep]
            tw = weight - sum(getattr(data[k], "weight_kg", 0.0) for k in T)
            local = [e for e in generate_extreme_points(tb, container_dims)
                     if region[0] - margin <= e.x <= region[1] and region[2] - margin <= e.y <= region[3]
                     and region[4] - margin <= e.z <= region[5]]
            if not local:
                stats["no_local"] += 1
                continue
            if _try_place(U, tb, td, tp, container_dims, tw, max_weight, is_lcl, last_seq,
                          sort_extreme_points(local), min_support_ratio) is None:
                stats["u_fail"] += 1
                continue
            stats["u_ok"] += 1
            tw += uw
            back, lost = 0.0, []
            for k in sorted(T, key=lambda k: (bboxes[k].min_z, -getattr(data[k], "weight_kg", 0.0))):
                eps = sort_extreme_points(generate_extreme_points(tb, container_dims))
                if _try_place(data[k], tb, td, tp, container_dims, tw, max_weight, is_lcl, last_seq,
                              eps, min_support_ratio) is not None:
                    tw += getattr(data[k], "weight_kg", 0.0)
                    back += _unit_volume(data[k])
                else:
                    lost.append(data[k])
            if uvol + back <= tvol + 1e-6:
                stats["no_gain"] += 1
                continue
            bboxes, data, postures, weight = tb, td, tp, tw
            pending = [p for p in pending if p[0] is not U] + [(x, "swapped_out") for x in lost]
            n_swaps += 1
            stats["swap"] += 1
            break
    logger.debug("repair stats %s trials %d elapsed %.1fs", dict(stats), trials, time.perf_counter() - t0)
    return bboxes, data, postures, pending, weight, n_swaps
