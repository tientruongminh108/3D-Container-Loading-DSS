from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from app.config import get_settings
from app.solver.geometry import BoundingBox, calculate_cog, check_cog_balance, FLOOR_EPSILON, unit_weight
from app.solver.parsing import Box


def _actual_volume(unit) -> float:
    """Actual l*w*h of a carton (or static block) in cm^3."""
    if hasattr(unit, "length_cm") and hasattr(unit, "width_cm") and hasattr(unit, "height_cm"):
        return float(unit.length_cm * unit.width_cm * unit.height_cm)
    if hasattr(unit, "volume") and callable(unit.volume):
        return float(unit.volume())
    return 0.0


@dataclass
class FitnessResult:
    fitness: float
    placed_volume: float
    unplaced_count: int
    cog_deviation_xy: float
    cog_deviation_z: float
    stacking_violations: int
    stability_violations: int
    frag_penalty: float = 0.0
    unplaced_volume: float = 0.0


def calculate_fitness(
    placed_bboxes: List[BoundingBox],
    placed_data: List[Box],
    unplaced: List[Tuple[Box, str]],
    container_dims,
    max_weight: float,
) -> FitnessResult:
    settings = get_settings()

    placed_volume = sum(
        (b.max_x - b.min_x) * (b.max_y - b.min_y) * (b.max_z - b.min_z)
        for b in placed_bboxes
    )
    container_volume = container_dims.length * container_dims.width * container_dims.height
    fill_rate = placed_volume / container_volume if container_volume > 0 else 0

    unplaced_count = len(unplaced)

    # Center of gravity calculation (normalized per Section 5.3.2, 5.4.3)
    weights = [b.weight_kg for b in placed_data]
    cog = calculate_cog(placed_bboxes, weights)
    
    # Normalized CoG deviations per Section 5.3.2
    ideal_x = container_dims.length / 2
    ideal_y = container_dims.width / 2
    ideal_z = container_dims.height / 2
    
    B1_norm = abs(cog.x - ideal_x) / (container_dims.length / 2) if container_dims.length > 0 else 0
    B2_norm = abs(cog.y - ideal_y) / (container_dims.width / 2) if container_dims.width > 0 else 0
    B3_norm = max(0, cog.z - ideal_z) / (container_dims.height / 2) if container_dims.height > 0 else 0
    
    # Legacy raw deviations for reporting
    _, cog_dev_xy, cog_dev_z = check_cog_balance(
        cog, container_dims, settings.COG_TOLERANCE_XY, settings.COG_TOLERANCE_Z
    )

    # Check constraints for penalty terms (B4, B5)
    stacking_violations = 0
    stability_violations = 0

    # Build a z-level index so supporters are looked up in O(1)
    # instead of scanning all N boxes for each of N candidates.
    z_to_supporters = defaultdict(list)
    for j, other in enumerate(placed_bboxes):
        # Key on rounded max_z to handle floating-point near-equality
        z_to_supporters[round(other.max_z, 6)].append((j, other))

    for i, bbox in enumerate(placed_bboxes):
        if bbox.min_z > FLOOR_EPSILON:
            support_area = 0.0
            footprint = (bbox.max_x - bbox.min_x) * (bbox.max_y - bbox.min_y)
            box_unit_wt = unit_weight(placed_data[i])
            for j, other in z_to_supporters.get(round(bbox.min_z, 6), []):
                if other.supports(bbox):
                    support_area += other.contact_area(bbox)
                    sup_unit_wt = unit_weight(placed_data[j])
                    if box_unit_wt > sup_unit_wt + 1e-3:
                        stacking_violations += 1
            if footprint > 0 and (support_area / footprint) < settings.SUPPORT_RATIO:
                stability_violations += 1

    # Fitness (all terms normalised so that the volume metric dominates):
    #   fitness = fill_rate
    #             - w_cog  * (B1 + B2 + B3)
    #             - w_frag * frag_penalty
    #             - UNPLACED_VOLUME_WEIGHT * (unplaced_volume / container_volume)
    #             - UNPLACED_COUNT_WEIGHT  * (unplaced_count / (placed_count + unplaced_count))
    #             - INFEASIBLE_PENALTY          if stacking / stability violations
    # unplaced_volume uses the ACTUAL l*w*h of the unplaced cartons.
    unplaced_volume = sum(_actual_volume(u) for u, _ in unplaced)
    total_count = len(placed_data) + unplaced_count
    unplaced_volume_frac = unplaced_volume / container_volume if container_volume > 0 else 0.0
    unplaced_count_frac = unplaced_count / total_count if total_count > 0 else 0.0
    unplaced_penalty = (
        settings.UNPLACED_VOLUME_WEIGHT * unplaced_volume_frac
        + settings.UNPLACED_COUNT_WEIGHT * unplaced_count_frac
    )

    # Anti-fragmentation penalty (exposed frontier)
    advancing_boxes = [b for b in placed_bboxes if not getattr(b, "is_door_anchor", False)]
    door_anchor_boxes = [b for b in placed_bboxes if getattr(b, "is_door_anchor", False)]

    if advancing_boxes:
        front_x = max(b.max_x for b in advancing_boxes)
        if door_anchor_boxes:
            min_door_x = min(b.min_x for b in door_anchor_boxes)
            if front_x >= min_door_x - 1e-4:
                front_x = max(b.max_x for b in placed_bboxes)
    elif placed_bboxes:
        front_x = max(b.max_x for b in placed_bboxes)
    else:
        front_x = 0.0

    frag_penalty = 0.0
    if front_x > 0 and container_volume > 0:
        vol_behind = sum(
            (b.max_x - b.min_x) * (b.max_y - b.min_y) * (b.max_z - b.min_z)
            for b in placed_bboxes
            if b.min_x < front_x
        )
        wasted_behind = max(
            0.0,
            front_x * container_dims.width * container_dims.height - vol_behind
        )
        frag_penalty = wasted_behind / container_volume

    frag_weight = getattr(settings, "FITNESS_FRAG_PENALTY_WEIGHT", 0.5)
    fill_term = (
        fill_rate
        - settings.FITNESS_COG_PENALTY_WEIGHT * (B1_norm + B2_norm + B3_norm)
        - frag_weight * frag_penalty
    )

    feasible = (stacking_violations == 0 and stability_violations == 0)

    fitness = (
        fill_term
        - unplaced_penalty
        - (0.0 if feasible else settings.INFEASIBLE_PENALTY)
    )

    return FitnessResult(
        fitness=fitness,
        placed_volume=placed_volume,
        unplaced_count=unplaced_count,
        cog_deviation_xy=cog_dev_xy,
        cog_deviation_z=cog_dev_z,
        stacking_violations=stacking_violations,
        stability_violations=stability_violations,
        frag_penalty=frag_penalty,
        unplaced_volume=unplaced_volume,
    )