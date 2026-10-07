from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from app.solver.geometry import Dimensions, Posture, FLOOR_EPSILON, BoundingBox
from app.solver.constraints import check_corner_clearance
from app.core.models import PlacedBox


@dataclass
class ValidationReport:
    is_valid: bool
    total_violations: int
    violations_by_type: Dict[str, int]
    error_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def validate_solution(
    placed_boxes: List[PlacedBox],
    container_dims: Dimensions,
    max_weight_kg: float,
    is_lcl: bool = False,
    support_ratio_threshold: float = 0.60,
    eps: float = 1e-4,
) -> ValidationReport:
    """Independent validator that recomputes all hard constraints strictly from
    the final exploded output boxes only.

    Hard constraints checked:
    1. Duplicate Box IDs (integrity)
    2. Orientation (posture in permitted_postures, This_Way_Up respected)
    3. Container Bounds (all cartons within [0, L] x [0, W] x [0, H])
    4. Non-Overlap (no 3D spatial intersection between any pair of cartons)
    5. Container Max Weight Capacity
    6. Physical Support Ratio (>= 0.60 for all cartons above floor z > eps)
    7. Weight Hierarchy (upper carton weight <= supporting carton weight)
    8. LIFO Delivery Order (LCL mode: earlier drop-off must not be trapped behind later drop-off in Y-Z slice)
    """
    violations = {
        "duplicate_id": 0,
        "orientation": 0,
        "bounds": 0,
        "corner_clearance": 0,
        "overlap": 0,
        "weight_capacity": 0,
        "support_ratio": 0,
        "weight_hierarchy": 0,
        "lifo": 0,
    }
    messages: List[str] = []

    n = len(placed_boxes)
    if n == 0:
        return ValidationReport(
            is_valid=True,
            total_violations=0,
            violations_by_type=violations,
            error_messages=[],
            metrics={"placed_count": 0, "total_weight_kg": 0.0},
        )

    # 1. Duplicate Box IDs
    seen_ids = set()
    for b in placed_boxes:
        if b.box_id in seen_ids:
            violations["duplicate_id"] += 1
            messages.append(f"Duplicate box_id detected: {b.box_id}")
        seen_ids.add(b.box_id)

    # 2. Orientation & This-Way-Up
    for b in placed_boxes:
        if b.permitted_postures and b.posture not in b.permitted_postures:
            violations["orientation"] += 1
            messages.append(f"Box {b.box_id} posture {b.posture} not in permitted {b.permitted_postures}")
        if b.this_way_up and b.posture not in [Posture.LWH, Posture.WLH]:
            violations["orientation"] += 1
            messages.append(f"Box {b.box_id} This_Way_Up=True but placed in posture {b.posture}")

    # 3. Container Bounds & Corner Clearance
    cL, cW, cH = container_dims.length, container_dims.width, container_dims.height
    for b in placed_boxes:
        bx2 = b.x + b.actual_length
        by2 = b.y + b.actual_width
        bz2 = b.z + b.actual_height
        if (
            b.x < -eps
            or b.y < -eps
            or b.z < -eps
            or bx2 > cL + eps
            or by2 > cW + eps
            or bz2 > cH + eps
        ):
            violations["bounds"] += 1
            messages.append(
                f"Box {b.box_id} out of bounds: [{b.x:.2f}, {bx2:.2f}]x[{b.y:.2f}, {by2:.2f}]x[{b.z:.2f}, {bz2:.2f}] "
                f"vs container [{cL:.1f}, {cW:.1f}, {cH:.1f}]"
            )
        else:
            bbox = BoundingBox(b.x, b.y, b.z, bx2, by2, bz2)
            if not check_corner_clearance(bbox, container_dims):
                violations["corner_clearance"] += 1
                messages.append(
                    f"Box {b.box_id} intersects top-corner obstruction cuboid: "
                    f"[{b.x:.2f}, {bx2:.2f}]x[{b.y:.2f}, {by2:.2f}]x[{b.z:.2f}, {bz2:.2f}]"
                )

    # 4. Non-Overlap
    for i in range(n):
        b1 = placed_boxes[i]
        b1_x2 = b1.x + b1.actual_length
        b1_y2 = b1.y + b1.actual_width
        b1_z2 = b1.z + b1.actual_height
        for j in range(i + 1, n):
            b2 = placed_boxes[j]
            b2_x2 = b2.x + b2.actual_length
            b2_y2 = b2.y + b2.actual_width
            b2_z2 = b2.z + b2.actual_height

            dx = min(b1_x2, b2_x2) - max(b1.x, b2.x)
            dy = min(b1_y2, b2_y2) - max(b1.y, b2.y)
            dz = min(b1_z2, b2_z2) - max(b1.z, b2.z)

            if dx > eps and dy > eps and dz > eps:
                violations["overlap"] += 1
                messages.append(
                    f"Overlap between {b1.box_id} and {b2.box_id}: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
                )

    # 5. Weight Capacity
    total_weight = sum(b.weight_kg for b in placed_boxes)
    if total_weight > max_weight_kg + eps:
        violations["weight_capacity"] += 1
        messages.append(f"Total placed weight {total_weight:.2f} kg exceeds container max {max_weight_kg:.2f} kg")

    # 6. Physical Support Ratio & 7. Weight Hierarchy
    for i, b in enumerate(placed_boxes):
        if b.z > FLOOR_EPSILON:
            base_area = b.actual_length * b.actual_width
            if base_area <= 0:
                continue

            b_x2 = b.x + b.actual_length
            b_y2 = b.y + b.actual_width

            supporting_area = 0.0
            supporters: List[PlacedBox] = []

            for j, p in enumerate(placed_boxes):
                if i == j:
                    continue
                p_top = p.z + p.actual_height
                if abs(p_top - b.z) <= 1e-3:
                    # Flush top of p meets bottom of b
                    ov_x = min(b_x2, p.x + p.actual_length) - max(b.x, p.x)
                    ov_y = min(b_y2, p.y + p.actual_width) - max(b.y, p.y)
                    if ov_x > eps and ov_y > eps:
                        supporting_area += ov_x * ov_y
                        supporters.append(p)

            support_ratio = supporting_area / base_area
            if support_ratio < support_ratio_threshold - eps:
                violations["support_ratio"] += 1
                messages.append(
                    f"Box {b.box_id} at z={b.z:.2f} has insufficient support ratio: "
                    f"{support_ratio:.4f} < {support_ratio_threshold:.2f} (supporting area: {supporting_area:.1f}/{base_area:.1f})"
                )

            # Weight hierarchy: upper carton must never be heavier than any supporting carton
            for s in supporters:
                if b.weight_kg > s.weight_kg + 1e-3:
                    violations["weight_hierarchy"] += 1
                    messages.append(
                        f"Weight hierarchy violation: Box {b.box_id} ({b.weight_kg:.2f} kg) rests on "
                        f"lighter Box {s.box_id} ({s.weight_kg:.2f} kg)"
                    )

    # 8. LIFO Delivery Order (soft metric for LCL mode)
    # Earlier drop-off carton (lower sequence) must not be trapped behind a later drop-off carton (higher sequence)
    # towards the door (+X direction) in overlapping Y-Z profile.
    lifo_conflicts = 0
    lifo_blocked_cartons = 0
    if is_lcl:
        blocked_ids = set()
        for i in range(n):
            b1 = placed_boxes[i]
            seq1 = getattr(b1, "customer_sequence", 0) or 0
            if seq1 <= 0:
                continue
            b1_x2 = b1.x + b1.actual_length
            b1_y2 = b1.y + b1.actual_width
            b1_z2 = b1.z + b1.actual_height

            for j in range(n):
                if i == j:
                    continue
                b2 = placed_boxes[j]
                seq2 = getattr(b2, "customer_sequence", 0) or 0
                if seq2 <= 0:
                    continue
                # If b1 has earlier delivery than b2 (seq1 < seq2)
                # and b2 is closer to the door (+X) than b1
                if seq1 < seq2 and (b2.x + b2.actual_length) > b1_x2 + eps:
                    ov_y = min(b1_y2, b2.y + b2.actual_width) - max(b1.y, b2.y)
                    ov_z = min(b1_z2, b2.z + b2.actual_height) - max(b1.z, b2.z)
                    if ov_y > eps and ov_z > eps:
                        lifo_conflicts += 1
                        blocked_ids.add(b1.box_id)
        lifo_blocked_cartons = len(blocked_ids)

    # LIFO is reported as a soft metric (does not fail is_valid hard constraint check)
    violations["lifo"] = lifo_conflicts
    total_violations = sum(v for k, v in violations.items() if k != "lifo")
    is_valid = (total_violations == 0)

    # Compute additional physical load metrics for analysis
    c_vol = cL * cW * cH
    actual_volume = sum(b.actual_length * b.actual_width * b.actual_height for b in placed_boxes)
    physical_fill_rate = (actual_volume / c_vol) if c_vol > 0 else 0.0

    max_z = max((b.z + b.actual_height for b in placed_boxes), default=0.0)

    # Top-surface height standard deviation:
    # Identify top-level cartons (those with no carton resting on top of them)
    top_carton_heights = []
    for i, b in enumerate(placed_boxes):
        b_x2 = b.x + b.actual_length
        b_y2 = b.y + b.actual_width
        b_top = b.z + b.actual_height
        has_above = False
        for j, other in enumerate(placed_boxes):
            if i != j and other.z >= b_top - 1e-3:
                ov_x = min(b_x2, other.x + other.actual_length) - max(b.x, other.x)
                ov_y = min(b_y2, other.y + other.actual_width) - max(b.y, other.y)
                if ov_x > eps and ov_y > eps:
                    has_above = True
                    break
        if not has_above:
            top_carton_heights.append(b_top)

    top_surface_std = float(np.std(top_carton_heights)) if len(top_carton_heights) > 1 else 0.0

    metrics = {
        "placed_count": n,
        "total_weight_kg": total_weight,
        "physical_fill_rate": physical_fill_rate,
        "max_height_cm": max_z,
        "top_surface_height_std": top_surface_std,
        "lifo_conflicts": lifo_conflicts if is_lcl else None,
        "lifo_blocked_cartons": lifo_blocked_cartons if is_lcl else None,
    }

    return ValidationReport(
        is_valid=is_valid,
        total_violations=total_violations,
        violations_by_type=violations,
        error_messages=messages[:50],  # Cap sample messages
        metrics=metrics,
    )
