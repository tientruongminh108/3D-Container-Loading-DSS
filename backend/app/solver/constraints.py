from typing import List, Optional, Tuple
from dataclasses import dataclass
from app.config import get_settings
from app.solver.geometry import (
    BoundingBox,
    Dimensions,
    Position,
    Posture,
    FLOOR_EPSILON,
    check_support_ratio,
    check_cog_balance,
)
from app.solver.parsing import Box
from app.solver.block_generation import Block


@dataclass
class PlacementCandidate:
    position: Position
    posture: Posture
    dims: Dimensions
    actual_dims: Dimensions
    box: Box


def check_weight_capacity(
    current_weight: float,
    candidate_weight: float,
    max_weight: float,
) -> bool:
    return (current_weight + candidate_weight) <= max_weight


def check_orientation(box: Box, posture: Posture) -> bool:
    return posture in box.permitted_postures


def check_non_overlap(
    candidate: BoundingBox,
    placed_boxes: List[BoundingBox],
) -> bool:
    c_min_x, c_max_x = candidate.min_x, candidate.max_x
    c_min_y, c_max_y = candidate.min_y, candidate.max_y
    c_min_z, c_max_z = candidate.min_z, candidate.max_z
    for placed in reversed(placed_boxes):
        if not (
            c_max_x <= placed.min_x
            or placed.max_x <= c_min_x
            or c_max_y <= placed.min_y
            or placed.max_y <= c_min_y
            or c_max_z <= placed.min_z
            or placed.max_z <= c_min_z
        ):
            return False
    return True


def check_container_bounds(
    candidate: BoundingBox,
    container_dims: Dimensions,
) -> bool:
    return (
        candidate.min_x >= 0
        and candidate.min_y >= 0
        and candidate.min_z >= 0
        and candidate.max_x <= container_dims.length
        and candidate.max_y <= container_dims.width
        and candidate.max_z <= container_dims.height
    )


def check_stackability(
    candidate: BoundingBox,
    placed_boxes: List[BoundingBox],
    candidate_box,
    placed_boxes_data: List,
    min_support_ratio: float = 0.60,
) -> bool:
    c_min_z = candidate.min_z
    if c_min_z <= FLOOR_EPSILON:
        return True

    footprint_area = (candidate.max_x - candidate.min_x) * (candidate.max_y - candidate.min_y)
    if footprint_area <= 0:
        return False

    contact_area = 0.0
    cand_unit_wt = getattr(candidate_box, 'boxes', [candidate_box])[0].weight_kg
    c_min_x, c_max_x = candidate.min_x, candidate.max_x
    c_min_y, c_max_y = candidate.min_y, candidate.max_y

    for i, placed in enumerate(placed_boxes):
        if abs(placed.max_z - c_min_z) <= FLOOR_EPSILON:
            ox = min(placed.max_x, c_max_x) - max(placed.min_x, c_min_x)
            if ox > 1e-4:
                oy = min(placed.max_y, c_max_y) - max(placed.min_y, c_min_y)
                if oy > 1e-4:
                    sup_unit_wt = getattr(placed_boxes_data[i], 'boxes', [placed_boxes_data[i]])[0].weight_kg
                    if cand_unit_wt > sup_unit_wt + 1e-3:
                        return False
                    contact_area += ox * oy

    return (contact_area / footprint_area) >= min_support_ratio


def check_lifo(
    candidate: BoundingBox,
    placed_boxes: List[BoundingBox],
    placed_boxes_data: List,
    candidate_sequence: int,
) -> bool:
    """Order-independent, bidirectional LIFO check (door = x L, rear wall = x 0).

    In the authoritative convention, origin (0, 0, 0) is the rear wall; door is at x = L.
    Cargo for an EARLIER drop-off (smaller customer_sequence) unloads first, so it
    must be nearer the door (larger x) than cargo for a LATER drop-off (smaller x).

    For every placed p sharing a Y-Z cross-section with candidate c:
    - if p.seq < c.seq (p unloads earlier, must be nearer door): require c.max_x <= p.min_x
    - if p.seq > c.seq (p unloads later, must be deeper): require p.max_x <= c.min_x
    - if p.seq == c.seq: no constraint.
    """
    c_min_x, c_max_x = candidate.min_x, candidate.max_x
    c_min_y, c_max_y = candidate.min_y, candidate.max_y
    c_min_z, c_max_z = candidate.min_z, candidate.max_z
    c_seq = candidate_sequence
    eps = 1e-6

    for placed, p_data in zip(reversed(placed_boxes), reversed(placed_boxes_data)):
        p_seq = getattr(p_data, 'customer_sequence', 0)
        if p_seq == c_seq:
            continue
        if c_max_y <= placed.min_y or placed.max_y <= c_min_y or c_max_z <= placed.min_z or placed.max_z <= c_min_z:
            continue
        if p_seq < c_seq:
            # candidate is later drop-off (deeper): must not intrude past placed's rear face
            if c_max_x > placed.min_x + eps:
                return False
        else:
            # candidate is earlier drop-off (nearer door): placed must not intrude past candidate's rear face
            if placed.max_x > c_min_x + eps:
                return False
    return True


def check_all_constraints(
    candidate: PlacementCandidate,
    placed_boxes: List[BoundingBox],
    placed_boxes_data: List[Box],
    container_dims: Dimensions,
    current_weight: float,
    max_weight: float,
    is_lcl: bool,
) -> Tuple[bool, str]:
    if not check_orientation(candidate.box, candidate.posture):
        return False, "orientation"

    # Construct the bounding box once and reuse it for all subsequent checks.
    candidate_bbox = BoundingBox.from_position_and_dims(candidate.position, candidate.dims)

    if not check_container_bounds(candidate_bbox, container_dims):
        return False, "container_bounds"

    if not check_non_overlap(candidate_bbox, placed_boxes):
        return False, "non_overlap"

    if not check_weight_capacity(current_weight, candidate.box.weight_kg, max_weight):
        return False, "weight_capacity"

    if not check_stackability(
        candidate_bbox, placed_boxes, candidate.box, placed_boxes_data
    ):
        return False, "stackability"

    if is_lcl:
        if not check_lifo(
            candidate_bbox, placed_boxes, placed_boxes_data, candidate.box.customer_sequence
        ):
            return False, "lifo"

    return True, "ok"