from typing import List, Optional, Tuple, Any
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
    transform_position_by_posture,
    compute_block_content_rel_pos,
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


def check_corner_clearance(
    candidate: BoundingBox,
    container_dims: Dimensions,
) -> bool:
    """Check whether a candidate BoundingBox avoids the 4 top-corner obstruction
    cuboids (ISO 1161 corner casting approximation: 17.8 x 16.2 x 11.8 cm).

    The 4 top corners are:
    - Rear-left-top:  x in [0, X_CM],       y in [0, Y_CM],       z in [H - Z_CM, H]
    - Rear-right-top: x in [0, X_CM],       y in [W - Y_CM, W],   z in [H - Z_CM, H]
    - Door-left-top:  x in [L - X_CM, L],   y in [0, Y_CM],       z in [H - Z_CM, H]
    - Door-right-top: x in [L - X_CM, L],   y in [W - Y_CM, W],   z in [H - Z_CM, H]
    """
    settings = get_settings()
    cz = getattr(settings, "CORNER_BLOCK_Z_CM", 11.8)
    # Early-exit: most candidates are below the top corner obstruction zone
    if candidate.max_z <= container_dims.height - cz:
        return True

    cx = getattr(settings, "CORNER_BLOCK_X_CM", 17.8)
    cy = getattr(settings, "CORNER_BLOCK_Y_CM", 16.2)
    cL = container_dims.length
    cW = container_dims.width

    # ISO 1161 corner castings only apply when the container is sufficiently large
    # that opposite corner castings do not overlap (L > 2*cx, W > 2*cy, H > 2*cz).
    if cL <= 2 * cx or cW <= 2 * cy or container_dims.height <= 2 * cz:
        return True

    # Check X overlap with rear corners [0, cx]
    overlap_rear_x = (candidate.min_x < cx and candidate.max_x > 0)
    # Check X overlap with door corners [cL - cx, cL]
    overlap_door_x = (candidate.max_x > cL - cx and candidate.min_x < cL)

    if not (overlap_rear_x or overlap_door_x):
        return True

    # Check Y overlap with left corners [0, cy]
    overlap_left_y = (candidate.min_y < cy and candidate.max_y > 0)
    # Check Y overlap with right corners [cW - cy, cW]
    overlap_right_y = (candidate.max_y > cW - cy and candidate.min_y < cW)

    if (overlap_rear_x or overlap_door_x) and (overlap_left_y or overlap_right_y):
        return False

    return True


def check_block_cartons_stackability(
    block: Any,
    candidate_bbox: BoundingBox,
    posture: Posture,
    placed_boxes: List[BoundingBox],
    placed_boxes_data: List[Any],
    min_support_ratio: float = 0.60,
    placed_postures: Optional[List[Posture]] = None,
) -> bool:
    """Validate that every individual carton inside a static Block will have
    sufficient support ratio (>= min_support_ratio) and satisfy weight hierarchy
    once the Block is placed at candidate_bbox with posture and exploded into cartons.

    Siblings within the same Block that are below a carton are included as valid
    support surfaces.
    """
    contents = getattr(block, "contents", None)
    if not contents or len(contents) <= 1:
        return True

    # Fast-path: if the block's base rests on the floor, all bottom-layer cartons
    # are on the floor (100% support) and all higher layers sit on lower layers.
    if candidate_bbox.min_z <= FLOOR_EPSILON:
        return True

    # Pre-extract supporting surfaces from placed_boxes at candidate_bbox.min_z
    sup_surfaces = []
    for i, placed in enumerate(placed_boxes):
        if abs(placed.max_z - candidate_bbox.min_z) <= FLOOR_EPSILON:
            p_data = placed_boxes_data[i]
            p_posture = placed_postures[i] if (placed_postures is not None and i < len(placed_postures)) else Posture.LWH
            if hasattr(p_data, "contents") and len(p_data.contents) > 1:
                for sub_c in p_data.contents:
                    p_pos = compute_block_content_rel_pos(sub_c, p_posture)
                    sub_act = Dimensions(sub_c.length_cm, sub_c.width_cm, sub_c.height_cm).apply_posture(p_posture)
                    sub_top = placed.min_z + p_pos.z + sub_act.height
                    if abs(sub_top - candidate_bbox.min_z) <= FLOOR_EPSILON:
                        sup_surfaces.append((
                            placed.min_x + p_pos.x,
                            placed.min_x + p_pos.x + sub_act.length,
                            placed.min_y + p_pos.y,
                            placed.min_y + p_pos.y + sub_act.width,
                            sub_c.weight_kg,
                        ))
            elif hasattr(p_data, 'length_cm'):
                p_dims = Dimensions(p_data.length_cm, p_data.width_cm, p_data.height_cm).apply_posture(p_posture)
                p_max_x = placed.min_x + p_dims.length
                p_max_y = placed.min_y + p_dims.width
                sup_unit_wt = getattr(p_data, 'boxes', [p_data])[0].weight_kg
                sup_surfaces.append((placed.min_x, p_max_x, placed.min_y, p_max_y, sup_unit_wt))
            else:
                p_max_x = placed.max_x
                p_max_y = placed.max_y
                sup_unit_wt = getattr(p_data, 'boxes', [p_data])[0].weight_kg
                sup_surfaces.append((placed.min_x, p_max_x, placed.min_y, p_max_y, sup_unit_wt))

    if not sup_surfaces:
        return False

    first_content = contents[0]
    c_dims = Dimensions(
        first_content.length_cm, first_content.width_cm, first_content.height_cm
    ).apply_posture(posture)
    cand_unit_wt = first_content.weight_kg
    footprint_area = c_dims.length * c_dims.width
    if footprint_area <= 0:
        return False
    min_contact = footprint_area * min_support_ratio - 1e-4

    # Only cartons on the bottom face of the block (world_rel_pos.z <= 1e-4) need
    # support from external placed_boxes. Cartons in higher layers of the block sit
    # on identical sibling cartons directly beneath them within the intact block.
    for content in contents:
        world_rel_pos = compute_block_content_rel_pos(content, posture)
        if world_rel_pos.z > 1e-4:
            continue

        c_min_x = candidate_bbox.min_x + world_rel_pos.x
        c_max_x = c_min_x + c_dims.length
        c_min_y = candidate_bbox.min_y + world_rel_pos.y
        c_max_y = c_min_y + c_dims.width

        contact_area = 0.0
        for p_min_x, p_max_x, p_min_y, p_max_y, sup_unit_wt in sup_surfaces:
            ox = min(p_max_x, c_max_x) - max(p_min_x, c_min_x)
            if ox > 1e-4:
                oy = min(p_max_y, c_max_y) - max(p_min_y, c_min_y)
                if oy > 1e-4:
                    if cand_unit_wt > sup_unit_wt + 1e-3:
                        return False
                    contact_area += ox * oy

        if contact_area < min_contact:
            return False

    return True


def check_stackability(
    candidate: BoundingBox,
    placed_boxes: List[BoundingBox],
    candidate_box,
    placed_boxes_data: List,
    min_support_ratio: float = 0.60,
    posture: Optional[Posture] = None,
    placed_postures: Optional[List[Posture]] = None,
) -> bool:
    c_min_z = candidate.min_z
    if c_min_z <= FLOOR_EPSILON:
        return True

    cand_unit_wt = getattr(candidate_box, 'boxes', [candidate_box])[0].weight_kg

    if posture is not None and hasattr(candidate_box, 'length_cm') and hasattr(candidate_box, 'width_cm') and hasattr(candidate_box, 'height_cm'):
        cand_dims = Dimensions(candidate_box.length_cm, candidate_box.width_cm, candidate_box.height_cm).apply_posture(posture)
        footprint_area = cand_dims.length * cand_dims.width
        c_min_x = candidate.min_x
        c_max_x = candidate.min_x + cand_dims.length
        c_min_y = candidate.min_y
        c_max_y = candidate.min_y + cand_dims.width
    else:
        footprint_area = (candidate.max_x - candidate.min_x) * (candidate.max_y - candidate.min_y)
        c_min_x, c_max_x = candidate.min_x, candidate.max_x
        c_min_y, c_max_y = candidate.min_y, candidate.max_y

    if footprint_area <= 0:
        return False

    contact_area = 0.0
    for i, placed in enumerate(placed_boxes):
        if abs(placed.max_z - c_min_z) <= FLOOR_EPSILON:
            p_data = placed_boxes_data[i]
            p_posture = placed_postures[i] if (placed_postures is not None and i < len(placed_postures)) else Posture.LWH
            if hasattr(p_data, "contents") and len(p_data.contents) > 1:
                for sub_c in p_data.contents:
                    p_pos = compute_block_content_rel_pos(sub_c, p_posture)
                    sub_act = Dimensions(sub_c.length_cm, sub_c.width_cm, sub_c.height_cm).apply_posture(p_posture)
                    sub_top = placed.min_z + p_pos.z + sub_act.height
                    if abs(sub_top - c_min_z) <= FLOOR_EPSILON:
                        s_x1 = placed.min_x + p_pos.x
                        s_x2 = s_x1 + sub_act.length
                        s_y1 = placed.min_y + p_pos.y
                        s_y2 = s_y1 + sub_act.width
                        ox = min(s_x2, c_max_x) - max(s_x1, c_min_x)
                        if ox > 1e-4:
                            oy = min(s_y2, c_max_y) - max(s_y1, c_min_y)
                            if oy > 1e-4:
                                if cand_unit_wt > sub_c.weight_kg + 1e-3:
                                    return False
                                contact_area += ox * oy
                continue

            if (
                hasattr(p_data, 'length_cm')
                and hasattr(p_data, 'width_cm')
                and hasattr(p_data, 'height_cm')
            ):
                p_dims = Dimensions(p_data.length_cm, p_data.width_cm, p_data.height_cm).apply_posture(p_posture)
                p_max_x = placed.min_x + p_dims.length
                p_max_y = placed.min_y + p_dims.width
            else:
                p_max_x = placed.max_x
                p_max_y = placed.max_y

            ox = min(p_max_x, c_max_x) - max(placed.min_x, c_min_x)
            if ox > 1e-4:
                oy = min(p_max_y, c_max_y) - max(placed.min_y, c_min_y)
                if oy > 1e-4:
                    sup_unit_wt = getattr(p_data, 'boxes', [p_data])[0].weight_kg
                    if cand_unit_wt > sup_unit_wt + 1e-3:
                        return False
                    contact_area += ox * oy

    if (contact_area / footprint_area) < min_support_ratio - 1e-4:
        return False

    if posture is not None:
        if not check_block_cartons_stackability(
            candidate_box, candidate, posture, placed_boxes, placed_boxes_data, min_support_ratio, placed_postures=placed_postures
        ):
            return False

    return True


def check_all_constraints(
    candidate: PlacementCandidate,
    placed_boxes: List[BoundingBox],
    placed_boxes_data: List[Box],
    container_dims: Dimensions,
    current_weight: float,
    max_weight: float,
    is_lcl: bool = False,
    placed_postures: Optional[List[Posture]] = None,
) -> Tuple[bool, str]:
    if not check_orientation(candidate.box, candidate.posture):
        return False, "orientation"

    # Construct the bounding box once and reuse it for all subsequent checks.
    candidate_bbox = BoundingBox.from_position_and_dims(candidate.position, candidate.dims)

    if not check_container_bounds(candidate_bbox, container_dims):
        return False, "container_bounds"

    if not check_corner_clearance(candidate_bbox, container_dims):
        return False, "corner_clearance"

    if not check_non_overlap(candidate_bbox, placed_boxes):
        return False, "non_overlap"

    if not check_weight_capacity(current_weight, candidate.box.weight_kg, max_weight):
        return False, "weight_capacity"

    if not check_stackability(
        candidate_bbox, placed_boxes, candidate.box, placed_boxes_data, posture=candidate.posture, placed_postures=placed_postures
    ):
        return False, "stackability"

    return True, "ok"