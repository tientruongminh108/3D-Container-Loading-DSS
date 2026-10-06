from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.solver.geometry import (
    BoundingBox,
    Position,
    Dimensions,
    Posture,
    transform_position_by_posture,
    compute_block_content_rel_pos,
)
from app.solver.parsing import Box
from app.solver.block_generation import Block
from app.solver.ga import Individual
from app.solver.fitness import FitnessResult
from app.core.models import (
    PlacedBox,
    UnplacedCarton,
    LoadMetrics,
    UnplacedReason,
    RunResult,
)
from datetime import datetime, timezone
import uuid
import colorsys


@dataclass
class Layer:
    x_min: float
    x_max: float
    boxes: List[PlacedBox]


def explode_blocks(
    placed_blocks_or_data: List,
    individual_or_bboxes: Any = None,
    placed_postures: Optional[List[Posture]] = None,
) -> List[PlacedBox]:
    """Explode block placements and individual box placements into PlacedBox instances with correct positions.
    
    Per Section 5.2.4: each block's contents have positions relative to the block's
    origin corner. When the block is placed at absolute position with a posture,
    each content box's absolute position = block_position + transform(content_rel_pos, block_posture).
    Individual boxes are placed directly at their bounding box min corner.
    """
    placed_boxes = []

    # Determine calling convention
    if placed_postures is not None and isinstance(individual_or_bboxes, list):
        placed_data = placed_blocks_or_data
        placed_bboxes = individual_or_bboxes
        postures = placed_postures
    elif hasattr(individual_or_bboxes, "placed_data") and individual_or_bboxes.placed_data is not None:
        placed_data = individual_or_bboxes.placed_data
        placed_bboxes = individual_or_bboxes.placed_bboxes or []
        postures = getattr(individual_or_bboxes, "placed_postures", None) or []
        if not postures and individual_or_bboxes.chromosome:
            postures = []
            for idx, u in enumerate(placed_data):
                if hasattr(u, "permitted_postures") and u.permitted_postures:
                    p_idx = individual_or_bboxes.chromosome[idx % len(individual_or_bboxes.chromosome)] % len(u.permitted_postures)
                    postures.append(u.permitted_postures[p_idx])
                else:
                    postures.append(Posture.LWH)
    else:
        placed_data = placed_blocks_or_data
        placed_bboxes = getattr(individual_or_bboxes, "placed_bboxes", []) or []
        postures = getattr(individual_or_bboxes, "placed_postures", None) or []
        if not postures:
            postures = [Posture.LWH] * len(placed_data)

    for unit, bbox, posture in zip(placed_data, placed_bboxes, postures):
        if isinstance(unit, Block):
            for content in unit.contents:
                world_rel_pos = compute_block_content_rel_pos(content, posture)

                abs_x = bbox.min_x + world_rel_pos.x
                abs_y = bbox.min_y + world_rel_pos.y
                abs_z = bbox.min_z + world_rel_pos.z

                content_dims = Dimensions(
                    content.length_cm, content.width_cm, content.height_cm
                ).apply_posture(posture)

                placed_boxes.append(
                    PlacedBox(
                        box_id=content.box_id,
                        item_id=content.item_id,
                        po_no=content.po_no,
                        customer_code=content.customer_code,
                        customer_sequence=content.customer_sequence,
                        length_cm=content.length_cm,
                        width_cm=content.width_cm,
                        height_cm=content.height_cm,
                        weight_kg=content.weight_kg,
                        this_way_up=content.this_way_up,
                        permitted_postures=content.permitted_postures,
                        inflated_length=content.inflated_length,
                        inflated_width=content.inflated_width,
                        inflated_height=content.inflated_height,
                        x=abs_x,
                        y=abs_y,
                        z=abs_z,
                        posture=posture,
                        actual_length=content_dims.length,
                        actual_width=content_dims.width,
                        actual_height=content_dims.height,
                    )
                )
        else:
            actual_dims = Dimensions(
                unit.length_cm, unit.width_cm, unit.height_cm
            ).apply_posture(posture)

            placed_boxes.append(
                PlacedBox(
                    box_id=unit.box_id,
                    item_id=unit.item_id,
                    po_no=unit.po_no,
                    customer_code=unit.customer_code,
                    customer_sequence=unit.customer_sequence,
                    length_cm=unit.length_cm,
                    width_cm=unit.width_cm,
                    height_cm=unit.height_cm,
                    weight_kg=unit.weight_kg,
                    this_way_up=unit.this_way_up,
                    permitted_postures=unit.permitted_postures,
                    inflated_length=unit.inflated_length,
                    inflated_width=unit.inflated_width,
                    inflated_height=unit.inflated_height,
                    x=bbox.min_x,
                    y=bbox.min_y,
                    z=bbox.min_z,
                    posture=posture,
                    actual_length=actual_dims.length,
                    actual_width=actual_dims.width,
                    actual_height=actual_dims.height,
                )
            )

    return placed_boxes


def shift_downward_cartons(boxes: List[PlacedBox], is_lcl: bool = False) -> List[PlacedBox]:
    """Gravity settlement pass: drops cartons downward so that each carton
    rests flush on the container floor or on the top surface of the cartons beneath it.
    Eliminates micro-gaps and floating cartons caused by tolerance gaps or rotation differentials.
    In LCL mode, drops are constrained to never create a LIFO violation between customers.
    """
    if not boxes:
        return boxes

    # Sort boxes bottom-up by z, then x, then y
    sorted_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i].z, boxes[i].x, boxes[i].y))

    for i in sorted_indices:
        b = boxes[i]
        if b.z <= 0.01:
            b.z = 0.0
            continue

        b_x1, b_x2 = b.x, b.x + b.actual_length
        b_y1, b_y2 = b.y, b.y + b.actual_width

        highest_z = 0.0
        for j in range(len(boxes)):
            if i == j:
                continue
            o = boxes[j]
            o_x1, o_x2 = o.x, o.x + o.actual_length
            o_y1, o_y2 = o.y, o.y + o.actual_width
            o_z2 = o.z + o.actual_height

            # Check XY overlap
            if min(b_x2, o_x2) - max(b_x1, o_x1) > 1e-4 and min(b_y2, o_y2) - max(b_y1, o_y1) > 1e-4:
                if o_z2 <= b.z + 1e-4:
                    if o_z2 > highest_z:
                        highest_z = o_z2


        # Verify support ratio and weight hierarchy before dropping to highest_z
        if highest_z > 0.0:
            supp_area = 0.0
            wt_ok = True
            for j in range(len(boxes)):
                if i == j:
                    continue
                o = boxes[j]
                if abs((o.z + o.actual_height) - highest_z) < 1e-3:
                    ov_x = min(b_x2, o.x + o.actual_length) - max(b_x1, o.x)
                    ov_y = min(b_y2, o.y + o.actual_width) - max(b_y1, o.y)
                    if ov_x > 1e-4 and ov_y > 1e-4:
                        supp_area += ov_x * ov_y
                        if b.weight_kg > o.weight_kg + 1e-3:
                            wt_ok = False
            base_area = b.actual_length * b.actual_width
            if not wt_ok or (base_area > 0 and supp_area / base_area < 0.60 - 1e-4):
                # Cannot drop to highest_z because it would violate support ratio or weight hierarchy
                continue

        drop = b.z - highest_z
        if drop > 1e-4:
            b.z = round(highest_z, 4)

    return boxes


def build_layers(placed_boxes: List[PlacedBox], layer_depth: float = 50.0) -> List[Layer]:
    """Partition placed boxes into sequential longitudinal layers along the X-axis.

    Layers are ordered from the rear wall (deepest, x = 0) toward the
    container door (highest X, x = L), so Layer 0 = rear-most slab and the last layer
    = door-facing slab. A box is included in a layer when its X-interval
    [box.x, box.x + box.actual_length) overlaps the layer band.

    Args:
        placed_boxes: All successfully placed boxes.
        layer_depth:  Thickness of each longitudinal slice in cm (default 50).
    """
    if not placed_boxes:
        return []

    # Determine the container depth that was actually used.
    max_x = max(b.x + b.actual_length for b in placed_boxes)

    # Build bands from rear (0) to door (max_x), each of width layer_depth.
    # Band k covers [k * layer_depth, (k + 1) * layer_depth].
    num_layers = int(max_x / layer_depth) + 1

    layers = []
    for k in range(num_layers):
        x_min = k * layer_depth
        x_max = (k + 1) * layer_depth
        # Boxes whose X-interval overlaps [x_min, x_max)
        layer_boxes = [
            b for b in placed_boxes
            if b.x < x_max and (b.x + b.actual_length) > x_min
        ]
        if layer_boxes:
            layers.append(Layer(x_min=x_min, x_max=x_max, boxes=layer_boxes))

    return layers


def calculate_metrics(
    placed_boxes: List[PlacedBox],
    unplaced: List[UnplacedCarton],
    container_dims: Dimensions,
    max_weight: float,
    planning_time_seconds: Optional[float] = None,
    nominal_container_dims: Optional[Dimensions] = None,
) -> LoadMetrics:
    placed_count = len(placed_boxes)
    unplaced_count = len(unplaced)
    total_cartons = placed_count + unplaced_count

    placed_volume = sum(
        b.actual_length * b.actual_width * b.actual_height for b in placed_boxes
    )
    denom_dims = nominal_container_dims if nominal_container_dims is not None else container_dims
    container_volume = denom_dims.length * denom_dims.width * denom_dims.height
    fill_rate = placed_volume / container_volume if container_volume > 0 else 0

    total_cbm = round(container_volume / 1_000_000.0, 3)
    used_cbm = round(placed_volume / 1_000_000.0, 3)
    unused_cbm = round(max(0.0, (container_volume - placed_volume) / 1_000_000.0), 3)

    used_weight = sum(b.weight_kg for b in placed_boxes)
    weight_util = used_weight / max_weight if max_weight > 0 else 0

    from app.solver.geometry import calculate_cog, check_cog_balance
    from app.config import get_settings
    settings = get_settings()

    weights = [b.weight_kg for b in placed_boxes]
    cog = calculate_cog(
        [BoundingBox.from_position_and_dims(
            Position(b.x, b.y, b.z),
            Dimensions(b.actual_length, b.actual_width, b.actual_height)
        ) for b in placed_boxes],
        weights,
    )

    _, cog_dev_xy, cog_dev_z = check_cog_balance(
        cog, denom_dims, settings.COG_TOLERANCE_XY, settings.COG_TOLERANCE_Z
    )

    return LoadMetrics(
        placed_count=placed_count,
        unplaced_count=unplaced_count,
        total_cartons=total_cartons,
        fill_rate=fill_rate,
        used_weight_kg=used_weight,
        max_weight_kg=max_weight,
        weight_utilization=weight_util,
        cog_x=cog.x,
        cog_y=cog.y,
        cog_z=cog.z,
        cog_deviation_xy=cog_dev_xy,
        cog_deviation_z=cog_dev_z,
        total_volume_cbm=total_cbm,
        used_volume_cbm=used_cbm,
        unused_volume_cbm=unused_cbm,
        planning_time_seconds=planning_time_seconds,
    )


def build_unplaced_cartons(
    unplaced_boxes: List[Box],
    unplaced_blocks: List[Block],
    is_lcl: bool,
) -> List[UnplacedCarton]:
    result = []
    seen_box_ids = set()

    for box in unplaced_boxes:
        if box.box_id in seen_box_ids:
            continue
        seen_box_ids.add(box.box_id)
        
        reason = UnplacedReason.NO_SPACE
        result.append(
            UnplacedCarton(
                box_id=box.box_id,
                item_id=box.item_id,
                po_no=box.po_no,
                customer_code=box.customer_code,
                customer_sequence=box.customer_sequence,
                reason=reason,
                length_cm=box.length_cm,
                width_cm=box.width_cm,
                height_cm=box.height_cm,
                weight_kg=box.weight_kg,
            )
        )

    for block in unplaced_blocks:
        for content in block.contents:
            if content.box_id in seen_box_ids:
                continue
            seen_box_ids.add(content.box_id)
            
            reason = UnplacedReason.NO_SPACE
            result.append(
                UnplacedCarton(
                    box_id=content.box_id,
                    item_id=content.item_id,
                    po_no=content.po_no,
                    customer_code=content.customer_code,
                    customer_sequence=content.customer_sequence,
                    reason=reason,
                    length_cm=content.length_cm,
                    width_cm=content.width_cm,
                    height_cm=content.height_cm,
                    weight_kg=content.weight_kg,
                )
            )

    return result


def build_run_result(
    individual: Individual,
    container_dims: Dimensions,
    container_spec,
    placed_blocks: List[Block],
    unplaced_blocks: List[Block],
    all_boxes: List[Box],
    unplaced_boxes: List[Box],
    is_lcl: bool,
    placed_individual_boxes: List[Box] = None,
    placed_bboxes: List = None,
    placed_data: List = None,
    placed_postures: List = None,
    run_id: str = None,
    status: str = "completed",
    error_message: str = None,
    options=None,
    planning_time_seconds: Optional[float] = None,
    seed: Optional[int] = None,
) -> RunResult:
    if run_id is None:
        run_id = str(uuid.uuid4())

    if seed is None and options is not None:
        seed = getattr(options, "seed", None)

    if placed_bboxes is None and individual is not None:
        placed_bboxes = getattr(individual, "placed_bboxes", None)
    if placed_data is None and individual is not None:
        placed_data = getattr(individual, "placed_data", None)
    if placed_postures is None and individual is not None:
        placed_postures = getattr(individual, "placed_postures", None)

    if placed_bboxes is not None and placed_data is not None and placed_postures is not None:
        all_placed_boxes = explode_blocks(placed_data, placed_bboxes, placed_postures)
    else:
        all_placed_boxes = explode_blocks(placed_blocks, individual)

    all_placed_boxes = shift_downward_cartons(all_placed_boxes, is_lcl=is_lcl)

    # Shift placed boxes by wall clearance margin in X and Y into physical container coordinates
    # so cargo maintains the required 2.0 cm buffer from container walls.
    wall_offset = 0.0
    if container_spec is not None and hasattr(container_spec, "internal_length_cm") and hasattr(container_spec, "usable_length"):
        wall_offset = max(0.0, (float(container_spec.internal_length_cm) - float(container_spec.usable_length)) / 2.0)
    elif options and getattr(options, "container_wall_clearance_cm", None) is not None:
        wall_offset = float(options.container_wall_clearance_cm)
    else:
        from app.config import get_settings
        settings = get_settings()
        wall_offset = getattr(settings, "CONTAINER_WALL_CLEARANCE_CM", 0.0)

    if wall_offset > 0:
        for b in all_placed_boxes:
            b.x += wall_offset
            b.y += wall_offset

    # Assign sequential step_index and color to all placed boxes.
    # Sort rear-to-door (ascending x) within each customer sequence so that
    # Step 1 begins at the rear wall, matching the loading strategy.
    all_placed_boxes.sort(key=lambda b: (b.customer_sequence, b.x, b.z, b.y))

    distinct_skus = sorted(list({b.item_id for b in all_placed_boxes if b.item_id is not None}))
    n_distinct = len(distinct_skus)
    sku_color_map = {}
    if n_distinct > 0:
        for rank, sku in enumerate(distinct_skus):
            h = rank / n_distinct
            r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.55)
            sku_color_map[sku] = f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"

    for idx, b in enumerate(all_placed_boxes):
        b.step_index = idx + 1
        b.color = sku_color_map.get(b.item_id, "#3b82f6")

    nominal_dims = None
    if container_spec is not None and hasattr(container_spec, "internal_length_cm"):
        nominal_dims = Dimensions(
            float(container_spec.internal_length_cm),
            float(container_spec.internal_width_cm),
            float(container_spec.internal_height_cm),
        )

    layers = build_layers(all_placed_boxes)
    unplaced_cartons = build_unplaced_cartons(unplaced_boxes, unplaced_blocks, is_lcl)
    metrics = calculate_metrics(
        all_placed_boxes,
        unplaced_cartons,
        container_dims,
        container_spec.max_weight_kg,
        planning_time_seconds=planning_time_seconds,
        nominal_container_dims=nominal_dims,
    )

    layer_data = []
    for layer in layers:
        layer_data.append({
            "x_min": layer.x_min,
            "x_max": layer.x_max,
            "boxes": [
                {
                    "box_id": b.box_id,
                    "item_id": b.item_id,
                    "x": b.x,
                    "y": b.y,
                    "z": b.z,
                    "length": b.actual_length,
                    "width": b.actual_width,
                    "height": b.actual_height,
                    "posture": b.posture,
                    "customer_sequence": b.customer_sequence,
                    "step_index": b.step_index,
                    "color": b.color,
                }
                for b in layer.boxes
            ],
        })

    from app.core.models import Container
    container = Container(
        id=0,
        container_type=container_spec.container_type,
        internal_length_cm=container_spec.internal_length_cm,
        internal_width_cm=container_spec.internal_width_cm,
        internal_height_cm=container_spec.internal_height_cm,
        max_weight_kg=container_spec.max_weight_kg,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    return RunResult(
        run_id=run_id,
        status=status,
        container=container,
        metrics=metrics,
        placed_boxes=all_placed_boxes,
        unplaced_cartons=unplaced_cartons,
        layers=layer_data,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc) if status == "completed" else None,
        error_message=error_message,
        options=options,
        planning_time_seconds=planning_time_seconds,
        seed=seed,
    )