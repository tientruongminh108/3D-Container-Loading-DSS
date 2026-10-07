from typing import List, Callable, Optional, Tuple, Any
from dataclasses import dataclass
from app.config import get_settings
from app.solver.parsing import parse_and_join, Box, ContainerSpec, PackingListPreview, ShipmentType
from app.solver.sorting import initial_sort, resort_after_blocks
from app.solver.block_generation import build_blocks, Block
from app.solver.ga import genetic_algorithm, Individual
from app.solver.placement import decode_chromosome, place_blocks_greedy
from app.solver.output import build_run_result
import time
from app.solver.geometry import Dimensions, BoundingBox, get_unit_inflated_dims
from app.core.models import RunResult, RunStatus


@dataclass
class PipelineResult:
    result: RunResult
    best_individual: Individual
    placed_blocks: List[Block]
    unplaced_blocks: List[Block]
    all_boxes: List[Box]
    unplaced_boxes: List[Box]
    container_spec: ContainerSpec
    is_lcl: bool


@dataclass
class ResolvedStrategy:
    use_static_blocks: bool
    dynamic_blocks: bool
    ga_level: str
    group_key: str
    post_explode_compaction: bool


def _resolve_strategy(options, settings) -> ResolvedStrategy:
    """Resolve solver strategy options, ensuring user-specified options take precedence
    over default heuristics (preventing the options-override trap).
    """
    # Baseline defaults (E_PEC configuration)
    use_static_blocks = False
    dynamic_blocks = True
    ga_level = "group"
    group_key = "geometry"
    post_explode_compaction = True

    if options is not None:
        if options.use_static_blocks is True:
            use_static_blocks = True
            dynamic_blocks = False
            ga_level = "unit"
        elif options.ga_level == "carton":
            use_static_blocks = False
            dynamic_blocks = False
            ga_level = "carton"

        if options.use_static_blocks is not None:
            use_static_blocks = options.use_static_blocks
        if options.dynamic_blocks is not None:
            dynamic_blocks = options.dynamic_blocks
        if options.ga_level is not None:
            ga_level = options.ga_level
        if options.group_key is not None:
            group_key = options.group_key
        if options.post_explode_compaction is not None:
            post_explode_compaction = options.post_explode_compaction

    return ResolvedStrategy(
        use_static_blocks=use_static_blocks,
        dynamic_blocks=dynamic_blocks,
        ga_level=ga_level,
        group_key=group_key,
        post_explode_compaction=post_explode_compaction,
    )


def _explode_to_units(
    placed_data: List[Any],
    placed_bboxes: List[BoundingBox],
    placed_postures: List[Any],
    unplaced: List[Tuple[Any, str]],
) -> Tuple[List[BoundingBox], List[Box], List[Any], List[Tuple[Box, str]]]:
    """Explode placed and unplaced blocks into individual box units with bounding boxes."""
    from app.solver.output import explode_blocks
    exploded_cartons = explode_blocks(placed_data, placed_bboxes, placed_postures)
    c_bboxes = []
    c_data = []
    c_postures = []
    for c in exploded_cartons:
        _, c_inf = get_unit_inflated_dims(c, c.posture)
        c_bboxes.append(
            BoundingBox(
                c.x, c.y, c.z,
                c.x + c_inf.length,
                c.y + c_inf.width,
                c.z + c_inf.height,
            )
        )
        c_box = Box(
            box_id=c.box_id,
            item_id=c.item_id,
            po_no=c.po_no,
            customer_code=c.customer_code,
            customer_sequence=c.customer_sequence,
            length_cm=c.length_cm,
            width_cm=c.width_cm,
            height_cm=c.height_cm,
            weight_kg=c.weight_kg,
            this_way_up=c.this_way_up,
            permitted_postures=c.permitted_postures,
            inflated_length=c.inflated_length,
            inflated_width=c.inflated_width,
            inflated_height=c.inflated_height,
        )
        c_data.append(c_box)
        c_postures.append(c.posture)

    exploded_unplaced = []
    for u, r in unplaced:
        if isinstance(u, Block):
            for c in u.contents:
                exploded_unplaced.append((c, r))
        else:
            exploded_unplaced.append((u, r))

    return c_bboxes, c_data, c_postures, exploded_unplaced


def _finalize(
    best_individual: Individual,
    placed_bboxes: List[BoundingBox],
    placed_data: List[Any],
    placed_postures: List[Any],
    unplaced: List[Tuple[Any, str]],
    container_dims: Dimensions,
    container_spec: ContainerSpec,
    boxes: List[Box],
    is_lcl: bool,
    options: Optional[Any],
    start_time: float,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float, dict], None]] = None,
) -> PipelineResult:
    """Finalize run result, separating blocks and boxes, calculating timings and building PipelineResult."""
    placed_blocks = []
    placed_individual_boxes = []
    for unit in placed_data:
        if isinstance(unit, Block):
            placed_blocks.append(unit)
        else:
            placed_individual_boxes.append(unit)

    final_unplaced_blocks = []
    final_unplaced_boxes = []
    for u, r in unplaced:
        if isinstance(u, Block):
            u.unplaced_reason = r
            for c in u.contents:
                c.unplaced_reason = r
            final_unplaced_blocks.append(u)
        else:
            u.unplaced_reason = r
            final_unplaced_boxes.append(u)

    planning_time_sec = round(time.perf_counter() - start_time, 2)

    result = build_run_result(
        individual=best_individual,
        container_dims=container_dims,
        container_spec=container_spec,
        placed_blocks=placed_blocks,
        unplaced_blocks=final_unplaced_blocks,
        all_boxes=boxes,
        unplaced_boxes=final_unplaced_boxes,
        is_lcl=is_lcl,
        placed_bboxes=placed_bboxes,
        placed_data=placed_data,
        placed_individual_boxes=placed_individual_boxes,
        placed_postures=placed_postures,
        status=RunStatus.COMPLETED.value,
        options=options,
        planning_time_seconds=planning_time_sec,
        seed=seed,
    )

    if progress_callback:
        progress_callback("complete", 1.0, {"message": "Done", "run_id": result.run_id})

    return PipelineResult(
        result=result,
        best_individual=best_individual,
        placed_blocks=placed_blocks,
        unplaced_blocks=final_unplaced_blocks,
        all_boxes=boxes,
        unplaced_boxes=final_unplaced_boxes,
        container_spec=container_spec,
        is_lcl=is_lcl,
    )


def run_pipeline(
    packing_list_df,
    item_master_df,
    container_df,
    options=None,
    progress_callback: Callable[[str, float, dict], None] = None,
) -> PipelineResult:
    start_time = time.perf_counter()
    settings = get_settings()

    pop_size = options.population_size if options else settings.POPULATION_SIZE
    generations = options.generations if options else settings.GENERATIONS

    if progress_callback:
        progress_callback("parse", 0.05, {"message": "Parsing inputs..."})

    seed = options.seed if options and options.seed is not None else None
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    gap = float(options.tolerance_gap_cm) if options and options.tolerance_gap_cm is not None else settings.TOLERANCE_GAP_CM
    wall_clearance = float(options.container_wall_clearance_cm) if options and getattr(options, "container_wall_clearance_cm", None) is not None else getattr(settings, "CONTAINER_WALL_CLEARANCE_CM", 0.0)

    boxes, container_spec, preview, shipment_type = parse_and_join(
        packing_list_df, item_master_df, container_df, tolerance_gap=gap, wall_clearance=wall_clearance
    )

    if progress_callback:
        progress_callback("sort", 0.1, {"message": "Sorting boxes...", "total_boxes": len(boxes)})

    strategy = _resolve_strategy(options, settings)
    use_static_blocks = strategy.use_static_blocks
    dynamic_blocks = strategy.dynamic_blocks
    ga_level = strategy.ga_level
    group_key = strategy.group_key
    post_explode_compaction = strategy.post_explode_compaction

    container_dims = Dimensions(
        container_spec.usable_length,
        container_spec.usable_width,
        container_spec.usable_height,
    )
    is_lcl = shipment_type == ShipmentType.LCL

    if use_static_blocks:
        from app.solver.legacy.variants import run_variant_ab
        best_individual, placed_bboxes, placed_data, unplaced, current_weight, placed_postures = run_variant_ab(
            boxes=boxes,
            container_spec=container_spec,
            container_dims=container_dims,
            shipment_type=shipment_type,
            pop_size=pop_size,
            generations=generations,
            post_explode_compaction=post_explode_compaction,
            seed=seed,
            progress_callback=progress_callback,
        )

    elif ga_level == "carton":
        from app.solver.legacy.variants import run_variant_c
        best_individual, placed_bboxes, placed_data, unplaced, current_weight, placed_postures = run_variant_c(
            boxes=boxes,
            container_spec=container_spec,
            container_dims=container_dims,
            pop_size=pop_size,
            generations=generations,
            post_explode_compaction=post_explode_compaction,
            seed=seed,
            progress_callback=progress_callback,
        )

    else:
        # Variants D and E: Group-level GA (D: single-carton placement, E: dynamic blocks)
        from app.solver.strategy_variants import form_carton_groups, group_genetic_algorithm, decode_group_individual_dynamic
        carton_groups = form_carton_groups(boxes, group_key=group_key)

        def ga_progress_group(gen: int, best):
            if progress_callback:
                progress_callback(
                    "ga_progress",
                    0.3 + 0.5 * (gen / generations),
                    {
                        "message": f"Generation {gen}/{generations}",
                        "generation": gen,
                        "best_fitness": best.fitness_result.fitness if best.fitness_result else 0,
                    },
                )

        group_best = group_genetic_algorithm(
            groups=carton_groups,
            container_dims=container_dims,
            max_weight=container_spec.max_weight_kg,
            population_size=pop_size,
            generations=generations,
            use_dynamic_blocks=dynamic_blocks,
            progress_callback=ga_progress_group,
            seed=seed,
        )

        placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_group_individual_dynamic(
            group_best, carton_groups, container_dims, container_spec.max_weight_kg, use_dynamic_blocks=dynamic_blocks
        )

        from app.solver.compaction import run_compaction_pass
        if post_explode_compaction:
            placed_bboxes, placed_data, placed_postures, unplaced, current_weight, fitness_res = run_compaction_pass(
                placed_bboxes=placed_bboxes,
                placed_data=placed_data,
                placed_postures=placed_postures,
                unplaced=unplaced,
                container_dims=container_dims,
                max_weight=container_spec.max_weight_kg,
                current_weight=current_weight,
            )
        else:
            from app.solver.fitness import calculate_fitness
            fitness_res = calculate_fitness(placed_bboxes, placed_data, unplaced, container_dims, container_spec.max_weight_kg)

        best_individual = Individual(
            chromosome=list(group_best.posture_genes),
            order_keys=[float(l.order_key) for l in group_best.lots],
            fitness_result=fitness_res,
            placed_bboxes=placed_bboxes,
            placed_data=placed_data,
            unplaced=unplaced,
            current_weight=current_weight,
            placed_postures=placed_postures,
        )

    return _finalize(
        best_individual=best_individual,
        placed_bboxes=placed_bboxes,
        placed_data=placed_data,
        placed_postures=placed_postures,
        unplaced=unplaced,
        container_dims=container_dims,
        container_spec=container_spec,
        boxes=boxes,
        is_lcl=is_lcl,
        options=options,
        start_time=start_time,
        seed=seed,
        progress_callback=progress_callback,
    )