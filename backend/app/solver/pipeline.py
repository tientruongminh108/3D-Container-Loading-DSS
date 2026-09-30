from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass
from app.config import get_settings
from app.solver.parsing import parse_and_join, Box, ContainerSpec, PackingListPreview, ShipmentType
from app.solver.sorting import initial_sort, resort_after_blocks
from app.solver.block_generation import build_blocks, Block
from app.solver.ga import genetic_algorithm, Individual
from app.solver.placement import decode_chromosome, place_blocks_greedy
from app.solver.output import build_run_result
from app.solver.geometry import Dimensions, BoundingBox
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


def run_pipeline(
    packing_list_df,
    item_master_df,
    container_df,
    options=None,
    progress_callback: Callable[[str, float, dict], None] = None,
) -> PipelineResult:
    settings = get_settings()

    pop_size = options.population_size if options else settings.POPULATION_SIZE
    generations = options.generations if options else settings.GENERATIONS

    if progress_callback:
        progress_callback("parse", 0.05, {"message": "Parsing inputs..."})

    gap = float(options.tolerance_gap_cm) if options and options.tolerance_gap_cm is not None else settings.TOLERANCE_GAP_CM

    boxes, container_spec, preview, shipment_type = parse_and_join(
        packing_list_df, item_master_df, container_df, tolerance_gap=gap
    )

    if progress_callback:
        progress_callback("sort", 0.1, {"message": "Sorting boxes...", "total_boxes": len(boxes)})

    use_static_blocks = (
        options.use_static_blocks
        if options and options.use_static_blocks is not None
        else settings.USE_STATIC_BLOCKS
    )
    group_key = (
        options.group_key
        if options and options.group_key is not None
        else settings.GROUP_KEY
    )
    ga_level = (
        options.ga_level
        if options and options.ga_level is not None
        else settings.GA_LEVEL
    )
    dynamic_blocks = (
        options.dynamic_blocks
        if options and options.dynamic_blocks is not None
        else settings.DYNAMIC_BLOCKS
    )
    post_explode_compaction = (
        options.post_explode_compaction
        if options and options.post_explode_compaction is not None
        else settings.POST_EXPLODE_COMPACTION
    )

    container_dims = Dimensions(
        container_spec.usable_length,
        container_spec.usable_width,
        container_spec.usable_height,
    )
    is_lcl = shipment_type == ShipmentType.LCL

    # If strategy options are not explicitly specified, auto-select the best validated strategy:
    # - LCL shipments: Variant B (Static Blocks + Post-Explode Compaction) protects strict LIFO customer separation
    # - FCL shipments: Variant E_PEC (Dynamic Blocks + Post-Explode Compaction) combines B & E for optimal ~80% fill
    if options is None or (options.use_static_blocks is None and options.dynamic_blocks is None):
        if is_lcl:
            use_static_blocks = True
            dynamic_blocks = False
            ga_level = "carton"
            group_key = "item_id"
            post_explode_compaction = True
        else:
            use_static_blocks = False
            dynamic_blocks = True
            ga_level = "group"
            group_key = "geometry"
            post_explode_compaction = True

    if use_static_blocks:
        # Variants A and B: Static Blocks
        sorted_boxes = initial_sort(boxes, shipment_type.value)

        if progress_callback:
            progress_callback("blocks", 0.2, {"message": "Generating blocks..."})

        blocks, leftover = build_blocks(
            sorted_boxes,
            container_spec.usable_length,
            container_spec.usable_width,
            container_spec.usable_height,
        )

        all_units = blocks + leftover
        all_units = resort_after_blocks(all_units, shipment_type.value)

        if progress_callback:
            progress_callback("ga_start", 0.3, {"message": "Starting Genetic Algorithm...", "units": len(all_units)})

        def ga_progress(gen: int, best: Individual):
            if progress_callback:
                progress_callback(
                    "ga_progress",
                    0.3 + 0.5 * (gen / generations),
                    {
                        "message": f"Generation {gen}/{generations}",
                        "generation": gen,
                        "best_fitness": best.fitness_result.fitness if best.fitness_result else 0,
                        "placed": len(best.placed_data) if best.placed_data else 0,
                        "unplaced": len(best.unplaced) if best.unplaced else 0,
                    },
                )

        best_individual = genetic_algorithm(
            units=all_units,
            container_dims=container_dims,
            max_weight=container_spec.max_weight_kg,
            is_lcl=is_lcl,
            population_size=pop_size,
            generations=generations,
            progress_callback=ga_progress,
        )

        if progress_callback:
            progress_callback("decode", 0.95, {"message": "Building final solution..."})

        order = best_individual.placement_order(all_units)
        ordered_units = [all_units[i] for i in order]
        ordered_chromosome = [best_individual.chromosome[i] for i in order]
        placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_chromosome(
            ordered_chromosome, ordered_units, container_dims, container_spec.max_weight_kg, is_lcl
        )
        for idx, orig_i in enumerate(order):
            best_individual.chromosome[orig_i] = ordered_chromosome[idx]

        from app.solver.compaction import run_compaction_pass

        if post_explode_compaction:
            # Variant B: explode blocks into cartons FIRST, then run 4 compaction passes and insertion
            from app.solver.output import explode_blocks
            exploded_cartons = explode_blocks(placed_data, placed_bboxes, placed_postures)
            c_bboxes = []
            c_data = []
            c_postures = []
            for c in exploded_cartons:
                c_inf = Dimensions(c.inflated_length, c.inflated_width, c.inflated_height).apply_posture(c.posture)
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

            placed_bboxes, placed_data, placed_postures, unplaced, current_weight, fitness_res = run_compaction_pass(
                placed_bboxes=c_bboxes,
                placed_data=c_data,
                placed_postures=c_postures,
                unplaced=exploded_unplaced,
                container_dims=container_dims,
                max_weight=container_spec.max_weight_kg,
                is_lcl=is_lcl,
                current_weight=current_weight,
            )
        else:
            # Variant A: Baseline compaction on block bboxes
            placed_bboxes, placed_data, placed_postures, unplaced, current_weight, fitness_res = run_compaction_pass(
                placed_bboxes=placed_bboxes,
                placed_data=placed_data,
                placed_postures=placed_postures,
                unplaced=unplaced,
                container_dims=container_dims,
                max_weight=container_spec.max_weight_kg,
                is_lcl=is_lcl,
                current_weight=current_weight,
            )

        best_individual.placed_bboxes = placed_bboxes
        best_individual.placed_data = placed_data
        best_individual.unplaced = unplaced
        best_individual.current_weight = current_weight
        best_individual.placed_postures = placed_postures
        best_individual.fitness_result = fitness_res

    elif ga_level == "carton":
        # Variant C: No static blocks, naive sort by item_id, carton-level GA
        if is_lcl:
            # Customer sequence descending, then same item_id contiguous
            sorted_boxes = sorted(
                boxes,
                key=lambda b: (-b.customer_sequence, b.item_id, -(b.length_cm * b.width_cm * b.height_cm), -b.weight_kg, b.box_id),
            )
        else:
            sorted_boxes = sorted(
                boxes,
                key=lambda b: (b.item_id, -(b.length_cm * b.width_cm * b.height_cm), -b.weight_kg, b.box_id),
            )

        all_units = sorted_boxes

        def ga_progress(gen: int, best: Individual):
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

        best_individual = genetic_algorithm(
            units=all_units,
            container_dims=container_dims,
            max_weight=container_spec.max_weight_kg,
            is_lcl=is_lcl,
            population_size=pop_size,
            generations=generations,
            progress_callback=ga_progress,
        )

        order = best_individual.placement_order(all_units)
        ordered_units = [all_units[i] for i in order]
        ordered_chromosome = [best_individual.chromosome[i] for i in order]
        placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_chromosome(
            ordered_chromosome, ordered_units, container_dims, container_spec.max_weight_kg, is_lcl
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
                is_lcl=is_lcl,
                current_weight=current_weight,
            )
        else:
            from app.solver.fitness import calculate_fitness
            fitness_res = calculate_fitness(placed_bboxes, placed_data, unplaced, container_dims, container_spec.max_weight_kg)

        best_individual.placed_bboxes = placed_bboxes
        best_individual.placed_data = placed_data
        best_individual.unplaced = unplaced
        best_individual.current_weight = current_weight
        best_individual.placed_postures = placed_postures
        best_individual.fitness_result = fitness_res

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
            is_lcl=is_lcl,
            population_size=pop_size,
            generations=generations,
            use_dynamic_blocks=dynamic_blocks,
            progress_callback=ga_progress_group,
        )

        placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_group_individual_dynamic(
            group_best, carton_groups, container_dims, container_spec.max_weight_kg, is_lcl, use_dynamic_blocks=dynamic_blocks
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
                is_lcl=is_lcl,
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

    # Separate placed blocks from placed individual boxes
    placed_blocks = []
    placed_individual_boxes = []
    for unit in placed_data:
        if isinstance(unit, Block):
            placed_blocks.append(unit)
        else:
            placed_individual_boxes.append(unit)

    # Unplaced blocks and boxes from final solution
    final_unplaced_blocks = [u for u, _ in unplaced if isinstance(u, Block)]
    final_unplaced_boxes = [u for u, _ in unplaced if not isinstance(u, Block)]

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