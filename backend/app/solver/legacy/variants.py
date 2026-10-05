from typing import List, Callable, Optional, Tuple, Any
from app.solver.parsing import Box, ContainerSpec, ShipmentType
from app.solver.sorting import initial_sort, resort_after_blocks
from app.solver.block_generation import build_blocks, Block
from app.solver.ga import genetic_algorithm, Individual
from app.solver.placement import decode_chromosome
from app.solver.geometry import Dimensions, BoundingBox


def run_variant_ab(
    boxes: List[Box],
    container_spec: ContainerSpec,
    container_dims: Dimensions,
    shipment_type: ShipmentType,
    pop_size: int,
    generations: int,
    post_explode_compaction: bool,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float, dict], None]] = None,
):
    """Legacy Variant A (Static blocks) & Variant B (Static blocks + post-explode compaction)."""
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
        population_size=pop_size,
        generations=generations,
        progress_callback=ga_progress,
        seed=seed,
    )

    if progress_callback:
        progress_callback("decode", 0.95, {"message": "Building final solution..."})

    order = best_individual.placement_order(all_units)
    ordered_units = [all_units[i] for i in order]
    ordered_chromosome = [best_individual.chromosome[i] for i in order]
    placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_chromosome(
        ordered_chromosome, ordered_units, container_dims, container_spec.max_weight_kg
    )
    for idx, orig_i in enumerate(order):
        best_individual.chromosome[orig_i] = ordered_chromosome[idx]

    from app.solver.compaction import run_compaction_pass

    if post_explode_compaction:
        from app.solver.pipeline import _explode_to_units
        c_bboxes, c_data, c_postures, exploded_unplaced = _explode_to_units(
            placed_data, placed_bboxes, placed_postures, unplaced
        )
        placed_bboxes, placed_data, placed_postures, unplaced, current_weight, fitness_res = run_compaction_pass(
            placed_bboxes=c_bboxes,
            placed_data=c_data,
            placed_postures=c_postures,
            unplaced=exploded_unplaced,
            container_dims=container_dims,
            max_weight=container_spec.max_weight_kg,
            current_weight=current_weight,
        )
    else:
        placed_bboxes, placed_data, placed_postures, unplaced, current_weight, fitness_res = run_compaction_pass(
            placed_bboxes=placed_bboxes,
            placed_data=placed_data,
            placed_postures=placed_postures,
            unplaced=unplaced,
            container_dims=container_dims,
            max_weight=container_spec.max_weight_kg,
            current_weight=current_weight,
        )

    best_individual.placed_bboxes = placed_bboxes
    best_individual.placed_data = placed_data
    best_individual.unplaced = unplaced
    best_individual.current_weight = current_weight
    best_individual.placed_postures = placed_postures
    best_individual.fitness_result = fitness_res

    return best_individual, placed_bboxes, placed_data, unplaced, current_weight, placed_postures


def run_variant_c(
    boxes: List[Box],
    container_spec: ContainerSpec,
    container_dims: Dimensions,
    pop_size: int,
    generations: int,
    post_explode_compaction: bool,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float, dict], None]] = None,
):
    """Legacy Variant C: No static blocks, naive sort by item_id, carton-level GA."""
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
        population_size=pop_size,
        generations=generations,
        progress_callback=ga_progress,
        seed=seed,
    )

    order = best_individual.placement_order(all_units)
    ordered_units = [all_units[i] for i in order]
    ordered_chromosome = [best_individual.chromosome[i] for i in order]
    placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_chromosome(
        ordered_chromosome, ordered_units, container_dims, container_spec.max_weight_kg
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

    best_individual.placed_bboxes = placed_bboxes
    best_individual.placed_data = placed_data
    best_individual.unplaced = unplaced
    best_individual.current_weight = current_weight
    best_individual.placed_postures = placed_postures
    best_individual.fitness_result = fitness_res

    return best_individual, placed_bboxes, placed_data, unplaced, current_weight, placed_postures
