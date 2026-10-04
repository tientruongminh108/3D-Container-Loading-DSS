from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import copy
import random
import math
import numpy as np

from app.config import get_settings
from app.solver.parsing import Box
from app.solver.block_generation import Block
from app.solver.geometry import (
    Dimensions,
    Position,
    BoundingBox,
    Posture,
    ExtremePoint,
    FLOOR_EPSILON,
    generate_extreme_points,
    sort_extreme_points,
    calculate_contact_ratio,
    calculate_residual_volume,
    check_support_ratio,
)
from app.solver.constraints import (
    check_weight_capacity,
    check_non_overlap,
    check_container_bounds,
    check_corner_clearance,
    check_stackability,
)
from app.solver.placement import _add_box_extreme_points, is_better_tie_break
from app.solver.fitness import calculate_fitness, FitnessResult


def get_geometry_signature(box: Box) -> Tuple:
    """Compute the geometry signature for a box:
    (sorted_dims, this_way_up, weight_rounded_1kg).
    """
    sorted_dims = tuple(sorted([round(box.length_cm, 1), round(box.width_cm, 1), round(box.height_cm, 1)]))
    return (
        sorted_dims,
        box.this_way_up,
        round(box.weight_kg),
    )


@dataclass
class GroupLot:
    group_index: int
    lot_id: str
    carton_count: int
    order_key: float


@dataclass
class CartonGroup:
    group_index: int
    group_id: str
    customer_sequence: int
    signature: Tuple
    cartons: List[Box]
    permitted_postures: List[Posture]


def form_carton_groups(boxes: List[Box], group_key: str = "geometry") -> List[CartonGroup]:
    """Group cartons by either 'item_id' or 'geometry' signature."""
    groups_dict = {}
    order_list = []

    for box in boxes:
        if group_key == "item_id":
            key = box.item_id
        else:  # geometry
            key = get_geometry_signature(box)

        if key not in groups_dict:
            groups_dict[key] = []
            order_list.append(key)
        groups_dict[key].append(box)

    carton_groups = []
    for idx, key in enumerate(order_list):
        cartons = groups_dict[key]
        cust_seq = getattr(cartons[0], "customer_sequence", 0)
        permitted = cartons[0].permitted_postures
        carton_groups.append(
            CartonGroup(
                group_index=idx,
                group_id=f"GRP_{idx}_{key if group_key=='item_id' else 'geom'}",
                customer_sequence=cust_seq,
                signature=key if isinstance(key, tuple) else (key,),
                cartons=cartons,
                permitted_postures=permitted,
            )
        )
    return carton_groups


@dataclass
class GroupIndividual:
    # Posture gene per group
    posture_genes: List[int]
    # Lots representing partitions of groups with order keys
    lots: List[GroupLot]
    fitness_result: Optional[FitnessResult] = None
    placed_bboxes: List[BoundingBox] = field(default_factory=list)
    placed_data: List[Box] = field(default_factory=list)
    unplaced: List[Tuple[Box, str]] = field(default_factory=list)
    current_weight: float = 0.0
    placed_postures: List[Posture] = field(default_factory=list)

    def clone(self) -> "GroupIndividual":
        return GroupIndividual(
            posture_genes=list(self.posture_genes),
            lots=[GroupLot(l.group_index, l.lot_id, l.carton_count, l.order_key) for l in self.lots],
            fitness_result=self.fitness_result,
            placed_bboxes=self.placed_bboxes,
            placed_data=self.placed_data,
            unplaced=self.unplaced,
            current_weight=self.current_weight,
            placed_postures=self.placed_postures,
        )

    def ordered_lots(self, groups: List[CartonGroup], is_lcl: bool = False) -> List[GroupLot]:
        """Return lots in placement order."""
        return sorted(self.lots, key=lambda l: l.order_key)


def create_group_individual(groups: List[CartonGroup]) -> GroupIndividual:
    posture_genes = []
    lots = []
    for g_idx, group in enumerate(groups):
        n_postures = len(group.permitted_postures) if group.permitted_postures else 1
        posture_genes.append(random.randrange(n_postures))
        lots.append(
            GroupLot(
                group_index=g_idx,
                lot_id=f"L_{g_idx}_0",
                carton_count=len(group.cartons),
                order_key=float(g_idx) + random.uniform(-0.1, 0.1),
            )
        )
    return GroupIndividual(posture_genes=posture_genes, lots=lots)


def mutate_group_individual(
    ind: GroupIndividual,
    groups: List[CartonGroup],
    mutation_rate: float,
    split_prob: float = 0.15,
    merge_prob: float = 0.15,
) -> GroupIndividual:
    """Mutate group individual:
    - Flip group posture
    - Nudge lot order keys
    - Split a lot (e.g. 16 -> 10 + 6)
    - Merge lots of the same group
    """
    mutated = ind.clone()

    # 1. Posture mutations
    for g_idx, g in enumerate(groups):
        if random.random() < mutation_rate:
            n_postures = len(g.permitted_postures) if g.permitted_postures else 1
            if n_postures > 1:
                mutated.posture_genes[g_idx] = random.randrange(n_postures)

    # 2. Order key mutations
    for lot in mutated.lots:
        if random.random() < mutation_rate:
            lot.order_key += random.uniform(-1.5, 1.5)

    # 3. Lot Split operator
    if random.random() < split_prob:
        splittable = [l for l in mutated.lots if l.carton_count >= 4]
        if splittable:
            lot_to_split = random.choice(splittable)
            split_at = lot_to_split.carton_count // 2
            rem = lot_to_split.carton_count - split_at
            lot_to_split.carton_count = split_at
            new_lot = GroupLot(
                group_index=lot_to_split.group_index,
                lot_id=f"{lot_to_split.lot_id}_s",
                carton_count=rem,
                order_key=lot_to_split.order_key + random.uniform(0.1, 0.8),
            )
            mutated.lots.append(new_lot)

    # 4. Lot Merge operator
    if random.random() < merge_prob:
        from collections import defaultdict
        group_to_lots = defaultdict(list)
        for lot in mutated.lots:
            group_to_lots[lot.group_index].append(lot)
        mergeable_groups = [g_idx for g_idx, l_list in group_to_lots.items() if len(l_list) >= 2]
        if mergeable_groups:
            g_target = random.choice(mergeable_groups)
            l_list = group_to_lots[g_target]
            # Merge l_list[0] and l_list[1]
            l_list[0].carton_count += l_list[1].carton_count
            mutated.lots.remove(l_list[1])

    return mutated


def crossover_group_individuals(
    parent1: GroupIndividual,
    parent2: GroupIndividual,
    crossover_prob: float = 0.70,
) -> Tuple[GroupIndividual, GroupIndividual]:
    if random.random() >= crossover_prob or len(parent1.posture_genes) <= 1:
        return parent1.clone(), parent2.clone()

    n = len(parent1.posture_genes)
    pt = random.randint(1, n - 1)

    c1_postures = parent1.posture_genes[:pt] + parent2.posture_genes[pt:]
    c2_postures = parent2.posture_genes[:pt] + parent1.posture_genes[pt:]

    c1 = parent1.clone()
    c2 = parent2.clone()
    c1.posture_genes = c1_postures
    c2.posture_genes = c2_postures
    c1.fitness_result = None
    c2.fitness_result = None

    return c1, c2


def compute_free_cuboid_at_ep(
    ep: ExtremePoint,
    placed_bboxes: List[BoundingBox],
    container_dims: Dimensions,
) -> Tuple[float, float, float]:
    """Compute maximal available space (avail_dx, avail_dy, avail_dz) at an extreme point."""
    cL, cW, cH = container_dims.length, container_dims.width, container_dims.height
    avail_dx = max(0.0, cL - ep.x)
    avail_dy = max(0.0, cW - ep.y)
    avail_dz = max(0.0, cH - ep.z)

    # Tighten bounds if there is an obstacle directly in front / side / top
    for pb in placed_bboxes:
        # Obstacle ahead in X (overlaps in Y and Z)
        if pb.min_x >= ep.x - 1e-4:
            if (min(ep.y + avail_dy, pb.max_y) > max(ep.y, pb.min_y) + 1e-4 and
                min(ep.z + avail_dz, pb.max_z) > max(ep.z, pb.min_z) + 1e-4):
                if pb.min_x - ep.x > 1e-4:
                    avail_dx = min(avail_dx, pb.min_x - ep.x)

        # Obstacle to the right in Y (overlaps in X and Z)
        if pb.min_y >= ep.y - 1e-4:
            if (min(ep.x + avail_dx, pb.max_x) > max(ep.x, pb.min_x) + 1e-4 and
                min(ep.z + avail_dz, pb.max_z) > max(ep.z, pb.min_z) + 1e-4):
                if pb.min_y - ep.y > 1e-4:
                    avail_dy = min(avail_dy, pb.min_y - ep.y)

        # Obstacle above in Z (overlaps in X and Y)
        if pb.min_z >= ep.z - 1e-4:
            if (min(ep.x + avail_dx, pb.max_x) > max(ep.x, pb.min_x) + 1e-4 and
                min(ep.y + avail_dy, pb.max_y) > max(ep.y, pb.min_y) + 1e-4):
                if pb.min_z - ep.z > 1e-4:
                    avail_dz = min(avail_dz, pb.min_z - ep.z)

    return avail_dx, avail_dy, avail_dz


def enumerate_candidate_grids(
    rem_count: int,
    avail_dx: float,
    avail_dy: float,
    avail_dz: float,
    carton_dims_infl: Dimensions,
    carton_dims_actual: Dimensions,
    container_width: float,
    ep_y: float,
) -> List[Tuple[int, int, int, int, Dimensions, Dimensions]]:
    """Enumerate targeted candidate grid configurations:
    1. Full width priority: maximizes ny to bridge the container width / gap
    2. Full height priority: maximizes nz for solid vertical columns
    3. Compact cubical grid: maximizes carton absorption k
    4. Single carton fallback: (1, 1, 1)
    """
    dx_c = carton_dims_infl.length
    dy_c = carton_dims_infl.width
    dz_c = carton_dims_infl.height

    if dx_c <= 0 or dy_c <= 0 or dz_c <= 0:
        return []

    max_nx = min(rem_count, max(0, int(avail_dx // dx_c)))
    max_ny = min(rem_count, max(0, int(avail_dy // dy_c)))
    max_nz = min(rem_count, max(0, int(avail_dz // dz_c)))

    if max_nx < 1 or max_ny < 1 or max_nz < 1:
        return []

    candidates: List[Tuple[int, int, int]] = []
    seen = set()

    def add_cand(nx: int, ny: int, nz: int):
        k = nx * ny * nz
        if 1 <= k <= rem_count and 1 <= nx <= max_nx and 1 <= ny <= max_ny and 1 <= nz <= max_nz:
            if (nx, ny, nz) not in seen:
                seen.add((nx, ny, nz))
                candidates.append((nx, ny, nz))

    # 1. Single carton fallback (always present)
    add_cand(1, 1, 1)

    if rem_count >= 2:
        # 2. Width priority: maximize ny first
        ny1 = max_ny
        rem1 = rem_count // ny1
        nz1 = max(1, min(max_nz, rem1))
        nx1 = max(1, min(max_nx, rem1 // nz1))
        add_cand(nx1, ny1, nz1)

        # 3. Height priority: maximize nz first
        nz2 = max_nz
        rem2 = rem_count // nz2
        ny2 = max(1, min(max_ny, rem2))
        nx2 = max(1, min(max_nx, rem2 // ny2))
        add_cand(nx2, ny2, nz2)

        # 4. Length priority (depth): maximize nx first
        nx3 = max_nx
        rem3 = rem_count // nx3
        ny3 = max(1, min(max_ny, rem3))
        nz3 = max(1, min(max_nz, rem3 // ny3))
        add_cand(nx3, ny3, nz3)

        # 5. Full container width if applicable
        if ep_y <= 1e-4:
            needed_ny = int(container_width // dy_c)
            if 1 <= needed_ny <= max_ny:
                rem_w = rem_count // needed_ny
                nz_w = max(1, min(max_nz, rem_w))
                nx_w = max(1, min(max_nx, rem_w // nz_w))
                add_cand(nx_w, needed_ny, nz_w)

    result = []
    for nx, ny, nz in candidates:
        k = nx * ny * nz
        g_infl = Dimensions(nx * dx_c, ny * dy_c, nz * dz_c)
        g_act = Dimensions(nx * carton_dims_actual.length, ny * carton_dims_actual.width, nz * carton_dims_actual.height)
        result.append((k, nx, ny, nz, g_infl, g_act))

    # Sort candidates: multi-carton grids first (highest k), then single carton
    result.sort(key=lambda item: item[0], reverse=True)
    return result


def check_grid_cartons_stackability(
    ep: Position,
    nx: int,
    ny: int,
    dx_c: float,
    dy_c: float,
    c_act: Dimensions,
    cand_unit_wt: float,
    placed_grids: List[Any],
    min_support_ratio: float = 0.60,
) -> bool:
    """Validate that every individual carton on the bottom layer of a candidate grid
    will have sufficient physical support (>= min_support_ratio) and satisfy weight
    hierarchy when placed on the top surfaces of existing placed grids.
    """
    if ep.z <= FLOOR_EPSILON:
        return True

    # Pre-extract physical top surfaces of placed cartons at ep.z
    sup_surfaces = []
    for p_placement, p_cartons, p_posture, p_nx, p_ny, p_nz, p_infl, p_ep in placed_grids:
        if abs(p_placement.max_z - ep.z) <= FLOOR_EPSILON:
            p_rep = p_cartons[0]
            p_c_act = Dimensions(p_rep.length_cm, p_rep.width_cm, p_rep.height_cm).apply_posture(p_posture)
            p_gap_x = max(0.0, p_rep.inflated_length - p_rep.length_cm)
            p_gap_y = max(0.0, p_rep.inflated_width - p_rep.width_cm)
            p_gap = max(p_gap_x, p_gap_y)
            p_dx_c = p_c_act.length + p_gap
            p_dy_c = p_c_act.width + p_gap
            p_unit_wt = p_rep.weight_kg

            for p_ix in range(p_nx):
                for p_iy in range(p_ny):
                    p_x1 = p_ep.x + p_ix * p_dx_c
                    p_y1 = p_ep.y + p_iy * p_dy_c
                    p_x2 = p_x1 + p_c_act.length
                    p_y2 = p_y1 + p_c_act.width
                    sup_surfaces.append((p_x1, p_x2, p_y1, p_y2, p_unit_wt))

    if not sup_surfaces:
        return False

    c_footprint = c_act.length * c_act.width
    if c_footprint <= 0:
        return False
    min_contact = c_footprint * min_support_ratio - 1e-4

    # Check each carton on the bottom face (iz = 0) of the candidate grid
    for ix in range(nx):
        for iy in range(ny):
            c_x1 = ep.x + ix * dx_c
            c_y1 = ep.y + iy * dy_c
            c_x2 = c_x1 + c_act.length
            c_y2 = c_y1 + c_act.width

            contact_area = 0.0
            for p_x1, p_x2, p_y1, p_y2, p_unit_wt in sup_surfaces:
                ox = min(c_x2, p_x2) - max(c_x1, p_x1)
                if ox > 1e-4:
                    oy = min(c_y2, p_y2) - max(c_y1, p_y1)
                    if oy > 1e-4:
                        if cand_unit_wt > p_unit_wt + 1e-3:
                            return False
                        contact_area += ox * oy
                        if contact_area >= min_contact:
                            break
            if contact_area < min_contact:
                return False

    return True


def decode_group_individual_dynamic(
    individual: GroupIndividual,
    groups: List[CartonGroup],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool = False,
    use_dynamic_blocks: bool = True,
) -> Tuple[List[BoundingBox], List[Box], List[Tuple[Box, str]], float, List[Posture]]:
    """Decode a group individual.
    If use_dynamic_blocks is True (Variant E):
      Enumerate grids (nx, ny, nz) at extreme points, validate whole grid,
      minimize residual free volume, and immediately explode into individual cartons.
    If use_dynamic_blocks is False (Variant D):
      Place cartons individually according to the group's posture choice.
    """
    settings = get_settings()
    contact_wt = settings.CONTACT_RATIO_WEIGHT
    residual_wt = settings.RESIDUAL_VOLUME_WEIGHT
    c_vol = container_dims.volume()
    cL, cW, cH = container_dims.length, container_dims.width, container_dims.height
    wall_penalty = getattr(settings, "WALL_FIRST_PENALTY", 2.0)

    placed_grids = []
    placed_grid_bboxes: List[BoundingBox] = []
    placed_grid_data: List[Box] = []
    unplaced: List[Tuple[Box, str]] = []
    current_weight = 0.0

    extreme_points = [ExtremePoint(0, 0, 0)]
    seen_points = {ExtremePoint(0, 0, 0)}
    placement_count = 0

    ordered_lots = individual.ordered_lots(groups, is_lcl=is_lcl)
    # Track carton pointer per group
    group_carton_ptrs = [0] * len(groups)

    for lot in ordered_lots:
        group = groups[lot.group_index]
        ptr = group_carton_ptrs[lot.group_index]
        lot_cartons = group.cartons[ptr : ptr + lot.carton_count]
        group_carton_ptrs[lot.group_index] += len(lot_cartons)

        if not lot_cartons:
            continue

        rem_in_lot = list(lot_cartons)
        rep_box = rem_in_lot[0]
        permitted = group.permitted_postures or [Posture.LWH]

        # Selected posture from chromosome
        pref_posture_idx = individual.posture_genes[lot.group_index] % len(permitted)
        pref_posture = permitted[pref_posture_idx]
        postures_to_try = [pref_posture] + [p for p in permitted if p != pref_posture]

        while rem_in_lot:
            rem_count = len(rem_in_lot)
            rep_box = rem_in_lot[0]
            sorted_eps = sort_extreme_points(extreme_points)[:30]

            best_placement = None
            best_score = -float('inf')
            best_grid_spec = None  # (k, nx, ny, nz, posture, dims_infl, dims_act)
            best_ep = None

            # Wall front for wall-first overrun calculation
            front = max((b.max_x for b in placed_grid_bboxes), default=0.0)

            min_c_dim = min(rep_box.inflated_length, rep_box.inflated_width, rep_box.inflated_height)
            for ep in sorted_eps:
                avail_dx, avail_dy, avail_dz = compute_free_cuboid_at_ep(ep, placed_grid_bboxes, container_dims)
                if avail_dx < min_c_dim or avail_dy < min_c_dim or avail_dz < min_c_dim:
                    continue

                for posture in postures_to_try:
                    c_act = Dimensions(rep_box.length_cm, rep_box.width_cm, rep_box.height_cm).apply_posture(posture)
                    # Container horizontal axes (X, Y) receive tolerance_gap padding;
                    # Container vertical axis (Z) must NOT receive tolerance_gap padding
                    # to prevent floating cartons and vertical height mismatches.
                    gap_x = max(0.0, rep_box.inflated_length - rep_box.length_cm)
                    gap_y = max(0.0, rep_box.inflated_width - rep_box.width_cm)
                    gap = max(gap_x, gap_y)
                    c_infl = Dimensions(c_act.length + gap, c_act.width + gap, c_act.height)

                    if not use_dynamic_blocks:
                        # Variant D: single carton placement only
                        candidate_grids = [(1, 1, 1, 1, c_infl, c_act)]
                    else:
                        # Variant E: enumerate dynamic grids + single carton fallback
                        candidate_grids = enumerate_candidate_grids(
                            rem_count, avail_dx, avail_dy, avail_dz,
                            c_infl, c_act, cW, ep.y,
                        )

                    for k, nx, ny, nz, g_infl, g_act in candidate_grids:
                        grid_weight = k * rep_box.weight_kg
                        if current_weight + grid_weight > max_weight + 1e-3:
                            continue

                        cand_bbox = BoundingBox(
                            ep.x, ep.y, ep.z,
                            ep.x + g_infl.length,
                            ep.y + g_infl.width,
                            ep.z + g_infl.height,
                        )

                        # Container bounds check
                        if not check_container_bounds(cand_bbox, container_dims):
                            continue

                        # Corner clearance check
                        if not check_corner_clearance(cand_bbox, container_dims):
                            continue

                        # Overlap check
                        if not check_non_overlap(cand_bbox, placed_grid_bboxes):
                            continue

                        # Stackability check (support ratio >= 0.60 + weight hierarchy)
                        if not check_stackability(cand_bbox, placed_grid_bboxes, rep_box, placed_grid_data):
                            continue

                        # Per-carton physical stackability check
                        if not check_grid_cartons_stackability(
                            ep, nx, ny, c_act.length + gap, c_act.width + gap,
                            c_act, rep_box.weight_kg, placed_grids
                        ):
                            continue

                        # All constraints passed! Compute scoring
                        contact_ratio = calculate_contact_ratio(cand_bbox, placed_grid_bboxes, container_dims)
                        cuboid_vol = avail_dx * avail_dy * avail_dz
                        grid_vol = g_infl.volume()
                        residual_in_space = max(0.0, cuboid_vol - grid_vol)
                        res_score = -(residual_in_space / c_vol)

                        # Y-coverage bonus (reward spanning the container width or wall)
                        y_cov_bonus = 0.0
                        if cand_bbox.min_y <= 1e-4 and cand_bbox.max_y >= cW - 1e-4:
                            y_cov_bonus = 4.0
                        elif g_infl.width >= avail_dy - 1e-4:
                            y_cov_bonus = 2.0

                        # Cartons absorbed bonus (rewards multi-box consolidation)
                        absorbed_bonus = 1.5 * (k / rem_count) if rem_count > 0 else 0.0

                        # Preference for chosen chromosome posture
                        posture_pref_bonus = 0.5 if posture == pref_posture else 0.0

                        score = (
                            contact_wt * contact_ratio
                            + y_cov_bonus
                            + absorbed_bonus
                            + posture_pref_bonus
                            + residual_wt * res_score
                        )

                        # Overrun calculation (Wall-First overrun penalty for FCL)
                        c_overrun = max(0.0, cand_bbox.max_x - front)
                        effective_score = score - wall_penalty * (c_overrun / cL)

                        if effective_score > best_score:
                            best_score = effective_score
                            best_ep = ep
                            best_grid_spec = (k, nx, ny, nz, posture, g_infl, g_act)
                            best_placement = cand_bbox

                if best_placement and best_grid_spec and best_grid_spec[0] == rem_count:
                    # Found a placement that absorbs all remaining cartons in lot!
                    break

            if best_placement is None or best_grid_spec is None:
                # No space found for any grid or single carton at any extreme point
                for b in rem_in_lot:
                    unplaced.append((b, "no_space"))
                break

            # Placement found! Record grid placement
            k, nx, ny, nz, chosen_posture, g_infl, g_act = best_grid_spec
            cartons_to_place = rem_in_lot[:k]
            rem_in_lot = rem_in_lot[k:]

            placed_grids.append((best_placement, cartons_to_place, chosen_posture, nx, ny, nz, g_infl, best_ep))
            placed_grid_bboxes.append(best_placement)
            placed_grid_data.append(rep_box)
            current_weight += sum(c.weight_kg for c in cartons_to_place)

            placement_count += 1
            extreme_points = _add_box_extreme_points(
                best_placement, extreme_points, seen_points, placed_grid_bboxes,
                container_dims, placement_count,
            )

    # Explode all placed grids into individual cartons for the final solution
    placed_bboxes: List[BoundingBox] = []
    placed_data: List[Box] = []
    placed_postures: List[Posture] = []

    for best_placement, cartons_to_place, chosen_posture, nx, ny, nz, g_infl, best_ep in placed_grids:
        rep_carton = cartons_to_place[0]
        c_act = Dimensions(rep_carton.length_cm, rep_carton.width_cm, rep_carton.height_cm).apply_posture(chosen_posture)
        gap_x = max(0.0, rep_carton.inflated_length - rep_carton.length_cm)
        gap_y = max(0.0, rep_carton.inflated_width - rep_carton.width_cm)
        gap = max(gap_x, gap_y)
        dx_c = c_act.length + gap
        dy_c = c_act.width + gap
        dz_c = c_act.height
        c_idx = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if c_idx >= len(cartons_to_place):
                        break
                    carton = cartons_to_place[c_idx]
                    c_idx += 1
                    c_x = best_ep.x + ix * dx_c
                    c_y = best_ep.y + iy * dy_c
                    c_z = best_ep.z + iz * dz_c
                    single_bbox = BoundingBox(
                        c_x, c_y, c_z,
                        c_x + dx_c, c_y + dy_c, c_z + dz_c,
                    )
                    placed_bboxes.append(single_bbox)
                    placed_data.append(carton)
                    placed_postures.append(chosen_posture)

    return placed_bboxes, placed_data, unplaced, current_weight, placed_postures


def evaluate_group_individual(
    ind: GroupIndividual,
    groups: List[CartonGroup],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool = False,
    use_dynamic_blocks: bool = False,
) -> GroupIndividual:
    if ind.fitness_result is not None:
        return ind
    placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_group_individual_dynamic(
        ind, groups, container_dims, max_weight, is_lcl, use_dynamic_blocks=use_dynamic_blocks
    )
    fitness_res = calculate_fitness(
        placed_bboxes, placed_data, unplaced, container_dims, max_weight
    )
    ind.placed_bboxes = placed_bboxes
    ind.placed_data = placed_data
    ind.unplaced = unplaced
    ind.current_weight = current_weight
    ind.placed_postures = placed_postures
    ind.fitness_result = fitness_res
    return ind


def group_genetic_algorithm(
    groups: List[CartonGroup],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool = False,
    population_size: int = 60,
    generations: int = 100,
    use_dynamic_blocks: bool = False,
    progress_callback: Optional[Any] = None,
) -> GroupIndividual:
    """Genetic Algorithm operating at the Group level (Variants D and E)."""
    settings = get_settings()
    pop = [create_group_individual(groups) for _ in range(population_size)]

    for ind in pop:
        evaluate_group_individual(ind, groups, container_dims, max_weight, is_lcl, use_dynamic_blocks)

    pop.sort(key=lambda x: x.fitness_result.fitness if x.fitness_result else -float('inf'), reverse=True)
    best_ind = pop[0].clone()
    patience = settings.EARLY_STOP_PATIENCE
    stagnant_gens = 0

    for gen in range(1, generations + 1):
        # Elitism
        n_elite = max(1, int(settings.ELITE_FRACTION * population_size))
        new_pop = [ind.clone() for ind in pop[:n_elite]]

        # Selection & Reproduction
        while len(new_pop) < population_size:
            p1 = random.choice(pop[:max(2, population_size // 2)])
            p2 = random.choice(pop[:max(2, population_size // 2)])
            c1, c2 = crossover_group_individuals(p1, p2, settings.CROSSOVER_PROBABILITY)
            c1 = mutate_group_individual(c1, groups, settings.MUTATION_RATE_BASE)
            new_pop.append(c1)
            if len(new_pop) < population_size:
                c2 = mutate_group_individual(c2, groups, settings.MUTATION_RATE_BASE)
                new_pop.append(c2)

        for ind in new_pop[n_elite:]:
            evaluate_group_individual(ind, groups, container_dims, max_weight, is_lcl, use_dynamic_blocks)

        new_pop.sort(key=lambda x: x.fitness_result.fitness if x.fitness_result else -float('inf'), reverse=True)
        pop = new_pop

        if pop[0].fitness_result and pop[0].fitness_result.fitness > best_ind.fitness_result.fitness + settings.MIN_IMPROVEMENT:
            best_ind = pop[0].clone()
            stagnant_gens = 0
        else:
            stagnant_gens += 1

        if progress_callback:
            progress_callback(gen, best_ind)

        if stagnant_gens >= patience:
            break

    return best_ind
