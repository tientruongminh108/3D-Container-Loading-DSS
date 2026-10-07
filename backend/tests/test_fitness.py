"""Unit tests for multi-objective fitness function and anti-fragmentation penalty."""

import pytest
from app.solver.geometry import Dimensions, BoundingBox, Posture
from app.solver.parsing import Box
from app.solver.fitness import calculate_fitness
from app.config import get_settings


def create_box(box_id: str, length: float, width: float, height: float, weight: float = 10.0) -> Box:
    return Box(
        box_id=box_id,
        item_id=f"ITEM_{box_id}",
        po_no="PO_01",
        customer_code="CUST_01",
        customer_sequence=1,
        length_cm=length,
        width_cm=width,
        height_cm=height,
        weight_kg=weight,
        this_way_up=False,
        permitted_postures=[Posture.LWH],
        inflated_length=length,
        inflated_width=width,
        inflated_height=height,
    )


class TestFitnessAntiFragmentation:
    def test_compact_vs_fragmented_layout(self):
        """Layout with cavity trapped behind advancing front must have higher frag_penalty."""
        container_dims = Dimensions(1000.0, 200.0, 200.0)
        max_weight = 20000.0

        b1 = create_box("1", 50.0, 100.0, 100.0)
        b2 = create_box("2", 50.0, 100.0, 100.0)

        # Compact layout: boxes packed side-by-side at x in [0, 50]
        # front_x = 50.0
        compact_bboxes = [
            BoundingBox(0.0, 0.0, 0.0, 50.0, 100.0, 100.0),
            BoundingBox(0.0, 100.0, 0.0, 50.0, 200.0, 100.0),
        ]
        compact_result = calculate_fitness(
            compact_bboxes, [b1, b2], [], container_dims, max_weight
        )

        # Fragmented layout: b1 at [0, 50], b2 spaced far away at [250, 300]
        # front_x = 300.0, leaving a huge hole in between
        frag_bboxes = [
            BoundingBox(0.0, 0.0, 0.0, 50.0, 100.0, 100.0),
            BoundingBox(250.0, 0.0, 0.0, 300.0, 100.0, 100.0),
        ]
        frag_result = calculate_fitness(
            frag_bboxes, [b1, b2], [], container_dims, max_weight
        )

        # Placed volumes are identical
        assert compact_result.placed_volume == frag_result.placed_volume
        # Compact layout has strictly smaller fragmentation penalty
        assert compact_result.frag_penalty < frag_result.frag_penalty
        # Therefore compact layout has higher fitness
        assert compact_result.fitness > frag_result.fitness

    def test_door_anchor_aware_front(self):
        """Door anchor boxes at the far end do not collapse frag_penalty before front reaches them."""
        container_dims = Dimensions(1000.0, 200.0, 200.0)
        max_weight = 20000.0

        b_adv = create_box("adv", 100.0, 200.0, 200.0)
        b_door = create_box("door", 50.0, 50.0, 50.0)

        # Advancing box at x in [0, 100], door anchor at x in [950, 1000]
        bbox_adv = BoundingBox(0.0, 0.0, 0.0, 100.0, 200.0, 200.0, is_door_anchor=False)
        bbox_door = BoundingBox(950.0, 0.0, 0.0, 1000.0, 50.0, 50.0, is_door_anchor=True)

        res = calculate_fitness(
            [bbox_adv, bbox_door], [b_adv, b_door], [], container_dims, max_weight
        )

        # Since bbox_adv (100) has not reached door (950), front_x = 100.0
        # Envelope behind front_x = 100 * 200 * 200 = 4,000,000
        # Volume behind front_x = bbox_adv volume = 100 * 200 * 200 = 4,000,000
        # Wasted behind = 0.0, so frag_penalty = 0.0!
        assert res.frag_penalty == pytest.approx(0.0, abs=1e-6)

    def test_empty_placed_boxes(self):
        """Empty placed list gives 0 frag_penalty."""
        container_dims = Dimensions(1000.0, 200.0, 200.0)
        max_weight = 20000.0
        u = create_box("u1", 100.0, 100.0, 100.0)

        res = calculate_fitness([], [], [(u, "NO_SPACE")], container_dims, max_weight)
        assert res.frag_penalty == 0.0


class TestFitnessUnplacedBlend:
    """A3: unplaced penalty is a normalised volume/count blend aligned with the fill metric."""

    CONTAINER = Dimensions(400.0, 200.0, 200.0)
    MAX_W = 20000.0

    def test_leaving_out_small_carton_beats_leaving_out_large_carton(self):
        big = create_box("big", 100.0, 100.0, 100.0)
        small = create_box("small", 10.0, 10.0, 10.0)
        # Both layouts end at x=300, same unplaced COUNT (1); only the volume differs.
        placed_big = BoundingBox(200.0, 50.0, 0.0, 300.0, 150.0, 100.0)
        placed_small = BoundingBox(290.0, 95.0, 0.0, 300.0, 105.0, 10.0)

        leave_small = calculate_fitness([placed_big], [big], [(small, "no_space")], self.CONTAINER, self.MAX_W)
        leave_big = calculate_fitness([placed_small], [small], [(big, "no_space")], self.CONTAINER, self.MAX_W)

        assert leave_small.unplaced_count == leave_big.unplaced_count == 1
        assert leave_small.unplaced_volume == pytest.approx(10.0 ** 3)
        assert leave_big.unplaced_volume == pytest.approx(100.0 ** 3)
        assert leave_small.fitness > leave_big.fitness

    def test_more_placed_volume_beats_more_placed_count(self):
        large = create_box("large", 200.0, 200.0, 200.0)
        smalls = [create_box(f"s{i}", 20.0, 20.0, 20.0) for i in range(3)]
        one_large = calculate_fitness(
            [BoundingBox(100.0, 0.0, 0.0, 300.0, 200.0, 200.0)], [large],
            [(s, "no_space") for s in smalls], self.CONTAINER, self.MAX_W,
        )
        three_small = calculate_fitness(
            [
                BoundingBox(170.0, 90.0, 0.0, 190.0, 110.0, 20.0),
                BoundingBox(190.0, 90.0, 0.0, 210.0, 110.0, 20.0),
                BoundingBox(210.0, 90.0, 0.0, 230.0, 110.0, 20.0),
            ],
            smalls, [(large, "no_space")], self.CONTAINER, self.MAX_W,
        )
        assert three_small.unplaced_count < one_large.unplaced_count  # old objective preferred this
        assert one_large.fitness > three_small.fitness

    def test_infeasible_penalty_is_a_plain_constant(self):
        settings = get_settings()
        floating = create_box("f", 20.0, 20.0, 20.0)
        floor = create_box("g", 20.0, 20.0, 20.0)
        bb_float = BoundingBox(0.0, 0.0, 50.0, 20.0, 20.0, 70.0)  # no support -> stability violation
        one = calculate_fitness([bb_float], [floating], [], self.CONTAINER, self.MAX_W)
        two = calculate_fitness(
            [bb_float, BoundingBox(100.0, 0.0, 0.0, 120.0, 20.0, 20.0)], [floating, floor], [],
            self.CONTAINER, self.MAX_W,
        )
        assert one.stability_violations == two.stability_violations == 1
        assert one.fitness < -settings.INFEASIBLE_PENALTY + 1.0
        assert two.fitness < -settings.INFEASIBLE_PENALTY + 1.0
        assert abs(one.fitness - two.fitness) < 1.0  # no per-carton component in the penalty
