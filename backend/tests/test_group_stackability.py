"""A1 regression: grid-level stackability in the group decoder.

The decoder used to call ``constraints.check_stackability`` with ONE representative
Box per supporter grid and no postures, so every supporter looked like a single
carton in posture LWH while the candidate footprint was the whole grid.  Multi-carton
grids could therefore (almost) never be stacked on other grids.  The per-carton check
``check_grid_cartons_stackability`` is the authoritative one.
"""
import pytest

from app.solver.geometry import BoundingBox, Dimensions, ExtremePoint, Position, Posture
from app.solver.parsing import Box
from app.solver.strategy_variants import (
    GroupIndividual,
    GroupLot,
    check_grid_cartons_stackability,
    decode_group_individual_dynamic,
    form_carton_groups,
)


def _box(box_id, item_id, l, w, h, weight, upright=True):
    return Box(
        box_id=box_id, item_id=item_id, po_no="PO", customer_code=None, customer_sequence=0,
        length_cm=l, width_cm=w, height_cm=h, weight_kg=weight, this_way_up=upright,
        permitted_postures=[Posture.LWH, Posture.WLH] if upright else list(Posture),
        inflated_length=l, inflated_width=w, inflated_height=h,
    )


def _supporter_grid(nx, ny, weight, c=20.0, h=10.0):
    """A placed (nx x ny x 1) grid of identical cartons with its bottom corner at the origin."""
    cartons = [_box(f"S{i}", "SUP", c, c, h, weight) for i in range(nx * ny)]
    bbox = BoundingBox(0, 0, 0, nx * c, ny * c, h)
    infl = Dimensions(nx * c, ny * c, h)
    return (bbox, cartons, Posture.LWH, nx, ny, 1, infl, ExtremePoint(0, 0, 0))


class TestGridCartonsStackability:
    def test_2x2_on_3x3_equal_weight_is_accepted(self):
        sup = _supporter_grid(3, 3, weight=10.0)
        assert check_grid_cartons_stackability(
            ExtremePoint(0, 0, 10.0), 2, 2, 20.0, 20.0, Dimensions(20, 20, 10), 10.0, [sup]
        )

    def test_2x2_on_3x3_heavier_supporter_is_accepted(self):
        sup = _supporter_grid(3, 3, weight=15.0)
        assert check_grid_cartons_stackability(
            ExtremePoint(0, 0, 10.0), 2, 2, 20.0, 20.0, Dimensions(20, 20, 10), 10.0, [sup]
        )

    def test_2x2_on_lighter_supporter_is_rejected(self):
        sup = _supporter_grid(3, 3, weight=8.0)
        assert not check_grid_cartons_stackability(
            ExtremePoint(0, 0, 10.0), 2, 2, 20.0, 20.0, Dimensions(20, 20, 10), 10.0, [sup]
        )

    def test_insufficient_per_carton_support_is_rejected(self):
        sup = _supporter_grid(3, 3, weight=10.0)  # footprint x,y in [0, 60]
        # Grid starts at x=50: the first column of cartons spans [50, 70] -> only 10/20 = 50% supported.
        assert not check_grid_cartons_stackability(
            ExtremePoint(50.0, 0, 10.0), 2, 2, 20.0, 20.0, Dimensions(20, 20, 10), 10.0, [sup]
        )

    def test_exactly_sixty_percent_support_is_accepted(self):
        sup = _supporter_grid(3, 3, weight=10.0)
        # Column spans [48, 68]: 12/20 = 60% in x, full in y -> exactly the threshold.
        assert check_grid_cartons_stackability(
            ExtremePoint(48.0, 0, 10.0), 1, 1, 20.0, 20.0, Dimensions(20, 20, 10), 10.0, [sup]
        )


def _decode(a_weight, b_weight):
    # Container 60 x 60 x 100; A cartons are 60 high so the only multi-carton grid for A
    # is a 3x3x1 slab that fills the whole floor; B must therefore stack on top of it.
    a = [_box(f"A{i}", "A", 20, 20, 60, a_weight) for i in range(9)]
    b = [_box(f"B{i}", "B", 20, 20, 20, b_weight) for i in range(4)]
    groups = form_carton_groups(a + b, group_key="item_id")
    ind = GroupIndividual(
        posture_genes=[0, 0],
        lots=[GroupLot(0, "L_0", 9, 0.0), GroupLot(1, "L_1", 4, 1.0)],
    )
    return decode_group_individual_dynamic(ind, groups, Dimensions(60, 60, 100), 1e6, use_dynamic_blocks=True)


class TestGroupDecoderStacksGridsOnGrids:
    def test_multi_carton_grid_is_stacked_on_multi_carton_grid(self):
        bboxes, data, unplaced, _, _ = _decode(a_weight=10.0, b_weight=10.0)
        assert unplaced == []
        assert len(data) == 13
        assert sum(1 for bb in bboxes if bb.min_z >= 60.0 - 1e-6) == 4

    def test_grid_is_not_stacked_on_lighter_supporter(self):
        bboxes, data, unplaced, _, _ = _decode(a_weight=10.0, b_weight=12.0)
        assert len(data) == 9
        assert len(unplaced) == 4
        assert all(bb.min_z <= 1e-6 for bb in bboxes)
