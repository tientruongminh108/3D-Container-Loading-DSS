"""Tests for Phase 3: Wall-first placement rules (A, B, C) and tie-breaking."""

import pytest
from app.solver.geometry import Dimensions, Position, BoundingBox, Posture
from app.solver.parsing import Box
from app.solver.placement import is_better_tie_break, find_best_placement
from app.config import get_settings


def test_is_better_tie_break_zy():
    best_pos = Position(100.0, 50.0, 50.0)

    # 1. Smaller X always wins
    assert is_better_tie_break(90.0, 60.0, 60.0, best_pos, "ZY") is True
    assert is_better_tie_break(110.0, 40.0, 40.0, best_pos, "ZY") is False

    # 2. Equal X, secondary "ZY": lower Z wins
    assert is_better_tie_break(100.0, 60.0, 40.0, best_pos, "ZY") is True
    assert is_better_tie_break(100.0, 40.0, 60.0, best_pos, "ZY") is False

    # 3. Equal X, equal Z: smaller Y wins
    assert is_better_tie_break(100.0, 40.0, 50.0, best_pos, "ZY") is True
    assert is_better_tie_break(100.0, 60.0, 50.0, best_pos, "ZY") is False


def test_is_better_tie_break_yz():
    best_pos = Position(100.0, 50.0, 50.0)

    # 1. Smaller X always wins
    assert is_better_tie_break(90.0, 60.0, 60.0, best_pos, "YZ") is True

    # 2. Equal X, secondary "YZ": smaller Y wins
    assert is_better_tie_break(100.0, 40.0, 60.0, best_pos, "YZ") is True
    assert is_better_tie_break(100.0, 60.0, 40.0, best_pos, "YZ") is False

    # 3. Equal X, equal Y: lower Z wins
    assert is_better_tie_break(100.0, 50.0, 40.0, best_pos, "YZ") is True
    assert is_better_tie_break(100.0, 50.0, 60.0, best_pos, "YZ") is False


def test_find_best_placement_rules_a_vs_b(monkeypatch):
    container_dims = Dimensions(1000.0, 200.0, 200.0)
    # Existing box at [0, 100] x [0, 50] x [0, 50] -> front = 100
    placed_bbox = BoundingBox(0.0, 0.0, 0.0, 100.0, 50.0, 50.0)
    placed_box = Box(
        box_id="1", item_id="A", po_no="PO1", customer_code="C1", customer_sequence=1,
        length_cm=100.0, width_cm=50.0, height_cm=50.0,
        weight_kg=10.0, this_way_up=False, permitted_postures=[Posture.LWH],
        inflated_length=100.0, inflated_width=50.0, inflated_height=50.0
    )

    # Candidate box to place
    cand_box = Box(
        box_id="2", item_id="B", po_no="PO1", customer_code="C1", customer_sequence=1,
        length_cm=60.0, width_cm=50.0, height_cm=50.0,
        weight_kg=5.0, this_way_up=False, permitted_postures=[Posture.LWH],
        inflated_length=60.0, inflated_width=50.0, inflated_height=50.0
    )

    # Two extreme points:
    # EP1: (0, 50, 0) -> candidate max_x = 0 + 60 = 60 <= front (100) -> overrun = 0.
    # EP2: (100, 0, 0) -> candidate max_x = 100 + 60 = 160 > front (100) -> overrun = 60.
    # Notice: At EP2, candidate touches placed_box on rear face (X contact), so EP2 has higher contact score.
    # At EP1, candidate touches placed_box on side face (Y contact).

    from app.solver.geometry import ExtremePoint
    eps = [ExtremePoint(100.0, 0.0, 0.0), ExtremePoint(0.0, 50.0, 0.0)]

    # Rule A: should prefer higher score / tie-break
    settings = get_settings()
    monkeypatch.setattr(settings, "WALL_FIRST_RULE", "A")
    res_a, _ = find_best_placement(
        cand_box, [placed_bbox], [placed_box], container_dims,
        current_weight=10.0, max_weight=1000.0, is_lcl=True, extreme_points=eps
    )

    # Rule B: strict wall-first must choose EP1 because overrun=0 < overrun=60
    monkeypatch.setattr(settings, "WALL_FIRST_RULE", "B")
    res_b, _ = find_best_placement(
        cand_box, [placed_bbox], [placed_box], container_dims,
        current_weight=10.0, max_weight=1000.0, is_lcl=True, extreme_points=eps
    )

    assert res_b is not None
    assert res_b.position.x == 0.0  # EP1 chosen because overrun is 0!


def test_find_best_placement_rule_c(monkeypatch):
    container_dims = Dimensions(1000.0, 200.0, 200.0)
    placed_bbox = BoundingBox(0.0, 0.0, 0.0, 100.0, 50.0, 50.0)
    placed_box = Box(
        box_id="1", item_id="A", po_no="PO1", customer_code="C1", customer_sequence=1,
        length_cm=100.0, width_cm=50.0, height_cm=50.0,
        weight_kg=10.0, this_way_up=False, permitted_postures=[Posture.LWH],
        inflated_length=100.0, inflated_width=50.0, inflated_height=50.0
    )

    cand_box = Box(
        box_id="2", item_id="B", po_no="PO1", customer_code="C1", customer_sequence=1,
        length_cm=60.0, width_cm=50.0, height_cm=50.0,
        weight_kg=5.0, this_way_up=False, permitted_postures=[Posture.LWH],
        inflated_length=60.0, inflated_width=50.0, inflated_height=50.0
    )

    from app.solver.geometry import ExtremePoint
    eps = [ExtremePoint(100.0, 0.0, 0.0), ExtremePoint(0.0, 50.0, 0.0)]

    settings = get_settings()
    monkeypatch.setattr(settings, "WALL_FIRST_RULE", "C")
    # With a high penalty weight for overrun, EP1 (overrun=0) must be chosen over EP2 (overrun=60)
    monkeypatch.setattr(settings, "WALL_FIRST_WEIGHT", 10.0)
    res_c, _ = find_best_placement(
        cand_box, [placed_bbox], [placed_box], container_dims,
        current_weight=10.0, max_weight=1000.0, is_lcl=True, extreme_points=eps
    )
    assert res_c is not None
    assert res_c.position.x == 0.0


def test_find_best_placement_secondary_order(monkeypatch):
    container_dims = Dimensions(1000.0, 200.0, 200.0)
    cand_box = Box(
        box_id="1", item_id="A", po_no="PO1", customer_code="C1", customer_sequence=1,
        length_cm=50.0, width_cm=50.0, height_cm=50.0,
        weight_kg=5.0, this_way_up=False, permitted_postures=[Posture.LWH],
        inflated_length=50.0, inflated_width=50.0, inflated_height=50.0
    )
    # Two EPs at same x=0 and same z=0:
    # EP1: (0, 60, 0)
    # EP2: (0, 20, 0)
    from app.solver.geometry import ExtremePoint
    eps = [ExtremePoint(0.0, 60.0, 0.0), ExtremePoint(0.0, 20.0, 0.0)]

    settings = get_settings()
    monkeypatch.setattr(settings, "WALL_FIRST_SECONDARY_ORDER", "ZY")
    res, _ = find_best_placement(
        cand_box, [], [], container_dims,
        current_weight=0.0, max_weight=1000.0, is_lcl=True, extreme_points=eps
    )
    # Both have z=0, so smaller y wins
    assert res.position == Position(0.0, 20.0, 0.0)
