"""Remove-and-reinsert repair: accepted only when total placed volume strictly increases."""
from app.solver.geometry import BoundingBox, Dimensions, Posture
from app.solver.parsing import Box
from app.solver.repair import repair_swap


def _box(box_id, l, w, h, weight=10.0):
    return Box(
        box_id=box_id, item_id=f"I-{box_id}", po_no="PO", customer_code="C", customer_sequence=0,
        length_cm=l, width_cm=w, height_cm=h, weight_kg=weight, this_way_up=False,
        permitted_postures=[Posture.LWH], inflated_length=l, inflated_width=w, inflated_height=h,
    )


def _run(placed, unplaced, c, max_weight=1e6):
    bboxes = [BoundingBox(*pos) for _, pos in placed]
    data = [b for b, _ in placed]
    return repair_swap(
        bboxes, data, [Posture.LWH] * len(data), [(u, "no_space") for u in unplaced], c,
        sum(b.weight_kg for b in data), max_weight, budget_s=5.0, max_candidates=20,
    )


def test_swaps_small_carton_for_bigger_one_that_cannot_return():
    c = Dimensions(200.0, 200.0, 200.0)
    small = _box("S", 50.0, 50.0, 50.0)
    big = _box("B", 200.0, 200.0, 180.0)
    bb, data, _, pending, _, n = _run([(small, (0, 0, 0, 50, 50, 50))], [big], c)
    assert n == 1
    assert [d.box_id for d in data] == ["B"]
    assert [p[0].box_id for p in pending] == ["S"]


def test_rejects_swap_that_does_not_increase_volume():
    c = Dimensions(200.0, 200.0, 200.0)
    big = _box("B", 200.0, 200.0, 180.0)
    small = _box("S", 50.0, 50.0, 50.0)
    bb, data, _, pending, _, n = _run([(big, (0, 0, 0, 200, 200, 180))], [small], c)
    assert n == 0
    assert [d.box_id for d in data] == ["B"]
    assert [p[0].box_id for p in pending] == ["S"]


def test_swap_respects_weight_capacity():
    c = Dimensions(200.0, 200.0, 200.0)
    small = _box("S", 50.0, 50.0, 50.0, weight=10.0)
    big = _box("B", 200.0, 200.0, 180.0, weight=500.0)
    _, data, _, pending, _, n = _run([(small, (0, 0, 0, 50, 50, 50))], [big], c, max_weight=100.0)
    assert n == 0
    assert [d.box_id for d in data] == ["S"]
