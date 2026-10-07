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
        sum(b.weight_kg for b in data), max_weight, max_trials=500, max_candidates=20,
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


def test_result_does_not_depend_on_wall_clock(monkeypatch):
    """The work bound is a trial count: a wildly different clock must not change the layout."""
    import time
    c = Dimensions(200.0, 200.0, 200.0)
    small = _box("S", 50.0, 50.0, 50.0)
    big = _box("B", 200.0, 200.0, 180.0)
    ref = _run([(small, (0, 0, 0, 50, 50, 50))], [big], c)
    ticks = iter(range(0, 10**9, 10**6))
    monkeypatch.setattr(time, "time", lambda: next(ticks))
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    again = _run([(small, (0, 0, 0, 50, 50, 50))], [big], c)
    assert [d.box_id for d in ref[1]] == [d.box_id for d in again[1]]
    assert [(b.min_x, b.min_y, b.min_z) for b in ref[0]] == [(b.min_x, b.min_y, b.min_z) for b in again[0]]
    assert ref[5] == again[5]


def test_trial_bound_is_respected():
    c = Dimensions(200.0, 200.0, 200.0)
    small = _box("S", 50.0, 50.0, 50.0)
    big = _box("B", 200.0, 200.0, 180.0)
    bboxes = [BoundingBox(0, 0, 0, 50, 50, 50)]
    out = repair_swap(bboxes, [small], [Posture.LWH], [(big, "no_space")], c, 10.0, 1e6, max_trials=0)
    assert out[5] == 0

