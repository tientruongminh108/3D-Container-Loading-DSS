"""Tests for LIFO delivery order soft validator metric (Section B1)."""
from app.solver.geometry import Dimensions, Posture
from app.solver.validator import validate_solution
from app.core.models import PlacedBox


def test_lifo_soft_conflict_detected():
    """Verify that a later-drop carton blocking an earlier-drop carton towards +X triggers soft conflict."""
    container_dims = Dimensions(length=1200.0, width=235.0, height=239.0)

    # Box 1: Customer sequence 1 (earlier drop-off), placed at X=0..100, Y=0..50, Z=0..50
    b1 = PlacedBox(
        box_id="BOX_1",
        item_id="ITEM_A",
        po_no="PO1",
        length_cm=100.0,
        width_cm=50.0,
        height_cm=50.0,
        x=0.0,
        y=0.0,
        z=0.0,
        actual_length=100.0,
        actual_width=50.0,
        actual_height=50.0,
        weight_kg=20.0,
        posture=Posture.LWH,
        customer_sequence=1,
    )

    # Box 2: Customer sequence 2 (later drop-off), placed at X=100..200, Y=0..50, Z=0..50 (closer to door, overlapping in Y-Z)
    b2 = PlacedBox(
        box_id="BOX_2",
        item_id="ITEM_B",
        po_no="PO2",
        length_cm=100.0,
        width_cm=50.0,
        height_cm=50.0,
        x=100.0,
        y=0.0,
        z=0.0,
        actual_length=100.0,
        actual_width=50.0,
        actual_height=50.0,
        weight_kg=20.0,
        posture=Posture.LWH,
        customer_sequence=2,
    )

    # Validate with is_lcl=True
    report_lcl = validate_solution([b1, b2], container_dims, max_weight_kg=20000.0, is_lcl=True)
    # Must report conflict as soft metric without invalidating the solution
    assert report_lcl.is_valid is True
    assert report_lcl.total_violations == 0
    assert report_lcl.metrics["lifo_conflicts"] == 1
    assert report_lcl.metrics["lifo_blocked_cartons"] == 1

    # Validate with is_lcl=False (FCL mode)
    report_fcl = validate_solution([b1, b2], container_dims, max_weight_kg=20000.0, is_lcl=False)
    assert report_fcl.is_valid is True
    assert report_fcl.metrics["lifo_conflicts"] is None


def test_lifo_no_conflict_when_door_order_respected():
    """Verify 0 conflicts when earlier drop-off is closer to door (+X)."""
    container_dims = Dimensions(length=1200.0, width=235.0, height=239.0)

    # Box 2: Customer sequence 2 (later drop-off), placed deep at X=0..100
    b2 = PlacedBox(
        box_id="BOX_2",
        item_id="ITEM_B",
        po_no="PO2",
        length_cm=100.0,
        width_cm=50.0,
        height_cm=50.0,
        x=0.0,
        y=0.0,
        z=0.0,
        actual_length=100.0,
        actual_width=50.0,
        actual_height=50.0,
        weight_kg=20.0,
        posture=Posture.LWH,
        customer_sequence=2,
    )

    # Box 1: Customer sequence 1 (earlier drop-off), placed near door at X=100..200
    b1 = PlacedBox(
        box_id="BOX_1",
        item_id="ITEM_A",
        po_no="PO1",
        length_cm=100.0,
        width_cm=50.0,
        height_cm=50.0,
        x=100.0,
        y=0.0,
        z=0.0,
        actual_length=100.0,
        actual_width=50.0,
        actual_height=50.0,
        weight_kg=20.0,
        posture=Posture.LWH,
        customer_sequence=1,
    )

    report = validate_solution([b1, b2], container_dims, max_weight_kg=20000.0, is_lcl=True)
    assert report.is_valid is True
    assert report.metrics["lifo_conflicts"] == 0
    assert report.metrics["lifo_blocked_cartons"] == 0
