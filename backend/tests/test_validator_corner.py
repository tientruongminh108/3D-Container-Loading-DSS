"""Tests for corner-clearance validation in validator.py."""
import pytest
from app.config import get_settings
from app.core.models import PlacedBox
from app.solver.geometry import Dimensions, Posture
from app.solver.validator import validate_solution


def _make_placed_box(box_id: str, x: float, y: float, z: float, l: float, w: float, h: float) -> PlacedBox:
    return PlacedBox(
        box_id=box_id,
        item_id=f"ITEM-{box_id}",
        po_no="PO-1",
        customer_code="CUST",
        customer_sequence=1,
        length_cm=l,
        width_cm=w,
        height_cm=h,
        weight_kg=10.0,
        this_way_up=False,
        permitted_postures=[Posture.LWH],
        inflated_length=l,
        inflated_width=w,
        inflated_height=h,
        x=x,
        y=y,
        z=z,
        posture=Posture.LWH,
        actual_length=l,
        actual_width=w,
        actual_height=h,
    )


def test_validator_detects_corner_casting_violation():
    settings = get_settings()
    c = Dimensions(589.8, 235.2, 239.3)
    cx = settings.CORNER_BLOCK_X_CM
    cy = settings.CORNER_BLOCK_Y_CM

    # Box resting on the floor (z=0) but reaching container ceiling H, placed in corner (0,0)
    # This is within container bounds and fully supported, but intersects the rear-left top corner casting.
    violating_box = _make_placed_box(
        "B_VIOLATING",
        x=0.0,
        y=0.0,
        z=0.0,
        l=cx + 10.0,
        w=cy + 10.0,
        h=c.height,
    )

    report_bad = validate_solution([violating_box], c, max_weight_kg=28000.0)
    assert not report_bad.is_valid
    assert report_bad.violations_by_type["corner_clearance"] == 1
    assert "top-corner obstruction cuboid" in report_bad.error_messages[0]


def test_validator_accepts_clean_layout():
    settings = get_settings()
    c = Dimensions(589.8, 235.2, 239.3)
    cz = settings.CORNER_BLOCK_Z_CM

    # Clean box 1: on the floor, well below the ceiling
    box1 = _make_placed_box("B_FLOOR", x=100.0, y=100.0, z=0.0, l=100.0, w=100.0, h=100.0)

    # Clean box 2: stacked on box 1, reaches near the ceiling, but centered away from corners
    box2 = _make_placed_box("B_STACKED", x=100.0, y=100.0, z=100.0, l=100.0, w=100.0, h=c.height - 100.0)

    report_good = validate_solution([box1, box2], c, max_weight_kg=28000.0)
    assert report_good.is_valid
    assert report_good.violations_by_type["corner_clearance"] == 0
    assert report_good.total_violations == 0
