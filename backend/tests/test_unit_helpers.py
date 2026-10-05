import pytest
from app.config import get_settings
from app.solver.parsing import Box
from app.solver.block_generation import Block
from app.solver.geometry import Dimensions, Posture, unit_weight, unit_footprint


def test_unit_weight_and_footprint():
    box = Box(
        box_id="B1",
        item_id="I1",
        po_no="PO1",
        customer_code="CUST1",
        customer_sequence=1,
        length_cm=40.0,
        width_cm=30.0,
        height_cm=20.0,
        weight_kg=12.5,
        this_way_up=False,
        permitted_postures=[Posture.LWH, Posture.WLH, Posture.HLW],
        inflated_length=42.0,
        inflated_width=32.0,
        inflated_height=20.0,
    )
    assert unit_weight(box) == 12.5
    assert unit_footprint(box) == 1200.0
    assert unit_footprint(box, Posture.WLH) == 30.0 * 40.0
    assert unit_footprint(box, Posture.HLW) == 20.0 * 40.0

    block = Block(
        block_id="BLK1",
        boxes=[box, box],
        length_cm=80.0,
        width_cm=30.0,
        height_cm=20.0,
        weight_kg=25.0,
        customer_sequence=1,
        inflated_length=82.0,
        inflated_width=32.0,
        inflated_height=20.0,
        contents=[box, box],
    )
    # Unit weight of carton in block
    assert unit_weight(block) == 12.5
    assert unit_footprint(block) == 2400.0


def test_shelf_settings_defaults():
    settings = get_settings()
    assert hasattr(settings, "MIN_USABLE_SHELF_CM")
    assert hasattr(settings, "SHELF_OCCUPANCY_RATIO")
    assert settings.min_usable_shelf_cm == settings.MIN_USABLE_SHELF_CM
    assert settings.shelf_occupancy_ratio == settings.SHELF_OCCUPANCY_RATIO
