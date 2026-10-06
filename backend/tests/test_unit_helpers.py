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


def test_run_deterministic_mock_pack():
    from app.services.mock_packer import run_deterministic_mock_pack
    from app.core.models import Container, PackingListRow, Item
    from datetime import datetime, timezone

    container = Container(
        id=1,
        container_type="20GP",
        internal_length_cm=589.8,
        internal_width_cm=235.2,
        internal_height_cm=239.3,
        max_weight_kg=28000.0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    rows = [
        PackingListRow(po_no="PO1", item_id="ITEM_1", qty_pcs=10, qty_cartons=5, customer_code="CUST1"),
        PackingListRow(po_no="PO2", item_id="ITEM_2", qty_pcs=10, qty_cartons=3, customer_code="CUST1"),
    ]

    item_lookup = {
        "ITEM_1": Item(
            id=1,
            item_id="ITEM_1",
            description="Item 1",
            length_cm=40.0,
            width_cm=30.0,
            height_cm=20.0,
            weight_kg=10.0,
            this_way_up=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        "ITEM_2": Item(
            id=2,
            item_id="ITEM_2",
            description="Item 2",
            length_cm=50.0,
            width_cm=35.0,
            height_cm=25.0,
            weight_kg=15.0,
            this_way_up=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    }

    result = run_deterministic_mock_pack(container, rows, item_lookup)
    assert result.status == "completed"
    assert result.metrics.placed_count == 8
    assert result.metrics.used_volume_cbm is not None
    assert result.metrics.unused_volume_cbm is not None
    # Check that each SKU gets distinct color
    sku_colors = {b.item_id: b.color for b in result.placed_boxes}
    assert len(sku_colors) == 2
    assert len(set(sku_colors.values())) == 2
