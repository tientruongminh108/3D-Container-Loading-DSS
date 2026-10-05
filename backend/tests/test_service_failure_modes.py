"""Tests for service failure modes (Section B3)."""
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session
from app.services import RunService
from app.core.models import RunCreate, PackingListUpload, PackingListRow
from app.core.exceptions import ValidationError


def test_missing_item_in_master_raises_validation_error():
    """Verify that an item missing from Item Master raises ValidationError rather than inventing fake dimensions."""
    db = MagicMock(spec=Session)

    service = RunService(db)
    service.generate_run_id = MagicMock(return_value="051026-A01")
    service.container_service = MagicMock()
    service.container_service.get = MagicMock(return_value=MagicMock(
        id=1, container_type="40HC",
        internal_length_cm=1203.2, internal_width_cm=235.2, internal_height_cm=270.0,
        max_weight_kg=28000.0
    ))
    service.item_service = MagicMock()
    # Item lookup returns empty dict (item not in master)
    service.item_service.get_all = MagicMock(return_value=[])

    run_create = RunCreate(
        container_id=1,
        packing_list=PackingListUpload(
            rows=[
                PackingListRow(
                    item_id="UNKNOWN_ITEM",
                    po_no="PO1",
                    qty_cartons=5,
                    qty_pcs=5,
                )
            ]
        ),
    )

    with pytest.raises(ValidationError) as exc_info:
        service.create_run(run_create)

    assert "UNKNOWN_ITEM" in str(exc_info.value.message)
    assert "not found in Item Master" in str(exc_info.value.message)


def test_solver_failure_raises_without_mock_fallback():
    """Verify that solver exceptions are raised rather than silently returning mock packer results."""
    db = MagicMock(spec=Session)

    service = RunService(db)
    service.generate_run_id = MagicMock(return_value="051026-A01")
    service.container_service = MagicMock()
    service.container_service.get = MagicMock(return_value=MagicMock(
        id=1, container_type="40HC",
        internal_length_cm=1203.2, internal_width_cm=235.2, internal_height_cm=270.0,
        max_weight_kg=28000.0
    ))
    mock_item = MagicMock(
        item_id="ITEM1", description="Item 1",
        length_cm=50.0, width_cm=40.0, height_cm=30.0,
        weight_kg=10.0, this_way_up=True
    )
    db.query.return_value.filter.return_value.all.return_value = [mock_item]

    run_create = RunCreate(
        container_id=1,
        packing_list=PackingListUpload(
            rows=[
                PackingListRow(
                    item_id="ITEM1",
                    po_no="PO1",
                    qty_cartons=5,
                    qty_pcs=5,
                )
            ]
        ),
    )

    with patch("app.services.run_pipeline", side_effect=RuntimeError("Solver internal failure")):
        with pytest.raises(RuntimeError) as exc_info:
            service.create_run(run_create)
        assert "Solver internal failure" in str(exc_info.value)
