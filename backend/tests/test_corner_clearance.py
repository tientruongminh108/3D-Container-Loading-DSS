import pytest
from app.solver.geometry import Dimensions, BoundingBox, Position, Posture
from app.solver.constraints import check_corner_clearance
from app.solver.compaction import is_valid_shift
from app.solver.validator import validate_solution
from app.core.models import PlacedBox
from app.config import CORNER_BLOCK_X_CM, CORNER_BLOCK_Y_CM, CORNER_BLOCK_Z_CM


@pytest.fixture
def container_dims():
    return Dimensions(length=1200.0, width=235.0, height=269.0)


def test_corner_clearance_rear_left_top_rejected(container_dims):
    # Box at rear-left-top: x=[0, 10], y=[0, 10], z=[H-5, H]
    H = container_dims.height
    bbox = BoundingBox(0.0, 0.0, H - 5.0, 10.0, 10.0, H)
    assert not check_corner_clearance(bbox, container_dims)


def test_corner_clearance_rear_right_top_rejected(container_dims):
    # Box at rear-right-top: x=[0, 10], y=[W-10, W], z=[H-5, H]
    H = container_dims.height
    W = container_dims.width
    bbox = BoundingBox(0.0, W - 10.0, H - 5.0, 10.0, W, H)
    assert not check_corner_clearance(bbox, container_dims)


def test_corner_clearance_door_left_top_rejected(container_dims):
    # Box at door-left-top: x=[L-10, L], y=[0, 10], z=[H-5, H]
    H = container_dims.height
    L = container_dims.length
    bbox = BoundingBox(L - 10.0, 0.0, H - 5.0, L, 10.0, H)
    assert not check_corner_clearance(bbox, container_dims)


def test_corner_clearance_door_right_top_rejected(container_dims):
    # Box at door-right-top: x=[L-10, L], y=[W-10, W], z=[H-5, H]
    H = container_dims.height
    L = container_dims.length
    W = container_dims.width
    bbox = BoundingBox(L - 10.0, W - 10.0, H - 5.0, L, W, H)
    assert not check_corner_clearance(bbox, container_dims)


def test_corner_clearance_center_roof_accepted(container_dims):
    # Box at roof center: x=[50, 60], y=[50, 60], z=[H-5, H]
    H = container_dims.height
    bbox = BoundingBox(50.0, 50.0, H - 5.0, 60.0, 60.0, H)
    assert check_corner_clearance(bbox, container_dims)


def test_corner_clearance_below_corner_block_accepted(container_dims):
    # Box at rear-left corner but strictly below H - CORNER_BLOCK_Z_CM
    H = container_dims.height
    clear_z = H - CORNER_BLOCK_Z_CM
    bbox = BoundingBox(0.0, 0.0, clear_z - 20.0, 15.0, 15.0, clear_z)
    assert check_corner_clearance(bbox, container_dims)

    # Floor-level corner box is accepted
    floor_bbox = BoundingBox(0.0, 0.0, 0.0, 15.0, 15.0, 50.0)
    assert check_corner_clearance(floor_bbox, container_dims)


def test_compaction_is_valid_shift_rejects_corner_intrusion(container_dims):
    H = container_dims.height
    # Existing box at safe roof position: x=[30, 40], y=[0, 10], z=[H-5, H]
    safe_bbox = BoundingBox(30.0, 0.0, H - 5.0, 40.0, 10.0, H)
    bboxes = [safe_bbox]

    # Candidate shifting into rear-left-top corner: x=[5, 15]
    corner_candidate = BoundingBox(5.0, 0.0, H - 5.0, 15.0, 10.0, H)
    valid = is_valid_shift(
        i=0,
        new_bbox=corner_candidate,
        current_bboxes=bboxes,
        min_support_ratio=0.60,
        container_dims=container_dims,
    )
    assert not valid
    # Ensure current_bboxes was not corrupted
    assert bboxes[0] == safe_bbox


def test_validator_detects_top_corner_intrusion(container_dims):
    H = container_dims.height
    # PlacedBox encroaching rear-left-top corner
    bad_box = PlacedBox(
        box_id="B-BAD",
        item_id="ITEM-1",
        po_no="PO-001",
        length_cm=10.0,
        width_cm=10.0,
        height_cm=5.0,
        x=0.0,
        y=0.0,
        z=H - 5.0,
        weight_kg=10.0,
        posture=Posture.LWH,
        step_index=1,
        actual_length=10.0,
        actual_width=10.0,
        actual_height=5.0,
        permitted_postures=[Posture.LWH],
    )
    report = validate_solution(
        placed_boxes=[bad_box],
        container_dims=container_dims,
        max_weight_kg=28000.0,
    )
    assert not report.is_valid
    assert report.violations_by_type.get("corner_clearance", 0) > 0
    assert any("top-corner obstruction cuboid" in msg for msg in report.error_messages)
