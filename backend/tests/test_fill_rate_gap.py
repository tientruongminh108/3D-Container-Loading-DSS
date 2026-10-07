"""Tests for fill rate denominator and tolerance gap default/shift (Section B2)."""
import pytest
from app.solver.geometry import Dimensions, Posture
from app.solver.output import calculate_metrics
from app.core.models import PlacedBox, RunOptions
from app.config import get_settings


def test_tolerance_gap_defaults():
    settings = get_settings()
    assert settings.TOLERANCE_GAP_CM == 0.0
    assert settings.CONTAINER_WALL_CLEARANCE_CM == 0.0

    options = RunOptions()
    assert options.tolerance_gap_cm == 0.0
    assert options.container_wall_clearance_cm == 0.0


def test_calculate_metrics_nominal_volume_denominator():
    """Verify that calculate_metrics uses nominal_container_dims rather than usable interior."""
    usable_dims = Dimensions(length=100.0, width=100.0, height=100.0)  # vol = 1,000,000
    nominal_dims = Dimensions(length=120.0, width=120.0, height=100.0) # vol = 1,440,000

    # Box with volume 50 * 50 * 50 = 125,000
    box = PlacedBox(
        box_id="B1",
        item_id="I1",
        po_no="PO1",
        length_cm=50.0,
        width_cm=50.0,
        height_cm=50.0,
        x=10.0,
        y=10.0,
        z=0.0,
        actual_length=50.0,
        actual_width=50.0,
        actual_height=50.0,
        weight_kg=10.0,
        posture=Posture.LWH,
    )

    metrics = calculate_metrics(
        placed_boxes=[box],
        unplaced=[],
        container_dims=usable_dims,
        max_weight=1000.0,
        nominal_container_dims=nominal_dims,
    )

    expected_fill = 125_000.0 / 1_440_000.0
    assert metrics.fill_rate == pytest.approx(expected_fill, abs=1e-5)
    assert metrics.total_volume_cbm == pytest.approx(1.44, abs=1e-3)
