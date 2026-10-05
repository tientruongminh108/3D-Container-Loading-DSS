"""Tests for extreme point candidate limit and caching (Section A4)."""
import pytest
from app.config import get_settings
from app.solver.geometry import Dimensions, ExtremePoint, BoundingBox
from app.solver.parsing import Box
from app.solver.strategy_variants import (
    compute_free_cuboid_at_ep,
    sort_extreme_points,
    CartonGroup,
)


def test_ep_candidate_limit_setting():
    settings = get_settings()
    assert hasattr(settings, "EP_CANDIDATE_LIMIT")
    assert settings.EP_CANDIDATE_LIMIT == 100


def test_compute_free_cuboid_at_ep():
    container_dims = Dimensions(length=100.0, width=100.0, height=100.0)
    # Box placed at (0, 0, 0) of size (20, 20, 20)
    b1 = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=20, max_y=20, max_z=20)
    placed = [b1]

    # Free cuboid at (20, 0, 0)
    ep = ExtremePoint(x=20, y=0, z=0)
    dx, dy, dz = compute_free_cuboid_at_ep(ep, placed, container_dims)
    assert dx == 80.0
    assert dy == 100.0
    assert dz == 100.0

    # Free cuboid at (0, 0, 20)
    ep2 = ExtremePoint(x=0, y=0, z=20)
    dx2, dy2, dz2 = compute_free_cuboid_at_ep(ep2, placed, container_dims)
    assert dx2 == 100.0
    assert dy2 == 100.0
    assert dz2 == 80.0
