"""Phase 1 Coordinate Convention Unit Tests.

Conforms to coordinate convention (x=0 is rear wall, x=L is door):
- Free box slides toward x=0 until contact with container wall or another box.
(Obsolete check_lifo and LCL-specific LIFO tests removed per prompt TASK 1).
"""

import pytest
from app.solver.geometry import BoundingBox, Dimensions, Posture
from app.solver.parsing import Box
from app.solver.compaction import compact_x_rear


def make_dummy_box(box_id: str, cust_seq: int = 0, l: float = 20.0, w: float = 10.0, h: float = 10.0) -> Box:
    return Box(
        box_id=box_id,
        item_id=f"ITEM_{box_id}",
        po_no="PO1",
        customer_code=f"CUST_{cust_seq}",
        customer_sequence=cust_seq,
        length_cm=l,
        width_cm=w,
        height_cm=h,
        weight_kg=10.0,
        this_way_up=False,
        permitted_postures=[Posture.LWH],
        inflated_length=l,
        inflated_width=w,
        inflated_height=h,
    )


class TestPhase1Convention:

    def test_handmade_case4_free_box_slides_to_rear_wall(self):
        """Case 4: Free box slides toward x=0 until contact with container wall or another box."""
        c_dims = Dimensions(length=1200.0, width=240.0, height=260.0)
        b1 = make_dummy_box("B1", cust_seq=1, l=50.0, w=40.0, h=30.0)
        b2 = make_dummy_box("B2", cust_seq=1, l=50.0, w=40.0, h=30.0)

        # Place B1 at x=100 (in the middle), B2 at x=250 in line behind B1
        bbox1 = BoundingBox(100.0, 0.0, 0.0, 150.0, 40.0, 30.0)
        bbox2 = BoundingBox(250.0, 0.0, 0.0, 300.0, 40.0, 30.0)

        compacted = compact_x_rear([bbox1, bbox2], [b1, b2], c_dims, is_lcl=False)

        # B1 should slide to rear wall x=0: [0, 50]
        assert compacted[0].min_x == pytest.approx(0.0)
        assert compacted[0].max_x == pytest.approx(50.0)
        # B2 should slide to make contact with B1 at x=50: [50, 100]
        assert compacted[1].min_x == pytest.approx(50.0)
        assert compacted[1].max_x == pytest.approx(100.0)
