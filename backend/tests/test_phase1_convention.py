"""Phase 1 Coordinate Convention & LIFO Parity Unit Tests.

Conforms to PROMPT_coordinate_convention.md:
- Hand-made cases:
  1. Box A (cust 1) at x=50, Box B (cust 2) at x=10 (deeper than A): check_lifo(A, B) -> True.
  2. Box A (cust 1) at x=10, Box B (cust 2) at x=50 (shallower than A, blocking B): check_lifo(A, B) -> False.
  3. Box A (cust 1) and Box B (cust 2) both at x=10, but disjoint in Y: check_lifo(A, B) -> True.
  4. Free box slides toward x=0 until contact with container wall or another box.
  5. Box whose slide would put it deeper than a later customer's box is refused.
- Mirror parity test:
  check_lifo_new(A_new, B_new) == check_lifo_old(mirror_x(A_new, L), mirror_x(B_new, L))
- Order independence:
  check_lifo(A, [B], [b_data], seq_A) == check_lifo(B, [A], [a_data], seq_B)
"""

import pytest
from app.solver.geometry import BoundingBox, Dimensions, Position, Posture
from app.solver.parsing import Box
from app.solver.constraints import check_lifo
from app.solver.compaction import compact_x_rear


def make_dummy_box(box_id: str, cust_seq: int, l: float = 20.0, w: float = 10.0, h: float = 10.0) -> Box:
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


def check_lifo_old(candidate: BoundingBox, placed: BoundingBox, cand_seq: int, placed_seq: int) -> bool:
    """Old convention: Door is x=0, rear wall is x=L.
    Earlier drop-off (smaller cust_seq) must be near door (smaller x).
    Later drop-off (larger cust_seq) must be deeper (larger x).
    """
    if cand_seq == placed_seq:
        return True
    if not candidate.overlaps_yz(placed):
        return True
    if cand_seq < placed_seq:
        # candidate is earlier drop-off -> candidate must be nearer door (smaller x) than placed
        return candidate.max_x <= placed.min_x + 1e-6
    else:
        # candidate is later drop-off -> candidate must be deeper (larger x) than placed
        return placed.max_x <= candidate.min_x + 1e-6


def mirror_bbox(bbox: BoundingBox, L: float) -> BoundingBox:
    """Mirror along X axis for container of length L: [x1, x2] -> [L - x2, L - x1]."""
    return BoundingBox(
        min_x=L - bbox.max_x,
        min_y=bbox.min_y,
        min_z=bbox.min_z,
        max_x=L - bbox.min_x,
        max_y=bbox.max_y,
        max_z=bbox.max_z,
    )


class TestPhase1Convention:

    def test_handmade_case1_deeper_late_customer_passes(self):
        """Case 1: Box A (cust 1) at x=50, Box B (cust 2) at x=10 (deeper than A): check_lifo -> True."""
        b_a = make_dummy_box("A", cust_seq=1, l=20.0, w=10.0, h=10.0)
        b_b = make_dummy_box("B", cust_seq=2, l=20.0, w=10.0, h=10.0)

        # B is placed at [10, 30] (near rear wall x=0)
        bbox_b = BoundingBox(10.0, 0.0, 0.0, 30.0, 10.0, 10.0)
        # A is candidate at [50, 70] (nearer door x=L)
        bbox_a = BoundingBox(50.0, 0.0, 0.0, 70.0, 10.0, 10.0)

        assert check_lifo(bbox_a, [bbox_b], [b_b], candidate_sequence=1) is True
        assert check_lifo(bbox_b, [bbox_a], [b_a], candidate_sequence=2) is True

    def test_handmade_case2_shallower_late_customer_fails(self):
        """Case 2: Box A (cust 1) at x=10, Box B (cust 2) at x=50:
        Box A (earlier drop-off) is trapped behind Box B (later drop-off) -> check_lifo -> False.
        """
        b_a = make_dummy_box("A", cust_seq=1, l=20.0, w=10.0, h=10.0)
        b_b = make_dummy_box("B", cust_seq=2, l=20.0, w=10.0, h=10.0)

        bbox_a = BoundingBox(10.0, 0.0, 0.0, 30.0, 10.0, 10.0)
        bbox_b = BoundingBox(50.0, 0.0, 0.0, 70.0, 10.0, 10.0)

        assert check_lifo(bbox_b, [bbox_a], [b_a], candidate_sequence=2) is False
        assert check_lifo(bbox_a, [bbox_b], [b_b], candidate_sequence=1) is False

    def test_handmade_case3_disjoint_yz_passes(self):
        """Case 3: Box A (cust 1) and Box B (cust 2) both at x=10, but disjoint in Y: check_lifo -> True."""
        b_a = make_dummy_box("A", cust_seq=1, l=20.0, w=10.0, h=10.0)
        b_b = make_dummy_box("B", cust_seq=2, l=20.0, w=10.0, h=10.0)

        bbox_a = BoundingBox(10.0, 0.0, 0.0, 30.0, 10.0, 10.0)
        bbox_b = BoundingBox(10.0, 20.0, 0.0, 30.0, 30.0, 10.0)  # y=[20, 30], disjoint from y=[0, 10]

        assert check_lifo(bbox_a, [bbox_b], [b_b], candidate_sequence=1) is True
        assert check_lifo(bbox_b, [bbox_a], [b_a], candidate_sequence=2) is True

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

    def test_handmade_case5_slide_violating_lifo_is_refused(self):
        """Case 5: Box whose slide would put it deeper than a later customer's box is refused."""
        c_dims = Dimensions(length=1200.0, width=240.0, height=260.0)
        # B_DEEP belongs to customer 2 (later, deep)
        # B_EARLY belongs to customer 1 (earlier, door)
        b_deep = make_dummy_box("B_DEEP", cust_seq=2, l=50.0, w=40.0, h=30.0)
        b_early = make_dummy_box("B_EARLY", cust_seq=1, l=50.0, w=40.0, h=30.0)

        # B_DEEP is at x=[50, 100]
        bbox_deep = BoundingBox(50.0, 0.0, 0.0, 100.0, 40.0, 30.0)
        # B_EARLY is at x=[150, 200]
        bbox_early = BoundingBox(150.0, 0.0, 0.0, 200.0, 40.0, 30.0)

        compacted = compact_x_rear([bbox_deep, bbox_early], [b_deep, b_early], c_dims, is_lcl=True)

        # B_DEEP (cust 2) slides to rear wall x=0 -> [0, 50]
        assert compacted[0].min_x == pytest.approx(0.0)
        assert compacted[0].max_x == pytest.approx(50.0)
        # B_EARLY (cust 1) must NOT slide past B_DEEP; it must stop at x=50 -> [50, 100]
        assert compacted[1].min_x == pytest.approx(50.0)
        assert compacted[1].max_x == pytest.approx(100.0)

    def test_order_independence(self):
        """check_lifo(A, [B], [b_data], seq_A) == check_lifo(B, [A], [a_data], seq_B) for any configuration."""
        b_a = make_dummy_box("A", cust_seq=1, l=30.0, w=20.0, h=20.0)
        b_b = make_dummy_box("B", cust_seq=2, l=30.0, w=20.0, h=20.0)

        test_configs = [
            # (bbox_a, bbox_b)
            (BoundingBox(50.0, 0.0, 0.0, 80.0, 20.0, 20.0), BoundingBox(10.0, 0.0, 0.0, 40.0, 20.0, 20.0)),  # valid
            (BoundingBox(10.0, 0.0, 0.0, 40.0, 20.0, 20.0), BoundingBox(50.0, 0.0, 0.0, 80.0, 20.0, 20.0)),  # invalid
            (BoundingBox(10.0, 0.0, 0.0, 40.0, 20.0, 20.0), BoundingBox(10.0, 30.0, 0.0, 40.0, 50.0, 20.0)),  # disjoint Y
            (BoundingBox(10.0, 0.0, 0.0, 40.0, 20.0, 20.0), BoundingBox(10.0, 0.0, 30.0, 40.0, 20.0, 50.0)),  # disjoint Z
            (BoundingBox(40.0, 0.0, 0.0, 70.0, 20.0, 20.0), BoundingBox(10.0, 0.0, 0.0, 40.0, 20.0, 20.0)),  # touching at x=40
        ]

        for bbox_a, bbox_b in test_configs:
            res_a = check_lifo(bbox_a, [bbox_b], [b_b], candidate_sequence=1)
            res_b = check_lifo(bbox_b, [bbox_a], [b_a], candidate_sequence=2)
            assert res_a == res_b, f"Order independence failed for {bbox_a} and {bbox_b}: {res_a} != {res_b}"

    def test_mirror_parity_with_old_convention(self):
        """For any pair of boxes (A, B) with seq(A) < seq(B):
        check_lifo_new(A_new, B_new) == check_lifo_old(mirror_x(A_new, L), mirror_x(B_new, L)).
        """
        L = 1000.0
        b_a = make_dummy_box("A", cust_seq=1, l=30.0, w=20.0, h=20.0)
        b_b = make_dummy_box("B", cust_seq=2, l=30.0, w=20.0, h=20.0)

        # Generate a range of test pairs in the new convention
        positions = [0.0, 50.0, 100.0, 200.0, 500.0, 800.0]
        y_offsets = [0.0, 15.0, 30.0]  # 0 and 15 overlap (w=20), 30 is disjoint

        for x_a in positions:
            for x_b in positions:
                for y_b in y_offsets:
                    bbox_a_new = BoundingBox(x_a, 0.0, 0.0, x_a + 30.0, 20.0, 20.0)
                    bbox_b_new = BoundingBox(x_b, y_b, 0.0, x_b + 30.0, y_b + 20.0, 20.0)

                    # New convention check
                    res_new = check_lifo(bbox_a_new, [bbox_b_new], [b_b], candidate_sequence=1)

                    # Mirror into old convention coordinates
                    bbox_a_old = mirror_bbox(bbox_a_new, L)
                    bbox_b_old = mirror_bbox(bbox_b_new, L)

                    # Old convention check
                    res_old = check_lifo_old(bbox_a_old, bbox_b_old, cand_seq=1, placed_seq=2)

                    assert res_new == res_old, (
                        f"Parity failure at x_a={x_a}, x_b={x_b}, y_b={y_b}: "
                        f"new={res_new} vs old={res_old}\n"
                        f"new: a={bbox_a_new}, b={bbox_b_new}\n"
                        f"old: a={bbox_a_old}, b={bbox_b_old}"
                    )
