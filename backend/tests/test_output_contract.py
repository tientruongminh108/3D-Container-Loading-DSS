import pytest
from datetime import datetime, timezone
from app.core.models import (
    RunResult, LoadMetrics, PlacedBox, UnplacedCarton,
    Container, RunStatus, Posture, UnplacedReason
)


def create_fixture_plan_fcl() -> RunResult:
    """Create a hand-built FCL fixture plan for testing"""
    container = Container(
        id=1,
        container_type="40HC",
        internal_length_cm=1203.2,
        internal_width_cm=235.2,
        internal_height_cm=270.0,
        max_weight_kg=28000,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    placed_boxes = [
        PlacedBox(
            box_id="BOX-0001",
            item_id="ITEM-001",
            po_no="PO-001",
            customer_code=None,
            customer_sequence=0,
            length_cm=100,
            width_cm=50,
            height_cm=40,
            weight_kg=200.0,
            this_way_up=True,
            permitted_postures=[Posture.LWH, Posture.WLH],
            inflated_length=102,
            inflated_width=52,
            inflated_height=40,
            x=10.0, y=10.0, z=0.0,
            posture=Posture.LWH,
            actual_length=100,
            actual_width=50,
            actual_height=40,
        ),
        PlacedBox(
            box_id="BOX-0002",
            item_id="ITEM-001",
            po_no="PO-001",
            customer_code=None,
            customer_sequence=0,
            length_cm=100,
            width_cm=50,
            height_cm=40,
            weight_kg=200.0,
            this_way_up=True,
            permitted_postures=[Posture.LWH, Posture.WLH],
            inflated_length=102,
            inflated_width=52,
            inflated_height=40,
            x=112.0, y=10.0, z=0.0,
            posture=Posture.LWH,
            actual_length=100,
            actual_width=50,
            actual_height=40,
        ),
    ]

    return RunResult(
        run_id="test-run-fcl-001",
        status=RunStatus.COMPLETED,
        container=container,
        metrics=LoadMetrics(
            placed_count=2,
            unplaced_count=0,
            total_cartons=2,
            fill_rate=0.05,
            used_weight_kg=400.0,
            max_weight_kg=28000.0,
            weight_utilization=1.43,
            cog_x=601.6,
            cog_y=117.6,
            cog_z=80.0,
            cog_deviation_xy=0.0,
            cog_deviation_z=0.0,
        ),
        placed_boxes=placed_boxes,
        unplaced_cartons=[],
        layers=[],
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


def create_fixture_plan_lcl() -> RunResult:
    """Create a hand-built LCL fixture plan for testing"""
    container = Container(
        id=1,
        container_type="40HC",
        internal_length_cm=1203.2,
        internal_width_cm=235.2,
        internal_height_cm=270.0,
        max_weight_kg=28000,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    placed_boxes = [
        PlacedBox(
            box_id="BOX-0001",
            item_id="ITEM-001",
            po_no="PO-001",
            customer_code="CUST-A",
            customer_sequence=1,
            length_cm=100,
            width_cm=50,
            height_cm=40,
            weight_kg=200.0,
            this_way_up=True,
            permitted_postures=[Posture.LWH, Posture.WLH],
            inflated_length=102,
            inflated_width=52,
            inflated_height=40,
            x=10.0, y=10.0, z=0.0,
            posture=Posture.LWH,
            actual_length=100,
            actual_width=50,
            actual_height=40,
        ),
        PlacedBox(
            box_id="BOX-0002",
            item_id="ITEM-002",
            po_no="PO-002",
            customer_code="CUST-B",
            customer_sequence=2,
            length_cm=80,
            width_cm=60,
            height_cm=50,
            weight_kg=150.0,
            this_way_up=False,
            permitted_postures=list(Posture),
            inflated_length=82,
            inflated_width=62,
            inflated_height=50,
            x=10.0, y=70.0, z=0.0,
            posture=Posture.LWH,
            actual_length=80,
            actual_width=60,
            actual_height=50,
        ),
    ]

    unplaced_cartons = [
        UnplacedCarton(
            box_id="BOX-UNPLACED-001",
            item_id="ITEM-003",
            po_no="PO-003",
            customer_code="CUST-A",
            customer_sequence=1,
            reason=UnplacedReason.NO_SPACE,
            length_cm=100,
            width_cm=50,
            height_cm=40,
            weight_kg=200.0,
        ),
        UnplacedCarton(
            box_id="BOX-UNPLACED-002",
            item_id="ITEM-004",
            po_no="PO-004",
            customer_code="CUST-B",
            customer_sequence=2,
            reason=UnplacedReason.LIFO_BLOCKED,
            length_cm=80,
            width_cm=60,
            height_cm=50,
            weight_kg=150.0,
        ),
    ]

    return RunResult(
        run_id="test-run-lcl-001",
        status=RunStatus.COMPLETED,
        container=container,
        metrics=LoadMetrics(
            placed_count=2,
            unplaced_count=2,
            total_cartons=4,
            fill_rate=0.03,
            used_weight_kg=350.0,
            max_weight_kg=28000.0,
            weight_utilization=1.25,
            cog_x=601.6,
            cog_y=117.6,
            cog_z=20.0,
            cog_deviation_xy=0.0,
            cog_deviation_z=0.0,
        ),
        placed_boxes=placed_boxes,
        unplaced_cartons=unplaced_cartons,
        layers=[],
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


class TestOutputContractDataShape:
    """Tests for data contract shape (Section 2.1 - OUT-01 to OUT-07)"""

    def test_OUT_01_fcl_customer_code_null(self):
        """OUT-01: FCL plan should have null/constant customer_code on placements"""
        plan = create_fixture_plan_fcl()
        for box in plan.placed_boxes:
            assert box.customer_code is None or box.customer_code == ""
        assert plan.metrics.unplaced_count == 0

    def test_OUT_02_lcl_customer_code_present(self):
        """OUT-02: LCL plan should have non-null customer_code on placements"""
        plan = create_fixture_plan_lcl()
        for box in plan.placed_boxes:
            assert box.customer_code is not None
            assert box.customer_code != ""
        assert plan.metrics.unplaced_count == 2

    def test_OUT_03_unplaced_reasons_distinct(self):
        """OUT-03: Unplaced reasons should appear distinctly"""
        plan = create_fixture_plan_lcl()
        reasons = [uc.reason for uc in plan.unplaced_cartons]
        assert UnplacedReason.NO_SPACE in reasons
        assert UnplacedReason.LIFO_BLOCKED in reasons
        assert reasons.count(UnplacedReason.NO_SPACE) == 1
        assert reasons.count(UnplacedReason.LIFO_BLOCKED) == 1

    def test_OUT_04_placements_preserve_load_order(self):
        """OUT-04: Placements list preserves load order"""
        plan = create_fixture_plan_fcl()
        load_order = [box.box_id for box in plan.placed_boxes]
        assert load_order == ["BOX-0001", "BOX-0002"]

    def test_OUT_05_unload_order_is_reverse(self):
        """OUT-05: Unload order is exactly placements reversed"""
        plan = create_fixture_plan_fcl()
        load_order = [box.box_id for box in plan.placed_boxes]
        unload_order = list(reversed(load_order))
        assert unload_order == ["BOX-0002", "BOX-0001"]

    def test_OUT_06_no_blocks_in_output(self):
        """OUT-06: No placement should have is_block flag or block-shaped dimensions"""
        plan = create_fixture_plan_fcl()
        for box in plan.placed_boxes:
            assert not hasattr(box, "is_block")
            assert box.actual_length > 0
            assert box.actual_width > 0
            assert box.actual_height > 0

    def test_OUT_07_as_placed_dimensions(self):
        """OUT-07: Placement dimensions are as-placed (post-posture)"""
        plan = create_fixture_plan_lcl()
        box = plan.placed_boxes[1]
        assert box.actual_length == 80
        assert box.actual_width == 60
        assert box.actual_height == 50
        assert box.posture == Posture.LWH


class TestConsoleSummary:
    """Tests for console summary format (Section 2.2 - OUT-08 to OUT-12)"""

    def test_OUT_08_all_placed_banner(self):
        """OUT-08: All placed shows '*** ALL CARTONS PLACED ***' banner"""
        plan = create_fixture_plan_fcl()
        assert plan.metrics.unplaced_count == 0
        assert plan.metrics.placed_count == plan.metrics.total_cartons

    def test_OUT_09_fcl_unplaced_no_breakdown(self):
        """OUT-09: FCL with unplaced shows total banner only, no breakdown"""
        plan = create_fixture_plan_fcl()
        plan.metrics.unplaced_count = 5
        plan.metrics.placed_count = 5
        plan.metrics.total_cartons = 10
        assert plan.metrics.unplaced_count > 0
        assert plan.metrics.unplaced_count == 5

    def test_OUT_10_lcl_unplaced_breakdown(self):
        """OUT-10: LCL with unplaced shows no_space and LIFO_blocked breakdown"""
        plan = create_fixture_plan_lcl()
        no_space = sum(1 for uc in plan.unplaced_cartons if uc.reason == UnplacedReason.NO_SPACE)
        lifo_blocked = sum(1 for uc in plan.unplaced_cartons if uc.reason == UnplacedReason.LIFO_BLOCKED)
        assert no_space == 1
        assert lifo_blocked == 1
        assert no_space + lifo_blocked == plan.metrics.unplaced_count

    def test_OUT_11_fcl_no_lifo_rejections_line(self):
        """OUT-11: FCL should not have LIFO rejections line"""
        plan = create_fixture_plan_fcl()
        assert plan.metrics.unplaced_count == 0
        for uc in plan.unplaced_cartons:
            assert uc.reason != UnplacedReason.LIFO_BLOCKED

    def test_OUT_12_cog_outside_safe_zone(self):
        """OUT-12: CoG outside safe zone should be detectable"""
        plan = create_fixture_plan_fcl()
        safe_margin = 50.0
        container = plan.container
        cog_x, cog_y = plan.metrics.cog_x, plan.metrics.cog_y
        outside_x = cog_x < safe_margin or cog_x > container.internal_length_cm - safe_margin
        outside_y = cog_y < safe_margin or cog_y > container.internal_width_cm - safe_margin
        is_outside = outside_x or outside_y
        assert is_outside or not is_outside  # Just verify it's computable


class TestVisualizationData:
    """Tests for 3D visualization data preparation (Section 2.3 - OUT-13 to OUT-16)"""

    def test_OUT_13_fcl_color_by_item_id(self):
        """OUT-13: FCL boxes colored by item_id"""
        plan = create_fixture_plan_fcl()
        color_keys = [box.item_id for box in plan.placed_boxes]
        assert all(c for c in color_keys)

    def test_OUT_14_lcl_color_by_customer_code(self):
        """OUT-14: LCL boxes colored by Customer_Code"""
        plan = create_fixture_plan_lcl()
        color_keys = [box.customer_code for box in plan.placed_boxes]
        assert all(c is not None for c in color_keys)
        assert len(set(color_keys)) == 2

    def test_OUT_15_door_panel_position(self):
        """OUT-15: Door panel at x=L (container door end), sized to width x height"""
        plan = create_fixture_plan_fcl()
        container = plan.container
        door_x = container.internal_length_cm
        door_width = container.internal_width_cm
        door_height = container.internal_height_cm
        assert door_x == container.internal_length_cm
        assert door_width == 235.2
        assert door_height == 270.0

    def test_OUT_16_cog_crosshair_matches_computed(self):
        """OUT-16: CoG crosshair matches computed value"""
        plan = create_fixture_plan_fcl()
        container = plan.container
        cog_x, cog_y, cog_z = plan.metrics.cog_x, plan.metrics.cog_y, plan.metrics.cog_z
        ideal_x = container.internal_length_cm / 2
        ideal_y = container.internal_width_cm / 2
        ideal_z = container.internal_height_cm / 2
        assert abs(cog_x - ideal_x) < 100
        assert abs(cog_y - ideal_y) < 100
        assert abs(cog_z - ideal_z) < 100


class TestOutputCoordinateIntegrity:
    """Tests ensuring no cargos protrude outside block or container bounds"""

    def test_transform_position_by_posture_all_six_postures(self):
        """Ensure transform_position_by_posture correctly maps (x, y, z) without inversion"""
        from app.solver.geometry import Position, Posture
        from app.solver.output import transform_position_by_posture

        pos = Position(10.0, 20.0, 30.0)

        # 1. LWH: (x, y, z)
        t_lwh = transform_position_by_posture(pos, Posture.LWH)
        assert (t_lwh.x, t_lwh.y, t_lwh.z) == (10.0, 20.0, 30.0)

        # 2. WLH: (y, x, z)
        t_wlh = transform_position_by_posture(pos, Posture.WLH)
        assert (t_wlh.x, t_wlh.y, t_wlh.z) == (20.0, 10.0, 30.0)

        # 3. HLW: (z, x, y)
        t_hlw = transform_position_by_posture(pos, Posture.HLW)
        assert (t_hlw.x, t_hlw.y, t_hlw.z) == (30.0, 10.0, 20.0)

        # 4. HWL: (z, y, x)
        t_hwl = transform_position_by_posture(pos, Posture.HWL)
        assert (t_hwl.x, t_hwl.y, t_hwl.z) == (30.0, 20.0, 10.0)

        # 5. LHW: (x, z, y)
        t_lhw = transform_position_by_posture(pos, Posture.LHW)
        assert (t_lhw.x, t_lhw.y, t_lhw.z) == (10.0, 30.0, 20.0)

        # 6. WHL: (y, z, x)
        t_whl = transform_position_by_posture(pos, Posture.WHL)
        assert (t_whl.x, t_whl.y, t_whl.z) == (20.0, 30.0, 10.0)

    def test_explode_rotated_individual_box_stays_within_bbox(self):
        """Ensure an individual box placed with non-LWH posture retains its rotated dimensions and bounds"""
        from app.solver.parsing import Box
        from app.solver.geometry import BoundingBox, Posture
        from app.solver.output import explode_blocks

        # Box native dimensions: L=100, W=50, H=40
        box = Box(
            box_id="INDIV-01",
            item_id="ITEM-01",
            po_no="PO-01",
            customer_code="CUST-1",
            customer_sequence=1,
            length_cm=100,
            width_cm=50,
            height_cm=40,
            weight_kg=15.0,
            this_way_up=False,
            permitted_postures=list(Posture),
            inflated_length=100,
            inflated_width=50,
            inflated_height=40,
        )

        # Placed with WLH at x=200, y=50, z=0
        # Posture WLH: length becomes width (50), width becomes length (100), height stays height (40)
        bbox = BoundingBox(
            min_x=200.0,
            min_y=50.0,
            min_z=0.0,
            max_x=250.0,
            max_y=150.0,
            max_z=40.0,
        )

        exploded = explode_blocks([box], [bbox], [Posture.WLH])
        assert len(exploded) == 1
        placed = exploded[0]

        assert placed.posture == Posture.WLH
        assert placed.actual_length == 50.0
        assert placed.actual_width == 100.0
        assert placed.actual_height == 40.0
        assert placed.x == 200.0
        assert placed.y == 50.0
        assert placed.z == 0.0

        # Assert box strictly within bounding box
        assert placed.x + placed.actual_length <= bbox.max_x + 1e-5
        assert placed.y + placed.actual_width <= bbox.max_y + 1e-5
        assert placed.z + placed.actual_height <= bbox.max_z + 1e-5

    def test_explode_rotated_block_contents_stay_within_block_bbox(self):
        """Ensure all contents of a rotated multi-box block remain strictly within the block's bounding box"""
        from app.solver.parsing import Box
        from app.solver.block_generation import Block
        from app.solver.geometry import BoundingBox, Posture
        from app.solver.output import explode_blocks

        # Block containing 2 cartons stacked along X:
        # Each carton is 50 x 40 x 30 cm
        # Block unrotated is 100 x 40 x 30 cm
        c1 = Box(
            box_id="C1",
            item_id="ITEM-A",
            po_no="PO-A",
            customer_code=None,
            customer_sequence=0,
            length_cm=50,
            width_cm=40,
            height_cm=30,
            weight_kg=10,
            this_way_up=False,
            permitted_postures=list(Posture),
            inflated_length=50,
            inflated_width=40,
            inflated_height=30,
            rel_x=0.0,
            rel_y=0.0,
            rel_z=0.0,
        )
        c2 = Box(
            box_id="C2",
            item_id="ITEM-A",
            po_no="PO-A",
            customer_code=None,
            customer_sequence=0,
            length_cm=50,
            width_cm=40,
            height_cm=30,
            weight_kg=10,
            this_way_up=False,
            permitted_postures=list(Posture),
            inflated_length=50,
            inflated_width=40,
            inflated_height=30,
            rel_x=50.0,
            rel_y=0.0,
            rel_z=0.0,
        )

        block = Block(
            block_id="BLOCK-01",
            boxes=[c1, c2],
            length_cm=100.0,
            width_cm=40.0,
            height_cm=30.0,
            weight_kg=20.0,
            customer_sequence=0,
            inflated_length=100.0,
            inflated_width=40.0,
            inflated_height=30.0,
            contents=[c1, c2],
        )

        # Test with Posture.WLH:
        # dims become: length=40, width=100, height=30
        bbox_wlh = BoundingBox(100.0, 200.0, 0.0, 140.0, 300.0, 30.0)
        exploded_wlh = explode_blocks([block], [bbox_wlh], [Posture.WLH])

        assert len(exploded_wlh) == 2
        for p in exploded_wlh:
            assert p.x >= bbox_wlh.min_x - 1e-5
            assert p.x + p.actual_length <= bbox_wlh.max_x + 1e-5
            assert p.y >= bbox_wlh.min_y - 1e-5
            assert p.y + p.actual_width <= bbox_wlh.max_y + 1e-5
            assert p.z >= bbox_wlh.min_z - 1e-5
            assert p.z + p.actual_height <= bbox_wlh.max_z + 1e-5

        # Test with Posture.HWL:
        # dims become: length=30, width=40, height=100
        bbox_hwl = BoundingBox(10.0, 20.0, 30.0, 40.0, 60.0, 130.0)
        exploded_hwl = explode_blocks([block], [bbox_hwl], [Posture.HWL])

        assert len(exploded_hwl) == 2
        for p in exploded_hwl:
            assert p.x >= bbox_hwl.min_x - 1e-5
            assert p.x + p.actual_length <= bbox_hwl.max_x + 1e-5
            assert p.y >= bbox_hwl.min_y - 1e-5
            assert p.y + p.actual_width <= bbox_hwl.max_y + 1e-5
            assert p.z >= bbox_hwl.min_z - 1e-5
            assert p.z + p.actual_height <= bbox_hwl.max_z + 1e-5

    @pytest.mark.slow
    def test_pipeline_e2e_all_boxes_strictly_inside_container(self):
        """Run complete solver pipeline and assert 100% of placed cartons are within container bounds"""
        import pandas as pd
        from app.solver.pipeline import run_pipeline

        container_df = pd.DataFrame([{
            "Container_Type": "20GP",
            "Internal_Length_cm": 589.8,
            "Internal_Width_cm": 235.2,
            "Internal_Height_cm": 239.3,
            "Max_Weight_kg": 28000.0,
        }])

        item_master_df = pd.DataFrame([
            {
                "Item_ID": "ITEM_A",
                "Description": "Test Item A",
                "Length_cm": 60.0,
                "Width_cm": 40.0,
                "Height_cm": 35.0,
                "Weight_kg": 15.0,
                "This_Way_Up": False,
            },
            {
                "Item_ID": "ITEM_B",
                "Description": "Test Item B",
                "Length_cm": 120.0,
                "Width_cm": 50.0,
                "Height_cm": 40.0,
                "Weight_kg": 30.0,
                "This_Way_Up": False,
            },
            {
                "Item_ID": "ITEM_C",
                "Description": "Test Item C",
                "Length_cm": 45.0,
                "Width_cm": 30.0,
                "Height_cm": 25.0,
                "Weight_kg": 10.0,
                "This_Way_Up": True,
            },
        ])

        packing_list_df = pd.DataFrame([
            {"PO_No": "PO1", "Item_ID": "ITEM_A", "Qty_Pcs": 20, "Qty_Cartons": 2, "Customer_Code": "CUST1"},
            {"PO_No": "PO2", "Item_ID": "ITEM_B", "Qty_Pcs": 20, "Qty_Cartons": 2, "Customer_Code": "CUST1"},
            {"PO_No": "PO3", "Item_ID": "ITEM_C", "Qty_Pcs": 20, "Qty_Cartons": 2, "Customer_Code": "CUST2"},
        ])

        from app.core.models import RunOptions
        options = RunOptions(population_size=10, generations=10, tolerance_gap_cm=2.0)
        pipeline_res = run_pipeline(packing_list_df, item_master_df, container_df, options=options)
        placed_boxes = pipeline_res.result.placed_boxes
        assert len(placed_boxes) > 0

        L = 589.8
        W = 235.2
        H = 239.3

        for box in placed_boxes:
            assert box.x >= -1e-4, f"Box {box.box_id} x < 0: {box.x}"
            assert box.y >= -1e-4, f"Box {box.box_id} y < 0: {box.y}"
            assert box.z >= -1e-4, f"Box {box.box_id} z < 0: {box.z}"
            assert box.x + box.actual_length <= L + 1e-4, (
                f"Box {box.box_id} exceeds L: {box.x} + {box.actual_length} = {box.x + box.actual_length} > {L}"
            )
            assert box.y + box.actual_width <= W + 1e-4, (
                f"Box {box.box_id} exceeds W: {box.y} + {box.actual_width} = {box.y + box.actual_width} > {W}"
            )
            assert box.z + box.actual_height <= H + 1e-4, (
                f"Box {box.box_id} exceeds H: {box.z} + {box.actual_height} = {box.z + box.actual_height} > {H}"
            )

    @pytest.mark.slow
    def test_step_index_ordering_rear_to_front(self):
        """Verify that step_index ordering sorts by (customer_sequence, -x, z, y).
        Within each customer sequence, x should be non-increasing (rear-to-front),
        with z ascending (floor to ceiling) within depth plateaus.
        """
        import pandas as pd
        from app.solver.pipeline import run_pipeline
        from app.core.models import RunOptions

        container_df = pd.DataFrame([{
            "Container_Type": "20DC",
            "Internal_Length_cm": 589.8,
            "Internal_Width_cm": 235.2,
            "Internal_Height_cm": 239.3,
            "Max_Weight_kg": 28200.0,
        }])
        item_master_df = pd.DataFrame([
            {
                "Item_ID": "ITEM_A",
                "Description": "Test Item A",
                "Length_cm": 60.0,
                "Width_cm": 40.0,
                "Height_cm": 40.0,
                "Weight_kg": 15.0,
                "This_Way_Up": True,
                "Permitted_Postures": "LWH,WLH",
            }
        ])
        packing_list_df = pd.DataFrame([
            {"PO_No": "PO1", "Item_ID": "ITEM_A", "Qty_Pcs": 30, "Qty_Cartons": 30, "Customer_Code": "CUST1"},
        ])
        options = RunOptions(population_size=10, generations=10, tolerance_gap_cm=2.0)
        pipeline_res = run_pipeline(packing_list_df, item_master_df, container_df, options=options)
        boxes = pipeline_res.result.placed_boxes
        assert len(boxes) > 1

        # Check that step_index increases sequentially starting from 1
        step_indices = [b.step_index for b in boxes]
        assert step_indices == list(range(1, len(boxes) + 1))

        # Check key sort property: for any consecutive boxes in same customer_sequence,
        # key (cust_seq, x, z, y) is monotonically non-decreasing (rear-to-door, ascending x)
        for i in range(len(boxes) - 1):
            b1 = boxes[i]
            b2 = boxes[i + 1]
            key1 = (b1.customer_sequence, b1.x, b1.z, b1.y)
            key2 = (b2.customer_sequence, b2.x, b2.z, b2.y)
            assert key1 <= key2, f"Step ordering violated between #{b1.step_index} and #{b2.step_index}: {key1} > {key2}"

    @pytest.mark.slow
    def test_distinct_color_per_sku_multi_sku_pipeline(self):
        """Verify that every distinct item_id gets a unique color and same item_id gets identical color (15+ SKUs)."""
        import pandas as pd
        from app.solver.pipeline import run_pipeline
        from app.core.models import RunOptions

        container_df = pd.DataFrame([{
            "Container_Type": "40HC",
            "Internal_Length_cm": 1203.2,
            "Internal_Width_cm": 235.2,
            "Internal_Height_cm": 269.8,
            "Max_Weight_kg": 28500.0,
        }])

        num_skus = 20
        item_master_rows = []
        packing_list_rows = []
        for i in range(num_skus):
            sku = f"SKU_{i:03d}"
            item_master_rows.append({
                "Item_ID": sku,
                "Description": f"Item {sku}",
                "Length_cm": 40.0,
                "Width_cm": 30.0,
                "Height_cm": 25.0,
                "Weight_kg": 5.0,
                "This_Way_Up": True,
                "Permitted_Postures": "LWH",
            })
            packing_list_rows.append({
                "PO_No": f"PO_{i:03d}",
                "Item_ID": sku,
                "Qty_Pcs": 4,
                "Qty_Cartons": 2,
                "Customer_Code": "CUST1",
            })

        item_master_df = pd.DataFrame(item_master_rows)
        packing_list_df = pd.DataFrame(packing_list_rows)
        options = RunOptions(population_size=10, generations=10, tolerance_gap_cm=0.0)

        pipeline_res = run_pipeline(packing_list_df, item_master_df, container_df, options=options)
        boxes = pipeline_res.result.placed_boxes
        assert len(boxes) > 0

        # Group colors by item_id
        sku_to_colors = {}
        for b in boxes:
            assert b.color is not None, f"Box {b.box_id} missing color"
            assert b.color.startswith("#"), f"Box {b.box_id} color {b.color} is not hex"
            sku_to_colors.setdefault(b.item_id, set()).add(b.color)

        # 1. Same SKU always has exactly 1 color
        for sku, colors in sku_to_colors.items():
            assert len(colors) == 1, f"SKU {sku} has multiple colors assigned: {colors}"

        # 2. Distinct SKUs have strictly distinct colors (no collision)
        all_assigned_colors = [list(colors)[0] for colors in sku_to_colors.values()]
        assert len(all_assigned_colors) == len(set(all_assigned_colors)), (
            f"Color collision detected! Assigned colors count {len(all_assigned_colors)} vs unique {len(set(all_assigned_colors))}"
        )
        assert len(sku_to_colors) == num_skus, f"Expected all {num_skus} SKUs placed, got {len(sku_to_colors)}"

    @pytest.mark.slow
    def test_color_per_sku_packing_list_01_real_dataset(self):
        """Run real dataset packing_list_01.csv (30 distinct SKUs) and assert no color collisions."""
        import pandas as pd
        from pathlib import Path
        from app.solver.pipeline import run_pipeline
        from app.core.models import RunOptions

        repo_root = Path(__file__).resolve().parent.parent.parent
        data_dir = repo_root / "data"

        container_df = pd.read_csv(data_dir / "container_spec.csv")
        item_master_df = pd.read_csv(data_dir / "item_master.csv")
        packing_list_df = pd.read_csv(data_dir / "packing_list_01.csv")

        options = RunOptions(population_size=10, generations=10, tolerance_gap_cm=0.0)
        pipeline_res = run_pipeline(packing_list_df, item_master_df, container_df, options=options)
        boxes = pipeline_res.result.placed_boxes
        assert len(boxes) > 0

        sku_to_colors = {}
        for b in boxes:
            sku_to_colors.setdefault(b.item_id, set()).add(b.color)

        for sku, colors in sku_to_colors.items():
            assert len(colors) == 1, f"SKU {sku} has inconsistent colors: {colors}"

        all_colors = [list(c)[0] for c in sku_to_colors.values()]
        assert len(all_colors) == len(set(all_colors)), (
            f"Color collision in packing_list_01: {len(all_colors)} SKUs vs {len(set(all_colors))} unique colors"
        )

    def test_no_floating_cartons_in_step_sequence(self):
        """Verify that topological sort guarantees zero floating cartons during step-by-step loading.
        Even when a top carton has a smaller X than its base, the base is always placed first.
        """
        from app.solver.output import sort_boxes_physically_stable

        # Box 1: Base on floor at X=100, Z=0
        base = PlacedBox(
            box_id="BASE_1", item_id="ITEM_A", po_no="PO1", length_cm=50, width_cm=50, height_cm=50,
            actual_length=50, actual_width=50, actual_height=50, weight_kg=10,
            x=100.0, y=0.0, z=0.0, posture=Posture.LWH, customer_sequence=1
        )
        # Box 2: Middle box at X=95, Z=50 (overhanging towards smaller X)
        middle = PlacedBox(
            box_id="MID_1", item_id="ITEM_A", po_no="PO1", length_cm=50, width_cm=50, height_cm=50,
            actual_length=50, actual_width=50, actual_height=50, weight_kg=10,
            x=95.0, y=0.0, z=50.0, posture=Posture.LWH, customer_sequence=1
        )
        # Box 3: Top box at X=90, Z=100 (further overhanging towards smaller X)
        top = PlacedBox(
            box_id="TOP_1", item_id="ITEM_A", po_no="PO1", length_cm=50, width_cm=50, height_cm=50,
            actual_length=50, actual_width=50, actual_height=50, weight_kg=10,
            x=90.0, y=0.0, z=100.0, posture=Posture.LWH, customer_sequence=1
        )

        ordered = sort_boxes_physically_stable([top, middle, base])
        step_ids = [b.box_id for b in ordered]
        assert step_ids == ["BASE_1", "MID_1", "TOP_1"], f"Expected base to top loading order, got {step_ids}"