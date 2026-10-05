import pandas as pd
import pytest
from app.core.models import RunOptions
from app.solver.pipeline import run_pipeline


@pytest.fixture
def sample_data():
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
        {"PO_No": "PO1", "Item_ID": "ITEM_A", "Qty_Pcs": 30, "Qty_Cartons": 4, "Customer_Code": "CUST1"},
        {"PO_No": "PO2", "Item_ID": "ITEM_B", "Qty_Pcs": 20, "Qty_Cartons": 3, "Customer_Code": "CUST1"},
        {"PO_No": "PO3", "Item_ID": "ITEM_C", "Qty_Pcs": 40, "Qty_Cartons": 5, "Customer_Code": "CUST2"},
    ])

    return container_df, item_master_df, packing_list_df


@pytest.mark.slow
def test_pipeline_seed_reproducibility(sample_data):
    container_df, item_master_df, packing_list_df = sample_data

    options1 = RunOptions(seed=12345, population_size=12, generations=10, tolerance_gap_cm=2.0)
    options2 = RunOptions(seed=12345, population_size=12, generations=10, tolerance_gap_cm=2.0)

    res1 = run_pipeline(packing_list_df, item_master_df, container_df, options=options1)
    res2 = run_pipeline(packing_list_df, item_master_df, container_df, options=options2)

    boxes1 = res1.result.placed_boxes
    boxes2 = res2.result.placed_boxes

    assert len(boxes1) == len(boxes2)
    assert res1.result.seed == 12345
    assert res2.result.seed == 12345

    coords1 = [(b.box_id, b.x, b.y, b.z, b.actual_length, b.actual_width, b.actual_height, b.posture) for b in boxes1]
    coords2 = [(b.box_id, b.x, b.y, b.z, b.actual_length, b.actual_width, b.actual_height, b.posture) for b in boxes2]

    assert coords1 == coords2
    assert res1.result.metrics.fill_rate == pytest.approx(res2.result.metrics.fill_rate, rel=1e-5)
    assert res1.result.metrics.weight_utilization == pytest.approx(res2.result.metrics.weight_utilization, rel=1e-5)
