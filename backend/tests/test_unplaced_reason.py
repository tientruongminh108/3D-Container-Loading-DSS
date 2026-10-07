import pandas as pd
from app.solver.pipeline import run_pipeline
from app.core.models import RunOptions, UnplacedReason
from app.solver.output import _map_unplaced_reason


def test_map_unplaced_reason():
    """Verify solver reason strings correctly map to UnplacedReason enum."""
    assert _map_unplaced_reason("weight_capacity") == UnplacedReason.WEIGHT_CAPACITY
    assert _map_unplaced_reason("WEIGHT_CAPACITY") == UnplacedReason.WEIGHT_CAPACITY
    assert _map_unplaced_reason("lifo_blocked") == UnplacedReason.LIFO_BLOCKED
    assert _map_unplaced_reason("swapped_out") == UnplacedReason.NO_SPACE
    assert _map_unplaced_reason("unknown_reason") == UnplacedReason.NO_SPACE
    assert _map_unplaced_reason(None) == UnplacedReason.NO_SPACE
    assert _map_unplaced_reason(UnplacedReason.WEIGHT_CAPACITY) == UnplacedReason.WEIGHT_CAPACITY


def test_weight_capacity_is_reported_when_weight_is_binding():
    """When container volume is abundant but max_weight is binding, unplaced cartons
    must report UnplacedReason.WEIGHT_CAPACITY instead of NO_SPACE."""
    container_df = pd.DataFrame([{
        "Container_Type": "TEST_LIGHT",
        "Internal_Length_cm": 1000.0,
        "Internal_Width_cm": 1000.0,
        "Internal_Height_cm": 1000.0,
        "Max_Weight_kg": 50.0,  # Only allows 1 of the 40kg cartons
    }])

    item_master_df = pd.DataFrame([{
        "Item_ID": "HEAVY-1",
        "Description": "Heavy item",
        "Length_cm": 50.0,
        "Width_cm": 50.0,
        "Height_cm": 50.0,
        "Weight_kg": 40.0,
        "This_Way_Up": False,
    }])

    packing_list_df = pd.DataFrame([{
        "Item_ID": "HEAVY-1",
        "PO_No": "PO-1",
        "Customer_Code": "",
        "Description": "Heavy item",
        "Qty_Pcs": 2,
        "Qty_Cartons": 2,
    }])

    opts = RunOptions(population_size=10, generations=10, seed=42)
    pipeline_res = run_pipeline(packing_list_df, item_master_df, container_df, options=opts)
    result = pipeline_res.result

    assert len(result.placed_boxes) == 1
    assert len(result.unplaced_cartons) == 1
    unplaced = result.unplaced_cartons[0]
    assert unplaced.reason == UnplacedReason.WEIGHT_CAPACITY
