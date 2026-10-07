"""Data integrity checks across item_master and all packing lists."""
from pathlib import Path
import pandas as pd


def get_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def test_item_master_integrity():
    data_dir = get_data_dir()
    im_path = data_dir / "item_master.csv"
    assert im_path.exists(), f"item_master.csv not found at {im_path}"

    im = pd.read_csv(im_path)
    # No duplicate Item_ID
    assert not im["Item_ID"].duplicated().any(), f"Duplicate Item_IDs found: {im[im['Item_ID'].duplicated()]['Item_ID'].tolist()}"

    # All dimensions and weights > 0
    for col in ["Length_cm", "Width_cm", "Height_cm", "Weight_kg"]:
        assert (im[col] > 0).all(), f"Found non-positive values in {col}: {im[im[col] <= 0][['Item_ID', col]].to_dict('records')}"


def test_packing_lists_reference_valid_items():
    data_dir = get_data_dir()
    im = pd.read_csv(data_dir / "item_master.csv")
    valid_item_ids = set(im["Item_ID"].astype(str).str.strip())

    pl_names = [f"packing_list_{i:02d}.csv" for i in range(1, 7)] + [
        "packing_list_BH-147.csv",
        "packing_list_MARTIN-40HQ-2of2.csv",
        "packing_list_NJR26-127.csv",
        "packing_list_TMI-085.csv",
    ]
    pl_files = [data_dir / name for name in pl_names if (data_dir / name).exists()]
    assert len(pl_files) >= 6, "Benchmark packing list files missing"

    for pl_file in pl_files:
        pl = pd.read_csv(pl_file)
        assert "Item_ID" in pl.columns, f"Missing Item_ID column in {pl_file.name}"
        pl_items = set(pl["Item_ID"].astype(str).str.strip())
        missing = pl_items - valid_item_ids
        assert not missing, f"{pl_file.name} references Item_IDs not in item_master.csv: {missing}"
