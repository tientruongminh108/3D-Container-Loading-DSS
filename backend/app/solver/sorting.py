from typing import List
from collections import defaultdict
from app.solver.parsing import Box
from app.solver.block_generation import Block
from app.config import get_settings
from app.solver.utils import get_unit_item_id


def initial_sort(boxes: List[Box], shipment_type: str = "FCL") -> List[Box]:
    """Sort boxes per unit sort key:
    (-volume, -weight, item_id, box_id)
    (large volume first, then heavy first)
    """
    def sort_key(box: Box):
        vol = box.length_cm * box.width_cm * box.height_cm
        item_id = box.item_id or ""
        box_id = box.box_id or ""
        return (-vol, -box.weight_kg, item_id, box_id)

    return sorted(boxes, key=sort_key)


def resort_after_blocks(units: List[Block], shipment_type: str = "FCL") -> List[Block]:
    """Re-sort blocks after block generation:
    (-volume, -weight, item_id, unit_id)
    """
    settings = get_settings()

    if getattr(settings, "SORT_ITEM_AFFINITY", False):
        # Legacy Item Affinity Clustering
        group_max_vol = defaultdict(float)
        group_tot_vol = defaultdict(float)
        group_max_wt = defaultdict(float)
        group_tot_wt = defaultdict(float)

        for unit in units:
            item_id = get_unit_item_id(unit)
            wt = unit.weight_kg
            vol = unit.length_cm * unit.width_cm * unit.height_cm
            key = item_id
            if vol > group_max_vol[key]:
                group_max_vol[key] = vol
            group_tot_vol[key] += vol
            if wt > group_max_wt[key]:
                group_max_wt[key] = wt
            group_tot_wt[key] += wt

        def legacy_sort_key(unit):
            item_id = get_unit_item_id(unit)
            wt = unit.weight_kg
            vol = unit.length_cm * unit.width_cm * unit.height_cm
            grp_key = item_id

            keys = []
            if item_id:
                keys.append(-group_max_vol[grp_key])
                keys.append(-group_tot_vol[grp_key])
                keys.append(-group_max_wt[grp_key])
                keys.append(-group_tot_wt[grp_key])
                keys.append(item_id)

            keys.append(-vol)
            keys.append(-wt)
            return tuple(keys)

        return sorted(units, key=legacy_sort_key)

    # Authoritative Unit Sort Key: (-vol, -wt, item_id, unit_id)
    def sort_key(unit):
        wt = unit.weight_kg
        vol = unit.length_cm * unit.width_cm * unit.height_cm
        item_id = get_unit_item_id(unit) or ""
        unit_id = getattr(unit, 'block_id', getattr(unit, 'box_id', ''))
        return (-vol, -wt, item_id, unit_id)

    return sorted(units, key=sort_key)