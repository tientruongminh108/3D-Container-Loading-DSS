import math
import random
import colorsys
from app.solver.colors import generate_sku_color_map


def _hex_to_rgb(hex_str: str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _min_pairwise_distance(color_map):
    rgbs = [_hex_to_rgb(c) for c in color_map.values()]
    min_d = float("inf")
    for i in range(len(rgbs)):
        for j in range(i + 1, len(rgbs)):
            c1, c2 = rgbs[i], rgbs[j]
            d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
            if d < min_d:
                min_d = d
    return min_d


def _legacy_min_pairwise_distance(n: int):
    rgbs = []
    for rank in range(n):
        h = rank / n
        r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.55)
        rgbs.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    min_d = float("inf")
    for i in range(len(rgbs)):
        for j in range(i + 1, len(rgbs)):
            c1, c2 = rgbs[i], rgbs[j]
            d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)
            if d < min_d:
                min_d = d
    return min_d


def test_colors_are_deterministic_per_sorted_sku_list():
    skus = [f"SKU_{i:03d}" for i in range(25)]
    shuffled_1 = list(skus)
    random.Random(1).shuffle(shuffled_1)
    shuffled_2 = list(skus)
    random.Random(2).shuffle(shuffled_2)

    map1 = generate_sku_color_map(shuffled_1)
    map2 = generate_sku_color_map(shuffled_2)
    assert map1 == map2
    assert len(map1) == 25
    for sku, color in map1.items():
        assert color.startswith("#")
        assert len(color) == 7


def test_minimum_pairwise_distance_clearly_larger_than_legacy():
    # Measured legacy: 30 SKUs = 25.00, 100 SKUs = 5.39
    # Measured new:    30 SKUs = 34.48, 100 SKUs = 29.83
    legacy_30 = _legacy_min_pairwise_distance(30)
    legacy_100 = _legacy_min_pairwise_distance(100)

    skus_30 = [f"ITEM_30_{i:03d}" for i in range(30)]
    new_map_30 = generate_sku_color_map(skus_30)
    new_dist_30 = _min_pairwise_distance(new_map_30)

    skus_100 = [f"ITEM_100_{i:03d}" for i in range(100)]
    new_map_100 = generate_sku_color_map(skus_100)
    new_dist_100 = _min_pairwise_distance(new_map_100)

    assert new_dist_30 > legacy_30, f"30 SKUs: {new_dist_30:.2f} must exceed legacy {legacy_30:.2f}"
    assert new_dist_100 > legacy_100, f"100 SKUs: {new_dist_100:.2f} must exceed legacy {legacy_100:.2f}"
    # Verify exact improvements
    assert new_dist_30 >= 30.0
    assert new_dist_100 >= 20.0
