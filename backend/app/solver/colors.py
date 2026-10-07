"""Per-SKU color assignment with golden-angle hue distribution and alternating bands.

Ensures high perceptual contrast even with 30-100+ distinct SKUs, replacing the legacy
linear hue generator which produced indistinguishable adjacent colors (min distance ~5-25 RGB units).
"""
import colorsys
from typing import Dict, Iterable, Optional

GOLDEN_RATIO_CONJUGATE = 0.618033988749895
LIGHTNESS_BANDS = (0.42, 0.58, 0.50)
SATURATION_BANDS = (0.65, 0.90)


def generate_sku_color_map(skus: Iterable[Optional[str]]) -> Dict[str, str]:
    """Return a deterministic mapping from SKU item_id to hexadecimal color string.

    The distinct, non-None SKUs are sorted alphabetically so the assignment is reproducible.
    Hue is stepped by the golden angle, with interleaved lightness and saturation bands
    to maximize pairwise Euclidean RGB distance across arbitrary SKU counts.
    """
    distinct_skus = sorted(list({s for s in skus if s is not None}))
    if not distinct_skus:
        return {}

    n_l = len(LIGHTNESS_BANDS)
    n_s = len(SATURATION_BANDS)
    sku_color_map: Dict[str, str] = {}

    for i, sku in enumerate(distinct_skus):
        hue = (i * GOLDEN_RATIO_CONJUGATE) % 1.0
        lightness = LIGHTNESS_BANDS[i % n_l]
        saturation = SATURATION_BANDS[(i // n_l) % n_s]
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
        sku_color_map[sku] = hex_color

    return sku_color_map
