from typing import Dict, Tuple, Optional

def overlap_1d(a_min, a_max, b_min, b_max) -> float:
    """Returns overlap length between two 1D segments."""
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def find_spatial_neighbors(
    tile_boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tiles: Dict[str, Tuple[float, float, float, float]],
    tolerance: float = 0.01
) -> Dict[str, Optional[str]]:
    """
    Find neighboring tiles based on shared edges (not overlap area).

    tolerance:
        Max distance between edges to still consider them touching
        (useful for floating-point or reprojection errors)
    """

    minx_a, maxx_a, miny_a, maxy_a = tile_boundary

    neighbors = {
        "east": None,
        "west": None,
        "north": None,
        "south": None
    }

    best = {
        "east": 0.0,
        "west": 0.0,
        "north": 0.0,
        "south": 0.0
    }

    for other_name, (minx_b, maxx_b, miny_b, maxy_b) in all_tiles.items():
        if other_name == tile_name:
            continue

        # EAST: this.maxx ≈ other.minx
        if abs(maxx_a - minx_b) <= tolerance:
            y_overlap = overlap_1d(miny_a, maxy_a, miny_b, maxy_b)
            if y_overlap > best["east"]:
                best["east"] = y_overlap
                neighbors["east"] = other_name

        # WEST: this.minx ≈ other.maxx
        if abs(minx_a - maxx_b) <= tolerance:
            y_overlap = overlap_1d(miny_a, maxy_a, miny_b, maxy_b)
            if y_overlap > best["west"]:
                best["west"] = y_overlap
                neighbors["west"] = other_name

        # NORTH: this.maxy ≈ other.miny
        if abs(maxy_a - miny_b) <= tolerance:
            x_overlap = overlap_1d(minx_a, maxx_a, minx_b, maxx_b)
            if x_overlap > best["north"]:
                best["north"] = x_overlap
                neighbors["north"] = other_name

        # SOUTH: this.miny ≈ other.maxy
        if abs(miny_a - maxy_b) <= tolerance:
            x_overlap = overlap_1d(minx_a, maxx_a, minx_b, maxx_b)
            if x_overlap > best["south"]:
                best["south"] = x_overlap
                neighbors["south"] = other_name

    return neighbors
