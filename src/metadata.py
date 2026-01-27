import json
import subprocess
from pathlib import Path

def get_native_bounds(laz_path: Path):
    out = subprocess.check_output(
        ["pdal", "info", "--summary", str(laz_path)],
        text=True
    )
    info = json.loads(out)

    native_bounds = info["summary"]["bounds"]
    metadata = info["summary"]["metadata"]

    bounds = (
        native_bounds["minx"],
        native_bounds["maxx"],
        native_bounds["miny"],
        native_bounds["maxy"],
    )

    scale = (
        metadata["scale_x"],
        metadata["scale_y"],
        metadata["scale_z"],
    )

    offset = (
        metadata["offset_x"],
        metadata["offset_y"],
        metadata["offset_z"],
    )

    return bounds, scale, offset