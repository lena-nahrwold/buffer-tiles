import re
import json
import pdal
from pathlib import Path

def crop_tiles(params):
    INPUT_DIR = params.input_dir
    OUTPUT_DIR = params.output_dir

    TILE_SIZE = params.tile_size

    FILENAME_RE = re.compile(r"(\d+)_(\d+)\.laz")

    for tile_path in INPUT_DIR.glob("*.laz"):
        m = FILENAME_RE.match(tile_path.name)
        if not m:
            continue

        x, y = map(int, m.groups())

        xmin = x
        ymin = y
        xmax = x + TILE_SIZE
        ymax = y + TILE_SIZE

        bounds = f"([{xmin},{xmax}],[{ymin},{ymax}])"

        out_file = OUTPUT_DIR / tile_path.name

        pipeline = {
            "pipeline": [
                str(tile_path),
                {
                    "type": "filters.crop",
                    "bounds": bounds
                },
                {
                    "type": "writers.las",
                    "filename": str(out_file),
                    "extra_dims": "all",
                    "compression": "laszip"
                }
            ]
        }

        p = pdal.Pipeline(json.dumps(pipeline))
        p.execute()
