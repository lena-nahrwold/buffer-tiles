import os
import re
import json
import pdal
from pathlib import Path

def buffer_tiles(args):
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    BUFFER = args.buffer_size         
    TILE_SIZE = None

    FILENAME_RE = re.compile(r"(\d+)_(\d+)\.laz")

    OUTPUT_DIR.mkdir(exist_ok=True)

    tiles = {}
    xs, ys = set(), set()

    for f in INPUT_DIR.glob("*.laz"):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        x, y = map(int, m.groups())
        tiles[(x, y)] = f
        xs.add(x)
        ys.add(y)

    if not tiles:
        raise RuntimeError("No tiles found")

    if TILE_SIZE is None:
        xs = sorted(xs)
        ys = sorted(ys)

        dx = min(xs[i+1] - xs[i] for i in range(len(xs) - 1))
        dy = min(ys[i+1] - ys[i] for i in range(len(ys) - 1))

        TILE_SIZE = min(dx, dy)

    print(f"Detected tile size: {TILE_SIZE}")

    # Neighbor offsets (8-connected)
    offsets = [
        (-TILE_SIZE, -TILE_SIZE), (0, -TILE_SIZE), (TILE_SIZE, -TILE_SIZE),
        (-TILE_SIZE, 0),                              (TILE_SIZE, 0),
        (-TILE_SIZE, TILE_SIZE),  (0, TILE_SIZE),  (TILE_SIZE, TILE_SIZE),
    ]

    for (x, y), tile_path in tiles.items():
        neighbors = []

        for dx, dy in offsets:
            key = (x + dx, y + dy)
            if key in tiles:
                neighbors.append(tiles[key])

        # Buffered bounds
        xmin = x - BUFFER
        ymin = y - BUFFER
        xmax = x + TILE_SIZE + BUFFER
        ymax = y + TILE_SIZE + BUFFER

        bounds = f"([{xmin},{xmax}],[{ymin},{ymax}])"

        out_file = OUTPUT_DIR / tile_path.name

        pipeline = {
            "pipeline": [
                str(tile_path),
                *map(str, neighbors),
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