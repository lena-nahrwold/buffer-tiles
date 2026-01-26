import re
import json
import pdal
import subprocess
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_native_bounds(laz_path: Path):
    out = subprocess.check_output(
        ["pdal", "info", str(laz_path)],
        text=True
    )
    info = json.loads(out)
    bbox = info["stats"]["bbox"]["native"]["bbox"]
    return bbox["minx"], bbox["maxx"], bbox["miny"], bbox["maxy"]

def buffer_single_tile(args: Tuple[List, Tuple, Dict, int, 
int, Path, Path]) -> Tuple[str, bool, str]:
    offsets, (x,y), tiles, buffer_size, tile_size, buffered_tiles_dir, tile_path = args

    try: 
        neighbors = []

        for dx, dy in offsets:
            key = (x + dx, y + dy)
            if key in tiles:
                neighbors.append(tiles[key])

        # Buffered bounds
        xmin, xmax, ymin, ymax = get_native_bounds(tile_path)

        xmin_b = xmin - buffer_size
        xmax_b = xmax + buffer_size
        ymin_b = ymin - buffer_size
        ymax_b = ymax + buffer_size

        bounds = f"([{xmin_b},{xmax_b}],[{ymin_b},{ymax_b}])"

        out_file = buffered_tiles_dir / tile_path.name

        # Create a temporary directory for buffer files
        tmp_dir = Path(tempfile.mkdtemp(prefix="pdal_buffer_", dir=buffered_tiles_dir))

        buffer_files = []

        # Create buffer tiles from neighbors
        for n_path in tqdm(neighbors, disable=True):
            buffer_file = tmp_dir / f"{n_path.stem}_buffer.laz"

            buffer_pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(n_path)},
                    {"type": "filters.crop", "bounds": bounds},
                    {"type": "writers.las", "filename": str(buffer_file),
                     "compression": "laszip", "extra_dims": "all"}
                ]
            }

            p = pdal.Pipeline(json.dumps(buffer_pipeline))
            p.execute()
            buffer_files.append(buffer_file)

        # Merge core tile + all buffer files
        merge_pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(tile_path)},
                *({"type": "readers.las", "filename": str(f)} for f in buffer_files),
                {"type": "filters.merge"},
                {"type": "writers.las", "filename": str(out_file),
                 "compression": "laszip", "extra_dims": "all"}
            ]
        }

        p_merge = pdal.Pipeline(json.dumps(merge_pipeline))
        p_merge.execute()

        # Delete the temporary folder with buffer tiles
        shutil.rmtree(tmp_dir)

        return (tile_path.name, True, "Success")
        
    except Exception as e:
        return (tile_path.name, False, str(e))
    

def buffer_tiles(
    input_dir: Path,
    output_dir: Path,
    buffer_size: int = 10,
    num_workers: int = 4
) -> Path:   
    """
    Buffer tiles with given buffer size and find neighboring tiles based on filenames.
    
    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Base output directory
        buffer_size: Buffer overlap in meters
        num_workers: Number of parallel conversion workers
    
    Returns:
        Path to buffered tiles directory
    """    
    # Find all LAZ files
    input_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    
    if not input_files:
        print(f"  No LAZ/LAS files found in {input_dir}")


    filename_re = re.compile(r"(\d+)_(\d+)\.laz")
    tiles = {}
    xs, ys = set(), set()

    for f in input_files:
        m = filename_re.match(f.name)
        if not m:
            continue
        x, y = map(int, m.groups())
        tiles[(x, y)] = f
        xs.add(x)
        ys.add(y)

    xs = sorted(xs)
    ys = sorted(ys)

    dx = min(xs[i+1] - xs[i] for i in range(len(xs) - 1))
    dy = min(ys[i+1] - ys[i] for i in range(len(ys) - 1))

    tile_size = min(dx, dy)

    print(f"  Found {len(input_files)} files to buffer")
    print(f"  Detected tile size: {tile_size}")
    print(f"  Using {num_workers} parallel workers")
    print()

    buffered_tiles_dir = output_dir / f"buffer_{int(buffer_size)}m"
    buffered_tiles_dir.mkdir(parents=True, exist_ok=True)

    # Neighbor offsets (8-connected)
    offsets = [
        (-tile_size, -tile_size), (0, -tile_size), (tile_size, -tile_size),
        (-tile_size, 0),                              (tile_size, 0),
        (-tile_size, tile_size),  (0, tile_size),  (tile_size, tile_size),
    ]

    # prepare conversion tasks
    tasks = []
    for (x, y), tile_path in tiles.items():
        tasks.append((offsets, (x,y), tiles, buffer_size, tile_size, buffered_tiles_dir, tile_path))
    
    # Process files in parallel
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(buffer_single_tile, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            filename, success, message = future.result()
            if success:
                successful += 1
                print(f"  ✓ Buffered: {filename}")
            else:
                failed += 1
                print(f"  ✗ Failed: {filename} - {message}")
    
    print()
    print(f"  Buffering complete: {successful} successful, {failed} failed")

    return buffered_tiles_dir