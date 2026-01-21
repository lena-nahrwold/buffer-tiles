import re
import json
import pdal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

def buffer_single_file(args: Tuple[List, Tuple, Dict, int, 
int, Path, Path]) -> Tuple[str, bool, str]:
    offsets, (x,y), tiles, buffer_size, tile_size, buffered_tiles_dir, tile_path = args

    try: 
        neighbors = []

        for dx, dy in offsets:
            key = (x + dx, y + dy)
            if key in tiles:
                neighbors.append(tiles[key])

        # Buffered bounds
        xmin = x - buffer_size
        ymin = y - buffer_size
        xmax = x + tile_size + buffer_size
        ymax = y + tile_size + buffer_size

        bounds = f"([{xmin},{xmax}],[{ymin},{ymax}])"

        out_file = buffered_tiles_dir / tile_path.name

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
        futures = {executor.submit(buffer_single_file, task): task[0] for task in tasks}
        
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