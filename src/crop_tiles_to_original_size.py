import json
import pdal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_tile(args: Tuple[Path, Path, Path, bool]):
    buffered_tile_path, orig_tiles_dir, cropped_tiles_dir = args
    
    try:
        original_tile = orig_tiles_dir / buffered_tile_path.name
        native_bounds, scale, offset = get_native_bounds(original_tile)

        minx, maxx, miny, maxy = native_bounds
        sx, sy, sz = scale
        ox, oy, oz = offset

        output_path = cropped_tiles_dir / buffered_tile_path.name

        bounds = f"([{minx},{maxx}], [{miny},{maxy}])"

        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(buffered_tile_path)},
                {
                    "type": "filters.crop",
                    "bounds": bounds
                },
                {
                    "type": "writers.las",
                    "filename": str(output_path),
                    "scale_x": sx,
                    "scale_y": sy,
                    "scale_z": sz,
                    "offset_x": ox,
                    "offset_y": oy,
                    "offset_z": oz,
                    "compression": "laszip",
                    "forward": "all",
                    "extra_dims": "all"
                }
            ]
        }
        p = pdal.Pipeline(json.dumps(pipeline))
        p.execute()

        return (output_path.name, True, "Success")
    
    except Exception as e:
        return (buffered_tile_path.name, False, str(e))
    

def get_native_bounds(laz_path):
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


    
def crop_tiles(
    input_dir: Path,
    output_dir: Path,
    orig_tiles_dir: Path,
    num_workers: int = 4
) -> Path: 
    """
    Crop buffered tiles to original bounds. 

    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Base output directory
        orig_tiles_dir: Directory containing the original tiles
        num_workers: Number of parallel conversion workers
    
    Returns:
        Path to cropped tiles directory
    """
    # Find all LAZ files
    input_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    if not input_files:
        print(f"  No LAZ/LAS files found in {input_dir}") 

    orig_files = list(orig_tiles_dir.glob("*.laz")) + list(orig_tiles_dir.glob("*.las"))
    if not orig_files:
        print(f"  No LAZ/LAS files found in {orig_tiles_dir}") 


    print(f"  Found {len(input_files)} files to crop")
    print(f"  Found {len(orig_files)} original files")
    print(f"  Using {num_workers} parallel workers")
    print()

    cropped_tiles_dir = output_dir / "processed_cropped_to_orig_bounds"
    cropped_tiles_dir.mkdir(parents=True, exist_ok=True) 

    # prepare conversion tasks
    tasks = []  
    for file_path in input_files:
        tasks.append((file_path, orig_tiles_dir, cropped_tiles_dir))

    # Process files in parallel
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_tile, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            filename, success, message = future.result()
            if success:
                successful += 1
                print(f"  ✓ Cropped: {filename}")
            else:
                failed += 1
                print(f"  ✗ Failed: {filename} - {message}")
    
    print()
    print(f"  Cropping complete: {successful} successful, {failed} failed")

    return cropped_tiles_dir
