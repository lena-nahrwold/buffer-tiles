import re
import json
import pdal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_native_bounds(laz_path):
    out = subprocess.check_output(
        ["pdal", "info", str(laz_path)],
        text=True
    )
    info = json.loads(out)
    bbox = info["stats"]["bbox"]["native"]["bbox"]
    return bbox["minx"], bbox["maxx"], bbox["miny"], bbox["maxy"]

def crop_single_tile(args: Tuple[Path, Path, Path]):
    file_path, orig_tiles_dir, cropped_tiles_dir = args

    try:
        original_tile = orig_tiles_dir / file_path.name

        minx, maxx, miny, maxy = get_native_bounds(original_tile)
        
        bounds = f"([{minx},{maxx}],[{miny},{maxy}])"
        
        out_file = cropped_tiles_dir / file_path.name
        pipeline = {
            "pipeline": [
                str(file_path),
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

        return (file_path.name, True, "Success")
    
    except Exception as e:
        return (file_path.name, False, str(e))
    

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
        futures = {executor.submit(crop_single_tile, task): task[0] for task in tasks}
        
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
