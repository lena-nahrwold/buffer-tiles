import json
import pdal
import laspy
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

def crop_single_tile(args: Tuple[Path, Path, Path, bool]):
    buffered_tile_path, orig_tiles_dir, cropped_tiles_dir, deduplicate = args
    
    try:
        # Load buffered tile
        las = laspy.read(str(buffered_tile_path), laz_backend=laspy.LazBackend.LazrsParallel)

        original_tile = orig_tiles_dir / buffered_tile_path.name
        minx, maxx, miny, maxy = get_native_bounds(original_tile)

        output_path = cropped_tiles_dir / buffered_tile_path.name

        if deduplicate:
            n_points = len(las.points)

            # Extract XYZ
            points = np.vstack([las.x, las.y, las.z]).T

            # Deduplicate
            unique_points = deduplicate_points(points)
            removed_count = n_points - len(unique_points)
            print(f"[INFO] {buffered_tile_path.name}: Removed {removed_count:,} duplicates ({100*removed_count/n_points:.1f}%)")

            # Build small local KDTree
            tree = cKDTree(points)
            distances, indices = tree.query(unique_points, k=1, distance_upper_bound=0.001)
            valid_indices = indices[~np.isinf(distances)].astype(int)

            # Create output with ALL original attributes preserved
            out_las = laspy.create(file_version=las.header.version, point_format=las.header.point_format)
            out_las.header.offsets = las.header.offsets
            out_las.header.scales = las.header.scales

            # Copy ALL dimensions for surviving points
            out_las.points = las.points[valid_indices]

            # Crop to original bounds
            original_tile = orig_tiles_dir / buffered_tile_path.name
            minx, maxx, miny, maxy = get_native_bounds(original_tile)

            mask = (
                (out_las.x >= minx) & (out_las.x <= maxx) &
                (out_las.y >= miny) & (out_las.y <= maxy)
            )
            out_las.points = out_las.points[mask]

            # Write output
            out_las.write(output_path)
        else:
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
                        "compression": "laszip",
                        "extra_dims": "all"
                    }
                ]
            }
            p = pdal.Pipeline(json.dumps(pipeline))
            p.execute()

        return (buffered_tile_path.name, True, "Success")
    
    except Exception as e:
        return (buffered_tile_path.name, False, str(e))


def deduplicate_points(
    points: np.ndarray,
    tolerance: float = 0.001,
    grid_size: float = 50.0,
) -> Tuple[np.ndarray]:
    """
    Remove duplicate points from overlapping tiles using grid-based processing.
    When duplicates exist, keep the one with higher instance ID.

    Uses spatial grid cells to reduce memory usage - instead of sorting billions
    of points at once, we process smaller cells independently.

    Args:
        points: Nx3 array of point coordinates
        tolerance: Distance tolerance (default 1mm)
        grid_size: Size of spatial grid cells in meters (default 50m)

    Returns:
        Tuple of (unique_points)
    """
    n_points = len(points)
    scale = 1.0 / tolerance

    # Compute grid cell indices for each point
    min_coords = points.min(axis=0)
    grid_indices = ((points[:, :2] - min_coords[:2]) / grid_size).astype(np.int32)

    # Create cell keys (combine x,y grid indices into single key)
    max_grid_y = grid_indices[:, 1].max() + 1
    cell_keys = grid_indices[:, 0] * max_grid_y + grid_indices[:, 1]

    # Round coordinates to tolerance for duplicate detection
    rounded = np.floor(points * scale).astype(np.int64)

    # Create point hash: combine cell key, rounded coords, and negative instance for sorting
    # We want higher instance IDs to come first within same position
    point_hash = rounded[:, 0] + rounded[:, 1] * 73856093 + rounded[:, 2] * 19349669

    # Sort by: cell_key, point_hash, then -instance (so higher instance comes first)
    sort_order = np.lexsort((point_hash, cell_keys))

    sorted_cell_keys = cell_keys[sort_order]
    sorted_point_hash = point_hash[sort_order]

    # Find duplicates: same cell AND same point hash
    is_duplicate = np.zeros(n_points, dtype=bool)
    is_duplicate[1:] = (sorted_cell_keys[1:] == sorted_cell_keys[:-1]) & (
        sorted_point_hash[1:] == sorted_point_hash[:-1]
    )

    # Map back to original indices
    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[sort_order[is_duplicate]] = False

    # Extract kept points
    unique_points = points[keep_mask]

    return unique_points

def get_native_bounds(laz_path):
    out = subprocess.check_output(
        ["pdal", "info", str(laz_path)],
        text=True
    )
    info = json.loads(out)
    bbox = info["stats"]["bbox"]["native"]["bbox"]
    return bbox["minx"], bbox["maxx"], bbox["miny"], bbox["maxy"]

    
def crop_tiles(
    input_dir: Path,
    output_dir: Path,
    orig_tiles_dir: Path,
    deduplicate: bool,
    num_workers: int = 4
) -> Path: 
    """
    Crop buffered tiles to original bounds. 

    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Base output directory
        orig_tiles_dir: Directory containing the original tiles
        deduplicate: If set, duplicate points are removed
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
    print(f"  Remove duplicate points: {deduplicate}")
    print(f"  Using {num_workers} parallel workers")
    print()

    cropped_tiles_dir = output_dir / "processed_cropped_to_orig_bounds"
    cropped_tiles_dir.mkdir(parents=True, exist_ok=True) 

    # prepare conversion tasks
    tasks = []  
    for file_path in input_files:
        tasks.append((file_path, orig_tiles_dir, cropped_tiles_dir, deduplicate))

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
