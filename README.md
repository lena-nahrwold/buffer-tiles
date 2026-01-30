# Find neighboring tiles and add buffer around tile
## Create conda environment
```
conda env create -f environment.yml
conda activate pdal-pipeline
```
## Input data requirements
- LAZ version 1.4 with a uniform scaling of 0.001.
- Square tiles, with variable tile size, but always a multiple of 10 m.
- File names correspond to the coordinates of the lower-left corner, e.g., `xxxxxx_yyyyyy.laz`.
- Coordinate Reference System (CRS) is UTM with WGS84 ellipsoid by default; Z values represent ellipsoid heights.
- Each tile’s bounding geometry is a polygon without holes.
- The point source ID field is populated with scan position information for TLS data or flight strip information for ULS data.

## Run buffer task
```
python src/run.py --task buffer --buffer_size 10 --input_dir /path/to/input --output_dir /path/to/output
```

## Run (instance) merge task
```
python src/merge_tiles.py \
    --input-dir /path/to/segmented_buffered_tiles \
    --original-tiles-dir /path/to/original_tiles \
    --output-merged /path/to/merged.laz \
    --output-tiles-dir /path/to/output_tiles
```
For complete smart tiling and merging pipeline see: https://github.com/3dTrees-earth/3dtrees_Smart_Tile

## Run simple crop task 
Crop buffered tiles back to original bounds:
```
python src/run.py --task crop --input_dir /path/to/input --orig_tiles_dir /path/to/original/tiles --output_dir /path/to/output
```



## ToDo's