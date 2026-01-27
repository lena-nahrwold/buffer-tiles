# Find neighboring tiles and add buffer around tile
## Create conda environment
```
conda env create -f environment.yml
conda activate pdal-pipeline
```

## Run buffer task
`python src/run.py --task buffer --buffer_size 10 --input_dir /path/to/input --output_dir /path/to/output`

## Run crop task 
Crop buffered tiles back to original bounds:
`python src/run.py --task crop --input_dir /path/to/input --orig_tiles_dir /path/to/original/tiles --output_dir /path/to/output`


## ToDo's