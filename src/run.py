import sys
import argparse
from pathlib import Path
from buffer_tiles import buffer_tiles
from crop_tiles_to_original_size import crop_tiles

def run_buffer_task(params):
    input_dir = params.input_dir
    output_dir = params.output_dir
    buffer_size = params.buffer_size
    workers = params.workers

    print("=" * 60)
    print("Running Buffer Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile buffer: {buffer_size}m")
    print(f"Workers: {workers}")
    print()

    buffered_tiles_dir = buffer_tiles(input_dir, output_dir, buffer_size, workers)

    print()
    print("=" * 60)
    print("Buffer Task Complete")
    print("=" * 60)
    print(f"  Buffered tiles directory: {buffered_tiles_dir}")

def run_crop_task(params):
    print("=" * 60)
    print("Running Crop Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {params.input_dir}")
    print(f"Output directory: {params.output_dir}")
    print(f"Original tiles directory: {params.orig_tiles_dir}")
    print(f"Workers: {params.workers}")
    print()

    crop_tiles(params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tile_size", default=None, type=int)
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--input_dir", type=str, default="./input")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--orig_tiles_dir", type=str)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)

    # Route to appropriate task function
    if args.task == "buffer":
        run_buffer_task(args)
    elif args.task == "crop":
        if not args.tile_size:
            print("Original tile size needed for crop task. Please re-run the script and add --tile_size.")
            sys.exit(1)
        run_crop_task(args)
    else:
        print(f"Error: Unknown task: {args.task}")
        print("Valid tasks: tile, merge")
        sys.exit(1)
    
    print("Task Complete")

if __name__ == "__main__":
    main()