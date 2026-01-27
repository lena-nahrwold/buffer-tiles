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
    print(f"Output base directory: {output_dir}")
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
    input_dir = params.input_dir
    output_dir = params.output_dir
    orig_tiles_dir= params.orig_tiles_dir
    workers = params.workers

    print("=" * 60)
    print("Running Crop Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output base directory: {output_dir}")
    print(f"Original tiles directory: {orig_tiles_dir}")
    print(f"Workers: {workers}")
    print()

    crop_tiles(input_dir, output_dir, orig_tiles_dir, workers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="src/output")
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
        if not args.orig_tiles_dir:
            print("Path to riginal tiles needed for crop task. Please re-run the script and add --orig_tiles_dir.")
            sys.exit(1)
        args.orig_tiles_dir = Path(args.orig_tiles_dir)
        if not args.orig_tiles_dir.exists():
            print(f"Error: Original tiles directory does not exist: {args.orig_tiles_dir}")
            sys.exit(1)
        run_crop_task(args)
    else:
        print(f"Error: Unknown task: {args.task}")
        print("Valid tasks: tile, merge")
        sys.exit(1)
    
    print("Task Complete")

if __name__ == "__main__":
    main()