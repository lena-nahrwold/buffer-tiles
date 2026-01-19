import sys
import argparse
from buffer_tiles import buffer_tiles
from crop_tiles_to_original_size import crop_tiles

def run_buffer_task(params):
    buffer_tiles(params)

def run_crop_task(params):
    crop_tiles(params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tile_size", default=None, type=int)
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--input_dir", type=str, default="./in")
    parser.add_argument("--output_dir", type=str, default="./out")

    args = parser.parse_args()

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

if __name__ == "__main__":
    main()