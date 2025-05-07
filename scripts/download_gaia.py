# download_gaia.py : used to download gaia benchmarks

import argparse
import sys
from pathlib import Path

try:
    import datasets
    from datasets import load_dataset, get_dataset_config_names
except ImportError:
    print("Error: The 'datasets' library is required to download the GAIA dataset.")
    print("Please install it using: pip install datasets")
    sys.exit(1)

def download_gaia_dataset(
    level: int,
    split: str,
    output_dir: str,
    cache_dir: str | None = None,
    token: str | bool | None = None,
):
    """
    Downloads the specified level and split of the GAIA dataset directly
    and saves it locally.

    Args:
        level (int): The GAIA level to download (1, 2, or 3).
        split (str): The dataset split to download ('validation' or 'test').
        output_dir (str): The directory to save the downloaded dataset files.
        cache_dir (str, optional): Directory to cache downloaded data. Defaults to HF default.
        token (str | bool | None, optional): Hugging Face Hub token for authentication if needed.
    """
    print(f"Starting download for GAIA Level {level} - {split} split...")

    # Construct the level-specific configuration name
    config_name = f"2023_level{level}"
    dataset_name = "gaia-benchmark/GAIA"

    try:
        # Load the specific level and split directly
        print(f"Loading dataset '{dataset_name}' with configuration '{config_name}' and split '{split}'...")
        # Use the 'split' argument to specify 'validation' or 'test'
        gaia_dataset_split = load_dataset(
            dataset_name,
            config_name,
            split=split, # Specify the desired split here
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True
        )
        print(f"Dataset split '{split}' for Level {level} loaded successfully.")
        print(f"Number of examples: {len(gaia_dataset_split)}")

        # Create the output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory '{output_path}' ensured.")

        # Determine the filename based on level and split
        output_filename = f"gaia_level{level}_{split}.jsonl"
        full_output_path = output_path / output_filename

        # Save the loaded dataset split as JSONL
        print(f"Saving dataset split '{split}' to '{full_output_path}' as JSONL...")
        gaia_dataset_split.to_json(full_output_path, orient="records", lines=True)

        print(f"Successfully saved GAIA Level {level} ({split} split) to '{full_output_path}'.")

        # Check for file_name column and inform user about associated files
        if "file_name" in gaia_dataset_split.column_names:
            print("\nNote: This dataset contains references to associated files (e.g., images, PDFs).")
            print("These files are typically downloaded and managed by the 'datasets' library cache.")
            print(f"When using the dataset later, ensure the cache ({gaia_dataset_split.cache_files}) is accessible or re-download.")
        else:
             print("\nNote: No 'file_name' column detected for associated files in this split.")

    except ValueError as e:
        if "not found in dataset configuration names" in str(e) or "Unknown split" in str(e):
             available_configs = get_dataset_config_names(dataset_name)
             print(f"Error: Configuration '{config_name}' or split '{split}' not found for dataset '{dataset_name}'.")
             print(f"Please check the configuration name and split.")
             print(f"Available configurations: {available_configs}")

        else:
             print(f"An unexpected ValueError occurred: {e}")
             import traceback
             traceback.print_exc()
    except Exception as e:
        print(f"An error occurred during dataset loading or saving: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a specific level and split of the GAIA benchmark dataset.")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="The GAIA level to download (default: 1).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="The dataset split to download (default: 'validation').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/gaia",
        help="The directory where the dataset will be saved (default: './data/gaia').",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for the Hugging Face datasets cache.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face Hub token for private dataset access (if needed). Can also use HF_TOKEN env var or hf login.",
    )

    args = parser.parse_args()

    download_gaia_dataset(
        level=args.level,
        split=args.split,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        token=args.token or True # Pass True to try using logged-in token if arg not provided
    )