import argparse
import multiprocessing as mp
import os
import subprocess
from glob import glob

from huggingface_hub import snapshot_download
from tqdm import tqdm


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract tar files from Hugging Face")
    parser.add_argument(
        "--download_config",
        type=str,
        required=True,
        help="Specify what datasets to download",
        choices=["sample", "single_artist", "multi_artist", "laion", "all"],
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for multiprocessing (default: 8)",
    )

    parser.add_argument(
        "--keep_tar_gz",
        action="store_true",
        help="Keep the downloaded tar.gz files after extraction (default: delete them)",
        default=False,
    )

    parser.add_argument(
        "--token",
        type=str,
        required=False,
        help="Hugging Face token for authentication if huggingface-cli login didn't store credentials",
        default=None,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def extraction_subprocess(tar_files, target_dir, keep_tar_gz, process_id):
    """Subprocess to extract tar files."""
    for file in tqdm(tar_files, desc=f"Process {process_id}"):
        try:
            print(f"[Process {process_id}] Extracting {file}...")
            subprocess.run(["tar", "-xzf", os.path.join(target_dir, file),
                            "-C", target_dir], check=True
                           )
            if not keep_tar_gz:
                os.remove(os.path.join(target_dir, file))
        except subprocess.CalledProcessError as e:
            print(f"[Process {process_id}] Error extracting {file}: {e}")


def get_download_patterns(download_config):
    """Get the download patterns based on the download configuration."""
    ignore_patterns = ["trained_models/*", "metadata/*", "readme_assets/*"]
    if download_config == "sample":
        allow_patterns = [
            "sample_datasets/*",
        ]

    elif download_config == "single_artist":
        allow_patterns = [
            "dataset_all_single_artist/*",
            # SDXL
            "dataset_sdxl_complex/*",
            "sdxl_images_mj_prompts-artist/*",
            "dataset_sdxl_simple/*",
            "sdxl_images_easy_prompts-artist/*",
            # SD1.5
            "dataset_sd15_complex/*",
            "sd15_images_mj_prompts-artist/*",
            "dataset_sd15_simple/*",
            "sd15_images_easy_prompts-artist/*",
            # PixArt-Sigma
            "dataset_pixart_complex/*",
            "pixart_sigma_images_mj_prompts-artist/*",
            "dataset_pixart_simple/*",
            "pixart_sigma_images_easy_prompts-artist/*",
            # midjourney
            "dataset_journeydb/*",
            "JourneyDB.tar.gz",
        ]

        ignore_patterns.append("sample_datasets/*")

    elif download_config == "multi_artist":
        allow_patterns = [
            "dataset_all_multi_artist/*",
            # multi-artist SDXL
            "dataset_2artist_sdxl_complex/*",
            "dataset_3artist_sdxl_complex/*",
            "sdxl_images_mj_prompts-multi-artist/*",
            "dataset_2artist_sdxl_simple/*",
            "dataset_3artist_sdxl_simple/*",
            "sdxl_images_easy_prompts-multi-artist/*",
        ]
        ignore_patterns.append("sample_datasets/*")

    elif download_config == "laion":
        allow_patterns = [
            "dataset_laion/*",
            "dataset_laion_for_journeydb/*",
            "laion_images_*.tar.gz",
        ]
        ignore_patterns.append("sample_datasets/*")

    elif download_config == "all":
        allow_patterns = None
        ignore_patterns.append("sample_datasets/*")
    else:
        raise ValueError(f"Unknown download_config: {download_config}")

    return ignore_patterns, allow_patterns


def main(args):
    # process args
    ignore_patterns, allow_patterns = get_download_patterns(
        args.download_config)

    target_dir = "./"

    # snapshot download according to args
    snapshot_download(
        repo_id=f"cmu-gil/PromptedArtistIdentificationDataset",
        repo_type="dataset",
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        local_dir=target_dir,
        token=args.token,
        max_workers=args.num_workers,
    )
    print(
        f"[INFO] Finished downloading dataset with config {args.download_config}")

    # if sample dataset, move everything out of sample_datasets/
    if args.download_config == "sample":
        print("[INFO] Moving sample_dataset/ files to target directory...")
        sample_dir = os.path.join(target_dir, "sample_datasets")
        if os.path.exists(sample_dir):
            for item in os.listdir(sample_dir):
                src_path = os.path.join(sample_dir, item)
                dst_path = os.path.join(target_dir, item)
                os.replace(src_path, dst_path)
            os.rmdir(sample_dir)

    # multiprocessing: extract tar files in parallel
    l_tar_files = glob(f"{target_dir}/**/*.tar.gz", recursive=True)
    print(f"[INFO] Found {len(l_tar_files)} tar.gz files to extract.")

    num_processes = min(args.num_workers, len(l_tar_files))
    keep_tar_gz = args.keep_tar_gz

    print(
        f"[INFO] Starting tar.gz extraction with {num_processes} processes. keep_tar_gz={keep_tar_gz}")

    split = len(l_tar_files) // num_processes
    remainder = len(l_tar_files) % num_processes
    processes = []
    for i in range(num_processes):
        start = i * split
        end = (i + 1) * split
        if i == num_processes - 1:
            end += remainder
        p = mp.Process(
            target=extraction_subprocess,
            args=(l_tar_files[start:end], target_dir, keep_tar_gz, i)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish and check exit codes
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Upload subprocess with PID {p.pid} exited with code {p.exitcode}.")

    print(f"[INFO] Finished extracting all tar.gz files under {target_dir}.")


if __name__ == "__main__":

    args = parse_args()
    main(args)
