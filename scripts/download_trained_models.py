import argparse

from download_prototypes import download_prototypes
from huggingface_hub import snapshot_download


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download trained model checkpoints from Hugging Face")
    parser.add_argument(
        "--download_config",
        type=str,
        required=True,
        help="Specify what models to download",
        choices=["all", "single_artist_prototypical",
                 "multi_artist_prototypical", "single_artist_vanilla"],
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for multiprocessing (default: 8)",
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


def download_models(args):
    """Download models from Hugging Face based on the specified configuration."""

    target_dir = "./"

    if args.download_config == "all":
        allow_patterns = ["trained_models/*"]
        download_prototypes(target_dir, args.token)
    elif args.download_config == "single_artist_prototypical":
        allow_patterns = [
            "trained_models/protoclip_trained_on_dataset_all_single_artist/*"]
        download_prototypes(target_dir, args.token)
    elif args.download_config == "multi_artist_prototypical":
        allow_patterns = [
            "trained_models/multi_artist_protoclip_trained_on_dataset_all_multi_artist/*"]
        download_prototypes(target_dir, args.token)
    elif args.download_config == "single_artist_vanilla":
        allow_patterns = [
            "trained_models/clipclassifier_trained_on_dataset_all_single_artist/*"]
    else:
        raise ValueError("Invalid download configuration specified.")

    snapshot_download(
        repo_id="cmu-gil/PromptedArtistIdentificationDataset",
        repo_type="dataset",
        token=args.token,
        local_dir=target_dir,
        allow_patterns=allow_patterns,
        max_workers=args.num_workers,
    )

    print(f"[INFO] Finished downloading models")


if __name__ == "__main__":
    args = parse_args()
    download_models(args)
