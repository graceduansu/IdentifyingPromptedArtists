import argparse

from huggingface_hub import snapshot_download


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download prototype embeddings from Hugging Face")
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


def download_prototypes(target_dir, token):
    """Download prototype embeddings from Hugging Face."""
    snapshot_download(
        repo_id="cmu-gil/PromptedArtistIdentificationDataset",
        repo_type="dataset",
        allow_patterns=["dataset_laion/*.npy",
                        "dataset_laion_for_journeydb/*.npy"],
        local_dir=target_dir,
        token=token,
    )
    print(f"[INFO] Finished downloading prototypes")


if __name__ == "__main__":
    args = parse_args()
    download_prototypes("./", args.token)
