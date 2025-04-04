import argparse
import copy

import clip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ArtistDataset
from data.image_transforms import build_transform


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_name", type=str, required=True)

    parser.add_argument("--artist_mapping_file", type=str,
                        default="test_artists.txt")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model_arch", type=str, default="ViT-L/14")
    return parser


def get_prototype(data_dir, csv_name, batch_size=64, workers=4, dtype=torch.float32, device="cuda",
                  artist_mapping_file="test_artists.txt", model_arch="ViT-L/14"):
    """
    Returns and saves a torch tensor of avg features for each artist in the dataset
    """
    ret_transform = build_transform('clip_base_noaug')

    # Get dataset
    split = csv_name.split('_imgs.csv')[0]
    dataset = ArtistDataset(split=split, image_prep="clip_base_noaug",
                            dataset_info_dir=data_dir, label_csv_col="prompt_label")

    # Load CLIP vision encoder
    clip_model, preprocess = clip.load(model_arch)
    clip_visual = copy.deepcopy(clip_model.visual)
    del clip_model
    del preprocess

    clip_visual.proj = None
    clip_visual.to(dtype=dtype, device=device)

    # Initialize prototype array
    prototype_arr = []
    dirname = data_dir.split('/')[-1]
    if model_arch == "ViT-L/14":
        vit_str = "clip_vit_large"
    elif model_arch == "ViT-B/16":
        vit_str = "clip_vit_base"

    prototype_path = f"{dirname}/prototype-{split}-{vit_str}-oneprocess.npy"
    print(f"Will save final prototype to {prototype_path}")

    # Get artist indices for the entire dataset
    all_dataset_artists = pd.read_csv(
        f"{data_dir}/{split}_imgs.csv")['prompt_label']
    all_dataset_artist_idxs = np.array(
        [dataset.class_mapping.index(artist) for artist in all_dataset_artists])

    # Compute prototype for each artist
    for artist_idx, artist in enumerate(dataset.class_mapping):
        artist_indices = np.where(all_dataset_artist_idxs == artist_idx)[0]
        print(
            f"Artist: {artist_idx} {artist} artist_indices: {artist_indices[:10]}")
        sub_dataset = torch.utils.data.Subset(dataset, artist_indices)
        data_loader = torch.utils.data.DataLoader(
            sub_dataset,
            sampler=None,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
        )
        print(f"Sub Dataset: {len(sub_dataset)} imgs")

        artist_features = []
        # Start inference
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), desc=f"Extracting for {artist_idx} {artist}", total=len(data_loader)):

                images = batch['img_pixel_values'].to(
                    dtype=dtype, device=device)
                features = clip_visual(images)
                if model_arch == "ViT-L/14":
                    assert features.shape[1] == 1024
                elif model_arch == "ViT-B/16":
                    assert features.shape[1] == 768

                artist_features.append(features)

        artist_features = torch.cat(artist_features, dim=0)

        prototype_arr.append(torch.mean(artist_features, dim=0))

    prototype_arr = torch.stack(prototype_arr, dim=0)

    np.save(prototype_path, prototype_arr.cpu().numpy())
    print(f"Prototype saved to {prototype_path}")
    print(f"Prototype shape: {prototype_arr.shape}")

    return prototype_arr


def main(args):
    get_prototype(data_dir=args.data_dir,
                  csv_name=args.csv_name,
                  artist_mapping_file=args.artist_mapping_file,
                  batch_size=args.batch_size,
                  workers=args.workers,
                  model_arch=args.model_arch,
                  dtype=torch.float32,
                  device="cuda",
                  )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
