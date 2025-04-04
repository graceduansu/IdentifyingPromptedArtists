import os

import pandas as pd
import torch
from PIL import Image

from .image_transforms import build_transform


class ArtistDataset(torch.utils.data.Dataset):
    """
    Dataset for single-artist classification.
    Each image is associated with a single artist label.
    """

    def __init__(
        self,
        split,
        image_prep,
        dataset_info_dir,
        label_csv_col="prompt_label",
        artist_mapping_file="map_prompt_label.txt",
        use_weighted_sampler=False,
    ):
        super().__init__()
        self.split = split
        csv_path = f"{dataset_info_dir}/{split}_imgs.csv"

        df = pd.read_csv(csv_path)

        self.captions = {}
        for i, row in df.iterrows():
            self.captions[row["img_path"]] = {
                "label_name": row[label_csv_col]}

        self.img_names = list(self.captions.keys())

        self.T = build_transform(image_prep)

        with open(f"{dataset_info_dir}/{artist_mapping_file}", "r") as f:
            self.class_mapping = [l.strip() for l in f.readlines()]

        # fix class_mapping
        if len(self.class_mapping) != len(df[label_csv_col].unique()):
            if split.startswith('test_all_unseen'):
                new_mapping_file = "test_artists.txt" 
            else:
                new_mapping_file = "map_prompt_label.txt"
            print(f"[INFO] Changing artist_mapping_file to {new_mapping_file}")

            with open(f"{dataset_info_dir}/{new_mapping_file}", "r") as f:
                self.class_mapping = [l.strip() for l in f.readlines()]

        print("----------------------------------------")
        print(f"split: {split}")
        print(f"num images: {len(self.img_names)}")
        print(f"num classes: {len(self.class_mapping)}")
        print(f"classes: {self.class_mapping[:10]}")

        # Get sample weights to train with weighted sampler
        if split == "train" and use_weighted_sampler:
            self.sample_weights = self.get_sample_weights(df, csv_path)
            print(f"weights: {self.sample_weights[:10]}")

    def get_sample_weights(self, df, csv_path):
        """
        Returns:
        list: Sample weights for the dataset.
        """
        if "weight" not in df.columns:
            print(
                "[INFO] No sample weights found in train_imgs.csv. Creating and saving 'weight' column.")
            # make is_easy
            df['is_easy'] = df['img_path'].apply(
                lambda x: '_easy_prompts-' in x)
            temp = df.groupby(['source_label', 'is_easy']).size().reset_index()
            temp['weight'] = len(df) / temp[0]
            temp.drop(columns=[0], inplace=True)
            # assign weight to corresponding rows in df
            df = df.merge(temp, on=['source_label', 'is_easy'], how='left')
            df.to_csv(csv_path, index=False)

        return df["weight"].tolist()

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(img_name)
        label_name = self.captions[img_name]["label_name"]
        label_idx = self.class_mapping.index(label_name)

        img_t = self.T(input_img)

        batch_dict = {
            "img_pixel_values": img_t,
            "label_name": label_name,
            "label_idx": label_idx,
            "img_path": img_name,
            "dataset_idx": idx,
        }

        return batch_dict


class MultiArtistDataset(torch.utils.data.Dataset):
    """
    For multi-label classification training with categorical cross-entropy loss:
    The ground truth is 1/N for the N artists in the prompt, and 0 everywhere else.
    """

    def __init__(self, split, image_prep, dataset_info_dir,
                 artist_mapping_file="map_prompt_label.txt",
                 label_csv_col="all_prompt_labels",
                 use_weighted_sampler=False):

        super().__init__()
        self.split = split

        csv_path = f"{dataset_info_dir}/{split}_imgs.csv"

        df = pd.read_csv(csv_path)

        self.captions = {}
        for i, row in df.iterrows():
            self.captions[row["img_path"]] = {
                "label_names": row[label_csv_col].split(";"),
            }

        self.img_names = list(self.captions.keys())

        self.T = build_transform(image_prep)

        # get artist class mapping
        with open(f"{dataset_info_dir}/{artist_mapping_file}", "r") as f:
            self.artists = [l.strip() for l in f.readlines()]

        # encode the labels
        self.labels = []

        for img_name in self.img_names:
            label_names = self.captions[img_name]["label_names"]
            label_vec = torch.zeros(len(self.artists))
            label_vec = label_vec.scatter(0, torch.tensor([self.artists.index(
                l) for l in label_names]), 1 / len(label_names))  # shape: num_classes,
            self.labels.append(label_vec)

        print("----------------------------------------")
        print(f"split: {split}")
        print(f"num images: {len(self.img_names)}")
        print(f"num labels: {len(self.labels)}")
        print(f"num artists: {len(self.artists)}")

        # Get sample weights to train with weighted sampler
        if split == "train" and use_weighted_sampler:
            self.sample_weights = self.get_sample_weights(
                df, label_csv_col, csv_path)
            print(f"weights: {self.sample_weights[:10]}")

    def get_sample_weights(self, df, label_csv_col, csv_path):
        """
        Returns:
        list: Sample weights for the dataset.
        """
        if "weight" not in df.columns:
            print(
                "[INFO] No sample weights found in train_imgs.csv. Creating and saving 'weight' column.")
            df['num_artists'] = df[label_csv_col].apply(
                lambda x: len(x.split(';')))
            df['is_easy'] = df['img_path'].apply(
                lambda x: '_easy_prompts-' in x)
            temp = df.groupby(['num_artists', 'is_easy']).size().reset_index()
            temp['weight'] = len(df) / temp[0]
            temp.drop(columns=[0], inplace=True)
            # assign weight to corresponding rows in df
            df = df.merge(temp, on=['num_artists', 'is_easy'], how='left')
            df.to_csv(csv_path, index=False)

        return df["weight"].tolist()

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(img_name)
        label = self.labels[idx]

        # input images
        img_t = self.T(input_img)

        batch_dict = {
            "img_pixel_values": img_t,
            "label": label,
            "img_path": img_name,
            "dataset_idx": idx,
        }

        return batch_dict
