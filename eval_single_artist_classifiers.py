import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ArtistDataset
from models import prototypical_network, vanilla_classifier
from utils.eval_utils import update_metrics_csv, evaluate_multi_artist_preds
from utils.eval_args import parse_eval_args


def evaluate_preds(parent_dir, split, key_name, model_type, model_id, eval_dataset):
    # open npz
    y_pred = {}
    y_true = {}

    with np.load(f"{parent_dir}/{split}/y_pred.npz") as f:
        y_pred[key_name] = f["key_name"]

    with np.load(f"{parent_dir}/{split}/y_true.npz") as f:
        y_true[key_name] = f["key_name"]

    acc = np.sum(y_true[key_name] == y_pred[key_name]) / len(y_true[key_name])

    # print accuracies
    print(f"Accuracy for {split} {key_name}: {acc:.4f}")

    # save to metrics_csv_path
    if "test_artist" in split:
        metrics_col = "Seen artists: Classifier"
    elif "test_all_unseen" in split:
        metrics_col = "Held-out artists: Classifier"

    metrics_dict = {
        "model_type": [model_type],
        "model_id": [model_id],
        metrics_col: [acc],
        "eval_dataset": [eval_dataset]
    }

    return metrics_dict


def main(args):
    parent_dir = os.path.join(args.save_dir, args.model_id)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"Created directory {parent_dir}")

    metrics_csv_path = os.path.join(args.save_dir,
                                    f"metrics_on_{os.path.basename(args.dataset_info_dir)}.csv")

    # save command args with omegaconf
    conf = OmegaConf.create(vars(args))
    OmegaConf.save(conf, f"{parent_dir}/eval_args.yaml")

    key_name = args.label_csv_col

    with open(f"{args.dataset_info_dir}/{args.artist_mapping_file}", "r") as f:
        mapping = [l.strip() for l in f.readlines()]

    class_mapping = {key_name: mapping}

    print(f"class_mapping: {class_mapping}")

    if args.model_type == "vanilla_classifier":
        model = vanilla_classifier.load_checkpoint(args.ckpt_path, num_classes=len(
            class_mapping[key_name]), model_arch=args.model_arch)

    elif args.model_type == "prototypical_network":
        model = prototypical_network.load_checkpoint(
            args.ckpt_path, args.prototype_path, model_arch=args.model_arch)

    val_data = ArtistDataset(split=args.eval_split, image_prep=args.image_prep,
                             dataset_info_dir=args.dataset_info_dir, label_csv_col=args.label_csv_col,
                             artist_mapping_file=args.artist_mapping_file)

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.dataloader_num_workers, pin_memory=True)

    os.makedirs(f"{parent_dir}/{args.eval_split}", exist_ok=True)
    print(f"Created directory {parent_dir}/{args.eval_split}")

    # cast model
    model = model.to(dtype=torch.float32, device="cuda")
    model.eval()
    split = args.eval_split

    y_true = {key_name: []}
    y_pred = {key_name: []}
    y_pred_probs = {key_name: []}

    if not os.path.exists(f"{parent_dir}/{split}/y_pred.npz"):
        # START INFERENCE
        print(f"Starting inference for {split}")
        out_csv_path = f"{parent_dir}/{split}/preds.csv"

        # Create lists to store data for pandas DataFrame
        img_paths = []
        true_artists = []
        pred_labels = []
        pred_probs_list = []

        for idx, batch in tqdm(enumerate(val_loader), desc=f"Validation {split}", total=len(val_loader)):
            images = batch["img_pixel_values"].cuda(non_blocking=True)
            gt_artists = batch["label_idx"].cuda(non_blocking=True)

            with torch.no_grad():
                output = model(images)

            pred_probs = torch.softmax(output, dim=1)
            pred_labels_batch = torch.argmax(pred_probs, dim=1)

            y_true[key_name].extend(gt_artists.cpu().numpy())

            y_pred[key_name].extend(pred_labels_batch.cpu().numpy())
            y_pred_probs[key_name].extend(pred_probs.cpu().numpy())

            # Collect data for pandas DataFrame
            for i in range(images.shape[0]):
                img_paths.append(batch["img_path"][i])
                true_artists.append(
                    class_mapping["prompt_label"][gt_artists[i].item()])
                pred_labels.append(
                    class_mapping[key_name][pred_labels_batch[i]])
                pred_probs_list.append(
                    pred_probs[i][pred_labels_batch[i]].item())

        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'original_img_path': img_paths,
            'true_artist': true_artists,
            f'pred_{key_name}': pred_labels,
            f'pred_{key_name}_prob': pred_probs_list
        })
        df.to_csv(out_csv_path, index=False)

        # save y_true, y_pred, pred_probs as npz
        np.savez(f"{parent_dir}/{split}/y_true.npz",
                 key_name=y_true[key_name])
        np.savez(f"{parent_dir}/{split}/y_pred.npz",
                 key_name=y_pred[key_name])
        np.savez(f"{parent_dir}/{split}/y_pred_probs.npz",
                 key_name=y_pred_probs[key_name])
        # END INFERENCE

    else:
        print(
            f"[INFO] Found existing y_pred.npz, skipping inference for {split}")

    print('------------------------------------\n\n')
    eval_dataset_name = args.dataset_info_dir.split("/")[-1]
    if args.evaluate_multi_artist_preds:
        print(f"Evaluating multi-artist predictions for {split}")
        metrics_dict = evaluate_multi_artist_preds(
            class_mapping, parent_dir, split, key_name, args.model_type, args.model_id, eval_dataset_name)
    else:
        print(f"Evaluating single-artist predictions for {split}")
        metrics_dict = evaluate_preds(
            parent_dir, split, key_name, args.model_type, args.model_id, eval_dataset_name)

    update_metrics_csv(metrics_csv_path, metrics_dict)


if __name__ == "__main__":
    args = parse_eval_args()
    if args.eval_split == "test_all_unseen_query":
        args.artist_mapping_file = "test_artists.txt"
    elif args.eval_split == "test_artist":
        args.artist_mapping_file = "map_prompt_label.txt"

    main(args)
