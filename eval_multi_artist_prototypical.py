import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import MultiArtistDataset
from models import prototypical_network
from utils.eval_utils import update_metrics_csv, evaluate_multi_artist_preds
from utils.eval_args import parse_eval_args


def main(args):
    parent_dir = os.path.join(args.save_dir, args.model_id)
    metrics_csv_path = os.path.join(parent_dir,
                                    f"metrics_on_{os.path.basename(args.dataset_info_dir)}.csv")

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"Created directory {parent_dir}")

    # save command args with omegaconf
    conf = OmegaConf.create(vars(args))
    OmegaConf.save(conf, f"{parent_dir}/eval_args.yaml")

    key_name = args.label_csv_col

    with open(f"{args.dataset_info_dir}/{args.artist_mapping_file}", "r") as f:
        mapping = [l.strip() for l in f.readlines()]

    class_mapping = {key_name: mapping}

    print(f"class_mapping: {class_mapping}")

    if args.model_type == "vanilla_classifier":
        raise ValueError(
            "This script is not meant for the vanilla_classifier. Use eval_classifier_method.py instead with the --evaluate_multi_artist_preds flag.")

    elif args.model_type == "prototypical_network":
        model = prototypical_network.load_checkpoint(
            args.ckpt_path, args.prototype_path, model_arch=args.model_arch)

    print(f"image_prep: {args.image_prep}")

    split = args.eval_split
    val_data = MultiArtistDataset(split=split, image_prep=args.image_prep,
                                  dataset_info_dir=args.dataset_info_dir, label_csv_col=args.label_csv_col,
                                  artist_mapping_file=args.artist_mapping_file)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.dataloader_num_workers, pin_memory=True)

    os.makedirs(f"{parent_dir}/{split}", exist_ok=True)
    print(f"Created directory {parent_dir}/{split}")

    # cast model
    model = model.to(dtype=torch.float32, device="cuda")
    model.eval()

    y_true = {key_name: []}
    y_pred = {key_name: []}
    y_pred_probs = {key_name: []}

    if not os.path.exists(f"{parent_dir}/{split}/y_pred.npz"):
        # START INFERENCE
        print(f"Starting inference for {split}")

        for idx, batch in tqdm(enumerate(val_loader), desc=f"Validation {split}", total=len(val_loader)):
            images = batch["img_pixel_values"].cuda(non_blocking=True)

            gt_labels = batch["label"].float().cuda(non_blocking=True)

            with torch.no_grad():
                output = model(images)

            pred_probs = torch.softmax(output, dim=-1)
            pred_labels = (pred_probs > 0.5).float()

            y_true[key_name].extend(gt_labels.cpu().numpy())

            y_pred[key_name].extend(pred_labels.cpu().numpy())
            y_pred_probs[key_name].extend(pred_probs.cpu().numpy())

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

    metrics_dict = evaluate_multi_artist_preds(
        class_mapping, parent_dir, split, key_name, args.model_type, args.model_id, eval_dataset_name)
    update_metrics_csv(metrics_csv_path, metrics_dict)


if __name__ == "__main__":
    args = parse_eval_args()
    if args.eval_split == "test_all_unseen_query":
        args.artist_mapping_file = "test_artists.txt"
    elif args.eval_split == "test_artist":
        args.artist_mapping_file = "map_prompt_label.txt"

    main(args)
