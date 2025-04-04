import os
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.datasets import MultiArtistDataset
from models.prototypical_network import ProtoClip
from train_single_artist_prototypical import parse_args

# suppress sklearn warnings for zero division during average precision calculation in validation loop
sys.stderr = open(os.devnull, "w")  # silence stderr
sys.stderr = sys.__stderr__  # unsilence stderr


def validation(model, val_loader, criterion, device="cuda"):
    val_log = {}

    split = val_loader.dataset.split
    val_loss_sum = 0

    y_true = []
    y_pred_probs = []

    for idx, batch in tqdm(
        enumerate(val_loader), desc=f"Validation {split}", total=len(val_loader)
    ):
        images = batch["img_pixel_values"].to(device)
        gt_labels = batch["label"].to(device)
        batch_size = gt_labels.shape[0]

        output = model.forward(images)

        loss = criterion(output, gt_labels)

        val_loss_sum += loss.item() * batch_size

        y_true.append(gt_labels.detach().cpu().numpy())
        y_pred_prob = torch.softmax(output, dim=1).detach().cpu().numpy()
        y_pred_probs.append(y_pred_prob)

    val_log[f"{split}/loss"] = val_loss_sum / len(val_loader.dataset)

    y_true = np.concatenate(y_true)  # shape: N, num_classes
    y_pred_probs = np.concatenate(y_pred_probs)
    y_indicators = (y_true > 0).astype(int)

    val_log[f"{split}/mAP"] = average_precision_score(
        y_indicators, y_pred_probs, average="macro"
    )

    return val_log


def main(args):
    OUTPUT_DIR = os.path.join("logs", args.run_name)
    os.makedirs(f"{OUTPUT_DIR}/ckpt", exist_ok=True)

    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # save args with omegaconf
    conf = OmegaConf.create(vars(args))
    OmegaConf.save(conf, f"{OUTPUT_DIR}/args.yaml")

    accelerator = Accelerator(log_with="wandb")

    if accelerator.is_main_process:
        # if ckpt dir is not empty, exit
        if len(os.listdir(f"{OUTPUT_DIR}/ckpt")) > 0:
            raise RuntimeError(
                f"Checkpoint dir {OUTPUT_DIR}/ckpt is not empty.")

        init_kwargs = {
            "wandb": {
                "dir": "wandb",
                "name": args.run_name,
            }
        }
        accelerator.init_trackers(
            args.tracker_project_name, config=dict(vars(args)), init_kwargs=init_kwargs
        )

    # Load dataset
    train_data = MultiArtistDataset(
        split="train",
        image_prep=args.image_prep,
        dataset_info_dir=args.dataset_info_dir,
        artist_mapping_file="map_prompt_label.txt",
        label_csv_col="all_prompt_labels",
        use_weighted_sampler=args.use_weighted_sampler,
    )

    # use weighted random sampler
    if args.use_weighted_sampler:
        weights = train_data.sample_weights
        assert len(weights) == len(train_data)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True)
        print(f"Using weighted random sampler: {np.unique(weights)}")
        train_loader = DataLoader(
            train_data,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

    # Create validation dataloaders
    if args.test_dataset_info_dir is None:
        args.test_dataset_info_dir = args.dataset_info_dir

    val_data = MultiArtistDataset(
        split=args.test_dataset_split,
        image_prep=args.image_prep,
        dataset_info_dir=args.test_dataset_info_dir,
        artist_mapping_file="map_prompt_label.txt",
        label_csv_col="all_prompt_labels",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Load model
    prototype = torch.from_numpy(np.load(args.prototype_path))
    print(f"Prototype shape: {prototype.shape}")

    model = ProtoClip(
        prototype=prototype, model_arch=args.model_arch, temperature=args.temperature
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    ### TRAINING ###

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader)

    # cast loss functions
    criterion = criterion.to(device=accelerator.device)

    if args.resume_from_ckpt:
        accelerator.load_state(args.resume_from_ckpt)
        print(f"Loaded model from {args.resume_from_ckpt}")

    for e in tqdm(range(args.num_training_epochs), desc="Training epochs"):
        model.train()
        running_loss = 0

        for idx, batch in tqdm(
            enumerate(train_loader), desc="batches", total=len(train_loader)
        ):
            images = batch["img_pixel_values"]  # B, C, H, W
            gt_labels = batch["label"].float()  # B, num_classes

            optimizer.zero_grad()

            output = model.forward(images)

            loss = criterion(output, gt_labels)

            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

            pred_probs = torch.softmax(output, dim=1)

            # calculate AP
            pred_probs, gt_labels = accelerator.gather_for_metrics(
                (pred_probs, gt_labels)
            )
            gt_indicators = (gt_labels > 0).int().detach().cpu().numpy()

            batch_mAP = average_precision_score(
                gt_indicators, pred_probs.detach().cpu().numpy(), average="macro"
            )

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    log_dict = {"train/loss": loss, "train/mAP": batch_mAP}
                    for tracker in accelerator.trackers:
                        tracker.log(log_dict)

        if accelerator.sync_gradients:
            if accelerator.is_main_process:
                # save model
                if e % args.checkpoint_every_n_epochs == 0:
                    accelerator.save_state(
                        output_dir=f"{OUTPUT_DIR}/ckpt/epoch{e}")
                    print(f"Saved model epoch {e}")

                if e % args.log_every_n_epochs == 0:
                    eval_model = accelerator.unwrap_model(model)
                    # Evaluate performance of each epoch
                    eval_model.eval()
                    eval_model = eval_model.to(device=accelerator.device)

                    val_log = {}

                    with torch.no_grad():
                        val_log = validation(
                            eval_model,
                            val_loader,
                            criterion=criterion,
                            device=accelerator.device,
                        )

                    # Log to wandb
                    log_dict = {**val_log}

                    for key in log_dict.keys():
                        print(f"{key}: {log_dict[key]:.3f}")

                    for tracker in accelerator.trackers:
                        tracker.log(log_dict)

    accelerator.end_training()
    return model


if __name__ == "__main__":
    args = parse_args()
    main(args)
