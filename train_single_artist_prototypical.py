import argparse
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.datasets import ArtistDataset
from models.prototypical_network import ProtoClip


def parse_args(input_args=None):
    """
    Parses command-line arguments for prototypical network training.
    This function sets up an argument parser to handle prototypical network specific options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    # dataset options
    parser.add_argument(
        "--image_prep", default="clip_base_randomresizedcrop_hflip_blurplusjpeg0.1", type=str)
    parser.add_argument("--dataset_info_dir", default="./", type=str)
    parser.add_argument("--label_csv_col", default="prompt_label", type=str)
    parser.add_argument("--test_dataset_split",
                        default="test_artist", type=str)
    parser.add_argument("--train_dataset_split", default="train", type=str)
    parser.add_argument("--test_dataset_info_dir", default=None, type=str)
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="Use a weighted sampler for the dataset.")


    # prototype options
    parser.add_argument("--prototype_path", default="N/A",
                        type=str, help="Path to the prototype file (.npy)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature parameter for prototypical network.")

    # model options
    parser.add_argument("--model_arch", type=str, default="ViT-L/14",
                        help="The CLIP model architecture to use.")

    # training details
    parser.add_argument("--resume_from_ckpt", type=str, default=None,
                        help="The path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_resume_id", type=str, default=None,
                        help="The wandb run id to resume training from.")
    parser.add_argument("--seed", type=int, default=0,
                        help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--dataloader_num_workers", type=int, default=12,
                        help="Number of dataloader workers per GPU.")

    # logging args
    parser.add_argument("--log_every_n_epochs", type=int,
                        default=1, help="Log training metrics every n epochs.")
    parser.add_argument("--checkpoint_every_n_epochs", type=int,
                        default=1, help="Save a checkpoint every n epochs.")
    parser.add_argument("--tracker_project_name", type=str,
                        default="prototypical_network", help="The name of the wandb project to log to.")
    parser.add_argument("--run_name", type=str,
                        default="prototypical_network", help="The name of the wandb run.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def validation(model, val_loader, criterion, device="cuda"):
    """Validation function for prototypical network"""
    val_log = {}

    split = val_loader.dataset.split
    val_loss_sum = 0
    val_corrects = 0

    for batch in tqdm(val_loader, desc=f"Validation {split}", total=len(val_loader)):
        images = batch["img_pixel_values"].to(device)
        gt_labels = batch["label_idx"].to(device)
        batch_size = gt_labels.shape[0]

        output = model.forward(images)
        loss = criterion(output, gt_labels)

        val_loss_sum += loss.item() * batch_size
        pred_labels = torch.argmax(output, dim=1)
        val_corrects += torch.sum(pred_labels == gt_labels)

    val_log[f"{split}/loss"] = val_loss_sum / len(val_loader.dataset)
    val_log[f"{split}/accuracy"] = val_corrects / len(val_loader.dataset)

    return val_log


def main(args):
    OUTPUT_DIR = os.path.join("logs", args.run_name)
    os.makedirs(f"{OUTPUT_DIR}/ckpt", exist_ok=True)

    set_seed(args.seed)

    # Save args with omegaconf
    conf = OmegaConf.create(vars(args))
    OmegaConf.save(conf, f"{OUTPUT_DIR}/args.yaml")

    # Setup accelerator
    accelerator = Accelerator(log_with="wandb")

    # Initialize wandb
    if accelerator.is_main_process:
        if args.resume_from_ckpt is not None:
            init_kwargs = {
                "wandb": {
                    "dir": "./wandb",
                    "resume": "must",
                    "id": args.wandb_resume_id
                }
            }
        else:
            init_kwargs = {
                "wandb": {
                    "dir": "./wandb",
                    "name": args.run_name,
                }
            }
        accelerator.init_trackers(args.tracker_project_name, config=dict(
            vars(args)), init_kwargs=init_kwargs)

    # Create training dataset
    train_data = ArtistDataset(
        split=args.train_dataset_split,
        image_prep=args.image_prep,
        label_csv_col=args.label_csv_col,
        dataset_info_dir=args.dataset_info_dir,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    # Create training dataloader
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
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )

    # Create validation dataset
    if args.test_dataset_info_dir is None:
        args.test_dataset_info_dir = args.dataset_info_dir

    val_img_prep = 'dinov2_base_noaug' if 'dinov2_base' in args.image_prep else 'clip_base_noaug'
    val_data = ArtistDataset(
        split=args.test_dataset_split,
        image_prep=val_img_prep,
        dataset_info_dir=args.test_dataset_info_dir,
        label_csv_col=args.label_csv_col,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    num_classes = len(train_data.class_mapping)

    # Load prototype
    prototype = torch.from_numpy(np.load(args.prototype_path))
    print(f"Prototype shape: {prototype.shape}")
    assert prototype.shape[0] == num_classes, f"Prototype shape {prototype.shape} does not match num_classes {num_classes}"

    # Create model
    model = ProtoClip(
        prototype=prototype,
        model_arch=args.model_arch,
        temperature=args.temperature
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Verify all parameters are trainable
    if accelerator.is_main_process:
        print("Assert there's no non-trainable parameters...")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert False, f"Parameter {name} is not trainable"

    # Prepare model, optimizer, and dataloaders
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Cast loss function
    criterion = criterion.to(device=accelerator.device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_ckpt is not None:
        accelerator.load_state(args.resume_from_ckpt)
        print(f"Loaded model from {args.resume_from_ckpt}")
        start_epoch = int(args.resume_from_ckpt.split("epoch")[-1])
        print(f"Resume from epoch {start_epoch}")

    # Training loop
    for e in tqdm(range(start_epoch, args.num_training_epochs), desc="Training epochs"):
        model.train()

        for idx, batch in tqdm(enumerate(train_loader), desc="batches", total=len(train_loader)):
            images = batch["img_pixel_values"]
            gt_labels = batch["label_idx"]

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, gt_labels)

            accelerator.backward(loss)
            optimizer.step()

            # Gather data for logging
            pred_labels = torch.argmax(output, dim=1)
            pred_labels, gt_labels = accelerator.gather_for_metrics(
                (pred_labels, gt_labels))
            batch_corrects = torch.sum(pred_labels == gt_labels)

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    log_dict = {
                        'train/loss': loss,
                        'train/accuracy': batch_corrects / len(gt_labels)
                    }
                    for tracker in accelerator.trackers:
                        tracker.log(log_dict)

        if accelerator.sync_gradients:
            if accelerator.is_main_process:
                # Save model
                if e % args.checkpoint_every_n_epochs == 0:
                    accelerator.save_state(
                        output_dir=f"{OUTPUT_DIR}/ckpt/epoch{e}")
                    print(f"Saved model epoch {e}")
                    # Remove old optimizer.bin
                    for i in range(e - 1, -1, -1):
                        if os.path.exists(f"{OUTPUT_DIR}/ckpt/epoch{i}/optimizer.bin"):
                            print(
                                f"Removing old optimizer.bin {OUTPUT_DIR}/ckpt/epoch{i}/optimizer.bin")
                            os.remove(
                                f"{OUTPUT_DIR}/ckpt/epoch{i}/optimizer.bin")

                # Validation
                if e % args.log_every_n_epochs == 0:
                    eval_model = accelerator.unwrap_model(model)
                    eval_model.eval()
                    eval_model = eval_model.to(device=accelerator.device)

                    with torch.no_grad():
                        val_log = validation(
                            eval_model,
                            val_loader,
                            criterion=criterion,
                            device=accelerator.device
                        )

                    # Log to wandb
                    for key in val_log.keys():
                        print(f"{key}: {val_log[key]:.3f}")

                    for tracker in accelerator.trackers:
                        tracker.log(val_log)

    accelerator.end_training()
    return model


if __name__ == "__main__":
    args = parse_args()
    main(args)
