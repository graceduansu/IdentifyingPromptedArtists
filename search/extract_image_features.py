import argparse
import multiprocessing as mp
import os
import sys
from glob import glob

# Add parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from data.datasets import ArtistDataset
from search.embeddings import Embeddings, save_chunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=4)

    # Dataset
    parser.add_argument('--data_dir', type=str, default=None,
                        help='The directory of concerned dataset')
    parser.add_argument('--csv_name', type=str, default='test_artist',
                        help='The csv file containing the dataset info')


    # Model
    parser.add_argument('--model_type', default='clip_vit_large', type=str,
                        choices=['clip_vit_large',
                                 'dinov2_vit_large', 'csd', 'abc_dino', 'abc_clip'],
                        help='The type of model to use for feature extraction')
    parser.add_argument('-mp', '--model_path', type=str, default=None,
                        help='Path to the model checkpoint if using CSD or AbC model')
    parser.add_argument('--abc_ckpt_dict_key', type=str, default='synth_mapper',
                        choices=['synth_mapper', 'real_mapper'],
                        help='Key to load the AbC model mapper state dict')
    # Dataloader
    parser.add_argument('--image_prep', type=str,
                        default='clip_base_noaug', help="Image preprocessing. Also automatically determined"
                        " by --model type in `set_image_prep()`")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    # Where to save the embeddings
    parser.add_argument('--embed_dir', default='./nn_embeddings',
                        type=str, help='Directory to save embeddings')

    parser.add_argument('--label_csv_col', type=str, default='prompt_label',
                        help='The column name of the label in csv')

    args = parser.parse_args()
    return args


def set_model(args):
    model_type = args.model_type
    model_path = args.model_path

    if model_type == 'clip_vit_large':
        import clip
        model, preprocess = clip.load('ViT-L/14')
        model.visual.proj = None
        model = model.visual

    elif model_type == 'dinov2_vit_large':
        model = torch.hub.load('facebookresearch/dinov2',
                               'dinov2_vitl14', pretrained=True)

    elif model_type == 'csd':
        from models.csd import CSD_CLIP, convert_state_dict
        model = CSD_CLIP('vit_large', 'default')
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict, strict=False)

    elif model_type.startswith('abc_'):
        from models.abc import LinearProbe as AbcModel
        if args.model_type == 'abc_clip':
            import clip
            feat_model = clip.load('ViT-B/16', jit=False)[0].visual
            feat_dim = 512
            model = AbcModel(feat_model, feat_dim)
        elif args.model_type == 'abc_dino':
            feat_model = torch.hub.load(
                'facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
            feat_dim = feat_model.num_features
            model = AbcModel(feat_model, feat_dim)

        all_dicts = torch.load(model_path, weights_only=False)
        print(f"Load abc ckpt dict key: {args.abc_ckpt_dict_key}")
        model.mapper.load_state_dict(all_dicts[args.abc_ckpt_dict_key])

    else:
        raise NotImplementedError(f"model_type {model_type} not implemented")

    return model


def set_image_prep(args):
    # if dinov2_vit_large, use dinov2_base_noaug
    if args.model_type == 'dinov2_vit_large' or args.model_type == 'abc_dino':
        args.image_prep = 'dinov2_base_noaug'
    else:
        args.image_prep = 'clip_base_noaug'
    print(f"[INFO] Image_prep set to: {args.image_prep}")
    return args.image_prep


def my_worker(args):

    torch.cuda.set_device(args.local_rank)
    torch.manual_seed(args.seed)

    embsavepath = os.path.join(
        args.embed_dir, args.dataset_name, args.csv_name)

    model = set_model(args)
    model.to(dtype=torch.float32, device='cuda')
    model.eval()

    args.image_prep = set_image_prep(args)

    dataset = ArtistDataset(split=args.csv_name, image_prep=args.image_prep,
                            dataset_info_dir=args.data_dir, label_csv_col=args.label_csv_col)
    dataset = torch.utils.data.Subset(dataset, args.dataset_idx_list)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Sub Dataset: {len(dataset)} imgs")

    # Start inference
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), desc=f"Device {args.local_rank} Extracting", total=len(data_loader)):
            images = batch['img_pixel_values'].to(dtype=torch.float32)
            images = images.cuda(non_blocking=True)
            filenames = batch['img_path']
            idxs = batch['dataset_idx']

            if args.model_type == 'csd':
                bb_feats, cont_feats, features = model(images)
            else:
                features = model(images)

            # save chunk
            l_query_features = list(features.cpu().detach().numpy())
            save_chunk(l_query_features, filenames, idxs[0].item(), embsavepath)


def main(args):
    gpus = list(range(args.num_gpus))

    process_stack = []

    csv_name = f"{args.csv_name}_imgs.csv"
    df = pd.read_csv(os.path.join(args.data_dir, csv_name))

    dataset_idx_list = df.index.tolist()

    print(f"Total number of images: {len(dataset_idx_list)}")

    split = len(dataset_idx_list) // len(gpus)
    remainder = len(dataset_idx_list) % len(gpus)
    for local_rank in gpus:
        worker_args = args
        worker_args.local_rank = local_rank
        start_idx = local_rank*split
        end_idx = (local_rank+1)*split
        if local_rank == len(gpus)-1:
            end_idx += remainder

        worker_args.dataset_idx_list = dataset_idx_list[start_idx:end_idx]

        print(
            f"[INFO] Starting process on GPU {local_rank}: Idx {start_idx} to Idx {end_idx}")

        process = mp.Process(target=my_worker, args=(worker_args,))
        process.start()
        process_stack.append(process)

    # wait for each process running
    for process in process_stack:
        process.join()

    # aggregate chunks and save as embeddings
    emb = Embeddings(data_dir=None, chunk_dir=embsavepath,
                     chunked=True, file_ext='.pkl')
    emb.filenames = list(emb.filenames)

    # delete all old temporary chunk files
    for chunk in os.listdir(embsavepath):
        os.remove(os.path.join(embsavepath, chunk))

    l_query_features = list(np.asarray(emb.embeddings))
    l_pathlist = emb.filenames
    save_chunk(l_query_features, l_pathlist, 0, embsavepath)

    print("Done!")


if __name__ == "__main__":
    # distributed setting
    mp.set_start_method("spawn")
    args = parse_args()
    os.makedirs(args.embed_dir, exist_ok=True)
    args.dataset_name = args.data_dir.split('/')[-1]
    print(f"dataset_name: {args.dataset_name}")

    # make embedding subfolder
    embsavepath = os.path.join(
        args.embed_dir, args.dataset_name, args.csv_name)
    print(f"Embedding save path: {embsavepath}")

    # check if embsavepath is empty
    if os.path.exists(embsavepath) and len(glob(f"{embsavepath}/*.pkl")) > 0:
        print(f"[INFO] Embedding save path is not empty. Quitting...")
        sys.exit(0)

    print(f"Creating embedding save path")
    os.makedirs(embsavepath, exist_ok=True)

    main(args)
