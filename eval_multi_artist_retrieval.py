import argparse
import os
import os.path as osp

from eval_single_artist_retrieval import get_database_chunk_dirs_list
from search import topk_nn_multi_artist_search


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10,
                        help='Top K for nearest neighbor search')

    # Dataset
    # dataset_multihead_100artists_sdxlonly
    parser.add_argument('--data_dir', type=str, default=None,
                        help='The name of the evaluation dataset')
    parser.add_argument('--what_to_retrieve', type=str, default='sdxl',
                        help='What datasets to retrieve from.')
    parser.add_argument('--eval_splits', type=str, default='test_all_unseen_query,test_artist',
                        help="comma separated list of the evaluation dataset splits to use")
    parser.add_argument('--train_splits', type=str, default='train,test_all_unseen_support',
                        help="comma separated list of the train dataset splits to retrieve embeddings from")
    parser.add_argument('--label_csv_col', type=str, default='prompt_label',
                        help='The column name of the label in the csv file')


    # Model
    parser.add_argument('--model_type', default='clip_vit_large', type=str,
                        choices=['clip_vit_large', 'dinov2_vit_large',
                                 'csd', 'abc_dino', 'abc_clip'],
                        help='The type of model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                        help="Needed if using CSD or AbC: the path to the model checkpoint")

    # Dataloader
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='embedding extraction: number of data loading workers per GPU')
    parser.add_argument('-b', '--batch_size', default=1024,
                        type=int, help="embedding extraction: batch size per GPU")

    # Where to save the embeddings
    parser.add_argument('--embed_dir', default='RESULTS', type=str,
                        help='Parent directory to save embeddings, metrics, and predictions. The model ID will be appended to this directory.')
    parser.add_argument('--model_id', type=str, default=None, required=True,
                        help="The name of the directory under embed_dir to save this evaluation run's results")

    args = parser.parse_args()
    return args


def main(args):

    os.makedirs(f"{args.embed_dir}/{args.model_id}/", exist_ok=True)

    print(f"Model ID: {args.model_id}")
    print(f"Model Type: {args.model_type}")
    print(f"Model path: {args.model_path}")

    base_data_dir = osp.basename(args.data_dir)
    metrics_csv_path = osp.join(
        args.embed_dir, f"metrics_on_{base_data_dir}.csv")
    print(f"Metrics will be saved at:")
    print(metrics_csv_path)

    # log model_ckpt.txt
    if args.model_path:
        with open(osp.join(args.embed_dir, args.model_id, "model_ckpt.txt"), 'w') as f:
            f.write(f"model_path: {args.model_path}\n")

    # NOTE: Feature extraction and saving only needs to be run once
    # Extract query embeddings
    eval_splits = args.eval_splits.split(',')

    for split in eval_splits:
        print(f"Extract {args.data_dir} {split} embeddings")
        command = f"""
        python search/extract_image_features.py --csv_name {split} \
            --num_gpus {args.num_gpus} \
            --model_type {args.model_type} \
            -b {args.batch_size} -j {args.workers} \
            --embed_dir {args.embed_dir}/{args.model_id}/nn_embeddings \
            --data_dir {args.data_dir} \
            --model_path  {args.model_path} \
            --label_csv_col {args.label_csv_col}
        """
        os.system(command)

    # Extract database embeddings
    l_data_dirs = get_database_chunk_dirs_list(args.what_to_retrieve)
    train_splits = args.train_splits.split(',')
    for data_dir in l_data_dirs:
        for split in train_splits:
            print(f"Extract {data_dir} {split} embeddings")
            command = f"""
            python search/extract_image_features.py --csv_name {split} \
                --num_gpus {args.num_gpus} \
                --model_type {args.model_type} \
                -b {args.batch_size} -j {args.workers} \
                --embed_dir {args.embed_dir}/{args.model_id}/nn_embeddings \
                --data_dir {data_dir} \
                --model_path  {args.model_path} \
                --label_csv_col {args.label_csv_col}
            """
            os.system(command)

    print("Search embeddings")
    # embedding dir format: {args.embed_dir}/{args.model_id}/nn_embeddings/{args.data-dir}/{split}/
    parser = topk_nn_multi_artist_search.get_parser()
    args = parser.parse_args(args=[])
    os.makedirs(osp.join(args.embed_dir, args.model_id,
                "nn_embeddings", args.data_dir), exist_ok=True)

    topk = args.topk

    metrics_csv_path = f"{args.embed_dir}/k{topk}_metrics_on_{base_data_dir}.csv"
    print(f"Metrics will be saved at:")
    print(metrics_csv_path)

    print("Evaluate seen artist accuracy --- Query: test_artist, Database: train")
    database_chunk_dir = ','.join(
        [f"{args.embed_dir}/{args.model_id}/nn_embeddings/{data_dir}/train" for data_dir in l_data_dirs])
    database_csvs = ','.join(
        [f"{data_dir}/train_imgs.csv" for data_dir in l_data_dirs])
    search_args = parser.parse_args([
        '--topk', str(topk),
        '--query_csv', 'test_artist',
        '--database_csv', database_csvs,
        '--query-chunk-dir', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_artist",
        '--database-chunk-dir', database_chunk_dir,
        '--method', 'IP',
        '--data_dir', f"{args.data_dir}",
        '--preds_csv', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{base_data_dir}/test_artist/k{topk}_preds.csv",
        '--metrics_csv_col', 'Seen artists: NN from gen',
        '--model_type', args.model_type,
        '--model_id', args.model_id,
        '--metrics_csv_path', metrics_csv_path,
    ])
    topk_nn_multi_artist_search.main(search_args)

    print("Evaluate held-out artist accuracy --- Query: test_all_unseen_query, Database: test_all_unseen_support")
    database_chunk_dir = ','.join(
        [f"{args.embed_dir}/{args.model_id}/nn_embeddings/{data_dir}/test_all_unseen_support" for data_dir in l_data_dirs])
    database_csvs = ','.join(
        [f"{data_dir}/test_all_unseen_support_imgs.csv" for data_dir in l_data_dirs])
    search_args = parser.parse_args([
        '--topk', str(topk),
        '--artist_mapping_file', "test_artists.txt",
        '--query_csv', 'test_all_unseen_query',
        '--database_csv', database_csvs,
        '--query-chunk-dir', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_all_unseen_query",
        '--database-chunk-dir', database_chunk_dir,
        '--method', 'IP',
        '--data_dir', f"{args.data_dir}",
        '--preds_csv', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{base_data_dir}/test_all_unseen_query/k{topk}_preds.csv",
        '--metrics_csv_col', 'Held-out artists: NN from gen',
        '--model_type', args.model_type,
        '--model_id', args.model_id,
        '--metrics_csv_path', metrics_csv_path,
    ])
    topk_nn_multi_artist_search.main(search_args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
