import argparse
import os
import os.path as osp

from search import top1_nn_single_artist_search


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_dir', type=str, default="dataset_sdxl_simple",
                        help='The name of the evaluation dataset')
    parser.add_argument('--eval_splits', type=str, default='test_all_unseen_query,test_artist',
                        help="comma separated list of the evaluation dataset splits to use")
    parser.add_argument('--what_to_retrieve', type=str, default='sdxl,sd15,pixart,journeydb',
                        help='What datasets to retrieve from. Options: sdxl, sd15, pixart, journeydb, sample_sdxl')
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


def get_database_chunk_dirs_list(what_to_retrieve):
    l_what_to_retrieve = what_to_retrieve.split(",")
    l_data_dirs = []

    if "sdxl" in l_what_to_retrieve:
        l_data_dirs.append("dataset_sdxl_complex")
        l_data_dirs.append("dataset_sdxl_simple")
    if "sd15" in l_what_to_retrieve:
        l_data_dirs.append("dataset_sd15_complex")
        l_data_dirs.append("dataset_sd15_simple")
    if "pixart" in l_what_to_retrieve:
        l_data_dirs.append("dataset_pixart_complex")
        l_data_dirs.append("dataset_pixart_simple")
    if "journeydb" in l_what_to_retrieve:
        l_data_dirs.append("dataset_journeydb_moreimages")
    if "sample_sdxl" in l_what_to_retrieve:
        l_data_dirs.append("dataset_sample_sdxl_complex")
        l_data_dirs.append("dataset_sample_sdxl_simple")

    print(f"{len(l_data_dirs)} datasets to retrieve from: {l_data_dirs}")
    return l_data_dirs


def main(args):
    os.makedirs(osp.join(args.embed_dir, args.model_id), exist_ok=True)

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

    # NOTE: Feature extraction only needs to be run once because it saves the embeddings to a directory
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

    print(" Search embeddings:")
    # embedding dir format: {args.embed_dir}/{args.model_id}/nn_embeddings/{args.data-dir}/{split}/
    parser = top1_nn_single_artist_search.get_parser()

    os.makedirs(osp.join(args.embed_dir, args.model_id,
                "nn_embeddings", args.data_dir), exist_ok=True)

    print("Evaluate seen artist accuracy --- Query: test_artist, Database: train")

    database_chunk_dir = ','.join(
        [f"{args.embed_dir}/{args.model_id}/nn_embeddings/{data_dir}/train" for data_dir in l_data_dirs])
    database_csvs = ','.join(
        [f"{data_dir}/train_imgs.csv" for data_dir in l_data_dirs])

    search_args = parser.parse_args([
        '--query_csv', 'test_artist',
        '--database_csv', database_csvs,
        '--query-chunk-dir', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_artist",
        '--database-chunk-dir', database_chunk_dir,
        '--method', 'IP',
        '--data_dir', f"{args.data_dir}",
        '--preds_csv', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_artist/preds-retrieve_from_{args.what_to_retrieve}.csv",
        '--metrics_csv_col', 'Seen artists: NN from gen',
        '--model_type', args.model_type,
        '--model_id', args.model_id,
        '--metrics_csv_path', metrics_csv_path,
    ])
    top1_nn_single_artist_search.main(search_args)

    print("Evaluate held-out artist accuracy --- Query: test_all_unseen_query, Database: test_all_unseen_support")
    database_chunk_dir = ','.join(
        [f"{args.embed_dir}/{args.model_id}/nn_embeddings/{data_dir}/test_all_unseen_support" for data_dir in l_data_dirs])
    database_csvs = ','.join(
        [f"{data_dir}/test_all_unseen_support_imgs.csv" for data_dir in l_data_dirs])

    search_args = parser.parse_args([
        '--artist_mapping_file', "test_artists.txt",
        '--query_csv', 'test_all_unseen_query',
        '--database_csv', database_csvs,
        '--query-chunk-dir', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_all_unseen_query",
        '--database-chunk-dir', database_chunk_dir,
        '--method', 'IP',
        '--data_dir', f"{args.data_dir}",
        '--preds_csv', f"{args.embed_dir}/{args.model_id}/nn_embeddings/{args.data_dir}/test_all_unseen_query/preds-retrieve_from_{args.what_to_retrieve}.csv",
        '--metrics_csv_col', 'Held-out artists: NN from gen',
        '--model_type', args.model_type,
        '--model_id', args.model_id,
        '--metrics_csv_path', metrics_csv_path,
    ])
    top1_nn_single_artist_search.main(search_args)

    print(f"Metrics saved at: {metrics_csv_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
