import argparse
import os
import sys

# Add parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from search.nn_search_utils import *
from utils.eval_utils import update_metrics_csv


def get_parser():
    parser = argparse.ArgumentParser('dynamicDistances-NN Search Module')
    parser.add_argument('--method', type=str, default='IP',
                        choices=['IP', 'L2'], help='The method to do NN search')
    parser.add_argument('--query-chunk-dir', type=str, required=True,
                        help='The directory where chunked query embeddings should are already saved')
    parser.add_argument('--database-chunk-dir', type=str, required=True,
                        help='The directory where chunked val embeddings should are already saved. Can be comma separated list of directories')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='The directory of the query dataset')
    parser.add_argument('--preds_csv', type=str, default='preds',
                        help='The csv file to save the predictions')
    parser.add_argument('--artist_mapping_file',
                        default='map_prompt_label.txt', type=str)

    parser.add_argument('--query_csv', type=str,
                        default='test_artist', help='prefix of query csv file')
    parser.add_argument('--database_csv', type=str, default=None,
                        help='comma separated paths of each database csv file')

    parser.add_argument('--label_csv_col', type=str, default='prompt_label',
                        help='The column name of the label in csv')

    # how to log metrics
    parser.add_argument('--metrics_csv_path', type=str,
                        required=True, help='The path to save the metrics csv')
    parser.add_argument('--metrics_csv_col', type=str, required=True,
                        # choices=['Seen artists: NN from gen',
                        #          'Seen artists: NN from real',
                        #         'Held-out artists: NN from real',
                        #         'Held-out artists: NN from gen',
                        # ],
                        help='The column name to save the metrics')
    parser.add_argument('--model_type', type=str,
                        required=True, help='The model type')
    parser.add_argument('--model_id', type=str,
                        required=True, help='The model id')
    return parser


def main(args):
    # Save results as csv
    save_dir = os.path.dirname(args.preds_csv)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}")
    print(f"preds_csv: {args.preds_csv}")

    # Load embeddings using utility
    query_embeddings, val_embeddings = load_embeddings(
        args.query_chunk_dir, args.database_chunk_dir
    )

    # Filter embeddings to match CSVs
    q_annotations, v_annotations, query_embeddings, val_embeddings = filter_embeddings(
        args.data_dir, args.query_csv, query_embeddings, args.database_csv, val_embeddings
    )

    # Load artist classes
    classes = load_artist_classes(args.database_csv, args.artist_mapping_file)

    # Perform KNN search (top-1)
    D, nns_all = perform_knn_search(
        query_embeddings, val_embeddings, topk=1, method=args.method)
    print(f"nn_all shape: {nns_all.shape}")

    # Prepare predictions DataFrame
    preds_df = prepare_predictions_dataframe(
        q_annotations, v_annotations, val_embeddings, nns_all, D, args.label_csv_col, topk=1
    )

    # Calculate accuracy
    mode_to_index = {classname: i for i, classname in enumerate(classes)}
    gts = np.asarray(
        list(map(lambda x: mode_to_index[x], preds_df['true_artist'].tolist())))
    preds = np.asarray(
        list(map(lambda x: mode_to_index[x], preds_df['pred_artist'].tolist())))
    acc = np.mean(gts == preds)
    print(f"Accuracy: {acc}")

    eval_dataset = args.data_dir.split('/')[-1]
    print(f"metrics csv eval_dataset: {eval_dataset}")
    metrics_dict = {
        "model_type": [args.model_type],
        "model_id": [args.model_id],
        args.metrics_csv_col: [acc],
        "eval_dataset": [eval_dataset],
    }
    update_metrics_csv(args.metrics_csv_path, metrics_dict)

    # Save predictions to csv
    preds_df.to_csv(args.preds_csv, index=False)
    print(f"Saved predictions to {args.preds_csv}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)
