import argparse
import os
import sys

# Add parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.nn_search_utils import *
from utils.eval_utils import calculate_ranked_map, update_metrics_csv


def get_parser():
    parser = argparse.ArgumentParser('dynamicDistances-NN Search Module')
    parser.add_argument('--topk', type=int, default=10,
                        help='Top K for nearest neighbor search')
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

    # Perform KNN search (top-k)
    D, nns_all = perform_knn_search(
        query_embeddings, val_embeddings, topk=args.topk, method=args.method)

    print(f"Saving top-{args.topk} predictions to {args.preds_csv}")

    # Prepare predictions DataFrame (multi-artist prediction case)
    preds_df = prepare_predictions_dataframe(
        q_annotations, v_annotations, val_embeddings, nns_all, D, args.label_csv_col, topk=args.topk
    )

    print(args.metrics_csv_col)

    ranked_map, precisions = calculate_ranked_map(preds_df, topk=args.topk)
    print(f"Ranked mAP@{args.topk}: {ranked_map}")
    # also save each row's ranked precision to preds csv
    # round precisions to 4 decimal places
    precisions = [round(precision, 4) for precision in precisions]
    preds_df["row_ranked_precision"] = precisions
    preds_df.to_csv(args.preds_csv, index=False)

    map_col_name = args.metrics_csv_col.replace(
        'artists:', f"artists Ranked mAP@{args.topk}:")

    eval_dataset = args.data_dir.split('/')[-1]
    metrics_dict = {
        "model_type": [args.model_type],
        "model_id": [args.model_id],
        map_col_name: [ranked_map],
        "eval_dataset": [eval_dataset],
    }

    update_metrics_csv(args.metrics_csv_path, metrics_dict)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)
