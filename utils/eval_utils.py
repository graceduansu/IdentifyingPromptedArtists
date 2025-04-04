import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_ranked_map(df, topk=5):
    """
    Returns:
    - float: mean average precision
    - list: of precisions for each image
    """
    # convert to str
    df['all_prompt_labels'] = df['all_prompt_labels'].astype(str)

    pred_names = [f"{i}_pred_artist" for i in range(topk)]
    precisions = []
    for i, row in df.iterrows():
        running_correct = 0
        true_labels = row['all_prompt_labels'].split(";")
        precision_sum = 0
        # NOTE: we only reward the first correct retrieval of a predicted artist so that retrieval-based method mAP calculation is
        # consistent with classifier mAP calculation
        already_predicted = set()
        for rank, pred_name in enumerate(pred_names):
            artist = row[pred_name]
            if artist in already_predicted:
                continue
            if artist in true_labels:
                running_correct += 1
                precision_sum += (running_correct) / (rank+1)
            already_predicted.add(artist)

        precisions.append(precision_sum / len(true_labels))

    return np.mean(precisions), precisions


def update_metrics_csv(metrics_csv_path, metrics_dict):
    col_order = ['model_type',
                 'model_id',
                 'eval_dataset',
                 'Seen artists: NN from real',
                 'Held-out artists: NN from real',

                 'Seen artists: NN from gen',
                 'Held-out artists: NN from gen',

                 'Seen artists: Classifier',
                 'Held-out artists: Classifier',
                 ]

    if not os.path.exists(metrics_csv_path):
        data = {}

        curr_df = pd.DataFrame(data, columns=col_order)

        df = pd.DataFrame.from_dict(metrics_dict)

        curr_df = pd.concat([curr_df, df], ignore_index=True)
        curr_df.to_csv(metrics_csv_path, index=False)
        return

    curr_df = pd.read_csv(metrics_csv_path)

    # check if combination of model_id and eval_dataset already exists
    match_row = curr_df[curr_df['model_id'] == metrics_dict['model_id'][0]]

    if len(match_row) == 0:
        df = pd.DataFrame.from_dict(metrics_dict)
        curr_df = pd.concat([curr_df, df], ignore_index=True)
    else:
        # update row
        for key, value in metrics_dict.items():
            curr_df.loc[match_row.index, key] = value[0]

    # sort by model_id
    curr_df = curr_df.sort_values(by=['model_id'])

    # round to 3 decimal places
    for col in curr_df.columns:
        if pd.api.types.is_float_dtype(curr_df[col]):
            curr_df[col] = curr_df[col].apply(lambda x: round(x, 3))

    # fill missing columns
    for col in col_order:
        if col not in curr_df.columns:
            curr_df[col] = None

    curr_df.to_csv(metrics_csv_path, index=False)


def evaluate_multi_artist_preds(class_mapping, parent_dir, split, key_name, model_type, model_id, eval_dataset, topk=10):
    """
    Evaluate predictions for multi-artist classification task.
    Args:
        class_mapping (dict): Mapping of class names to indices.
        parent_dir (str): Directory containing the predictions. 
        split (str): Dataset split (e.g., 'test_artist').
        key_name (str): Key name for the predictions.
        model_type (str): Name of the model used for evaluation.
        model_id (str): ID of the model used for evaluation.
        eval_dataset (str): Name of the evaluation dataset.
        topk (int): Number of top predictions to consider.
    Returns:
        metrics_dict (dict): Dictionary containing evaluation metrics.
    """
    print('class mapping')
    print(class_mapping)
    # open npz
    y_pred = {}
    y_true = {}
    y_pred_probs = {}

    with np.load(f"{parent_dir}/{split}/y_pred_probs.npz") as f:
        # shape: (num_samples, num_classes)
        y_pred_probs[key_name] = f["key_name"]

    with np.load(f"{parent_dir}/{split}/y_true.npz") as f:
        y_true[key_name] = f["key_name"]

    with np.load(f"{parent_dir}/{split}/y_pred.npz") as f:
        y_pred[key_name] = f["key_name"]

    dataset_info_dir = parent_dir.split("/")[-1]

    data_df = pd.read_csv(f"{dataset_info_dir}/{split}_imgs.csv")
    rename_dict = {"prompt_type": "true_prompt_type",
                   "source_label": "true_source", "img_path": "original_img_path"}
    data_df = data_df.rename(columns=rename_dict)

    # resave preds.csv with top k predictions and probabilities
    out_csv_path = f"{parent_dir}/{split}/preds.csv"

    topk_idxs = np.argsort(y_pred_probs[key_name], axis=1)[:, ::-1][:, :topk]
    topk_y_pred = {}
    topk_y_pred_probs = {}
    topk_y_pred_probs[key_name] = y_pred_probs[key_name][np.arange(len(y_pred_probs[key_name]))[:, None],
                                                         topk_idxs]  # shape: (num_samples, topk)
    topk_y_pred[key_name] = topk_idxs  # shape: (num_samples, topk)

    print(y_pred_probs[key_name].shape, 'y_pred_probs')
    print('class mapping', class_mapping[key_name])

    for k in range(topk):
        data_df[f'{k}_pred_artist_score'] = topk_y_pred_probs[key_name][:, k]
        data_df[f'{k}_pred_artist'] = [class_mapping[key_name][i]
                                       for i in topk_y_pred[key_name][:, k]]

    # save to metrics_csv_path

    metrics_dict = {
        "model_type": [model_type],
        "model_id": [model_id],
        "eval_dataset": [eval_dataset]
    }

    # Get one-hot vectors for y_true
    df_y_true = []
    for i, row in data_df.iterrows():
        label_names = row['all_prompt_labels'].split(';')
        label_vec = np.zeros(len(class_mapping[key_name]))
        target_idxs = np.array(
            [class_mapping[key_name].index(l) for l in label_names])
        np.put_along_axis(label_vec, target_idxs, 1, axis=0)
        df_y_true.append(label_vec)
    df_y_true = np.stack(df_y_true, axis=0)

    df_y_scores = y_pred_probs[key_name]

    assert df_y_true.shape == df_y_scores.shape, f"y_true shape: {df_y_true.shape}, y_scores shape: {df_y_scores.shape}"

    ap = average_precision_score(df_y_true, df_y_scores)

    # update metrics dict
    if "test_artist" in split or "train" in split:
        metrics_col = "AP on Seen artists"
    elif "test_all_unseen" in split:
        metrics_col = "AP on Unseen artists"

    metrics_dict[metrics_col] = [ap]

    ranked_map, precisions = calculate_ranked_map(data_df, topk=topk)

    # also save each row's ranked precision to preds csv
    # round precisions to 4 decimal places
    precisions = [round(precision, 4) for precision in precisions]
    data_df["row_ranked_precision"] = precisions
    data_df.to_csv(out_csv_path, index=False)

    print(df_y_true.shape)
    print(df_y_scores.shape)
    print(f"ranked_map@{topk}: {ranked_map}")
    metrics_dict[metrics_col.replace(
        "AP", f"Ranked mAP@{topk}")] = [ranked_map]

    # calculate roc auc
    roc_score = roc_auc_score(df_y_true, df_y_scores)

    # update metrics dict
    if "test_artist" in split or "train" in split:
        metrics_col = "ROC AuC on Seen artists"
    elif "test_all_unseen" in split:
        metrics_col = "ROC AuC on Unseen artists"

    metrics_dict[metrics_col] = [roc_score]

    return metrics_dict
