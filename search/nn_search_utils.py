import os
import sys

# Add parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import pandas as pd

from .embeddings import Embeddings
from .faiss_search import FaissIndex


def load_embeddings(query_chunk_dir, database_chunk_dir):
    """Load query and database embeddings from directories."""
    query_embeddings = Embeddings(None, query_chunk_dir,
                                  files=None,
                                  chunked=True,
                                  file_ext='.pkl')

    print(
        f"Number of query embeddings loaded: {len(query_embeddings.filenames)}")

    l_val_chunk_dirs = database_chunk_dir.split(',')
    l_val_embeddings = []

    for val_chunk_dir in l_val_chunk_dirs:
        val_embeddings = Embeddings(None, val_chunk_dir,
                                    files=None,
                                    chunked=True,
                                    file_ext='.pkl')
        l_val_embeddings.append(val_embeddings)

    # Combine embeddings from all directories
    val_embeddings = l_val_embeddings[0]  # Use first one as base
    val_embeddings.embeddings = np.concatenate(
        [val_emb.embeddings for val_emb in l_val_embeddings], axis=0)

    val_filenames = []
    for val_emb in l_val_embeddings:
        val_filenames.extend(val_emb.filenames)
    val_embeddings.filenames = val_filenames

    print(f"Number of val embeddings loaded: {len(val_embeddings.filenames)}")
    return query_embeddings, val_embeddings


def filter_embeddings(data_dir, query_csv, query_embeddings, database_csv, val_embeddings):
    """Filter embeddings to match available data in the CSVs"""
    q_annotations = pd.read_csv(f'{data_dir}/{query_csv}_imgs.csv')

    query_csv_len = len(q_annotations)

    temp = pd.DataFrame({'filename': query_embeddings.filenames,
                        'index': np.arange(len(query_embeddings.filenames))})
    q_annotations = q_annotations.merge(
        temp, left_on='img_path', right_on='filename', how='inner')
    query_embeddings.embeddings = query_embeddings.embeddings[q_annotations['index'].values]
    query_embeddings.filenames = q_annotations['filename'].values
    import pdb; pdb.set_trace()

    assert len(query_embeddings.filenames) != 0, "No images in query"
    assert len(
        query_embeddings.filenames) >= query_csv_len, f"Missing query embeddings: {query_csv_len - len(query_embeddings.filenames)}"

    print(f"Number of query images: {len(query_embeddings.filenames)}")

    v_annotations = []
    for csv_file in database_csv.split(','):
        v_annotations.append(pd.read_csv(csv_file))

    v_annotations = pd.concat(v_annotations, ignore_index=True)

    temp = pd.DataFrame({'filename': val_embeddings.filenames,
                        'index': np.arange(len(val_embeddings.filenames))})
    v_annotations = v_annotations.merge(
        temp, left_on='img_path', right_on='filename', how='inner')
    val_embeddings.embeddings = val_embeddings.embeddings[v_annotations['index'].values]
    val_embeddings.filenames = v_annotations['filename'].values

    assert len(val_embeddings.filenames) != 0, "No images in database"
    print(f"Number of database images: {len(val_embeddings.filenames)}")

    return q_annotations, v_annotations, query_embeddings, val_embeddings


def load_artist_classes(database_csv, artist_mapping_file):
    """Load artist class mapping from file."""
    classes = []
    map_dir = os.path.dirname(database_csv.split(',')[-1])
    with open(f'{map_dir}/{artist_mapping_file}', 'r') as f:
        for line in f:
            classes.append(line.strip())

    print(f"Number of classes: {len(classes)}")
    return classes


def perform_knn_search(query_embeddings, val_embeddings, topk=1, method='IP'):
    """Prepare embeddings and perform FAISS search."""
    # Convert filenames to lists
    query_embeddings.filenames = list(query_embeddings.filenames)
    val_embeddings.filenames = list(val_embeddings.filenames)

    # Reshape embeddings for search
    try:
        b, h, w = query_embeddings.embeddings.shape
        query_embeddings.embeddings = query_embeddings.embeddings.reshape(
            b, 1, h * w)
    except ValueError:
        b, d = query_embeddings.embeddings.shape
        query_embeddings.embeddings = query_embeddings.embeddings.reshape(
            b, 1, d)

    try:
        b, h, w = val_embeddings.embeddings.shape
        val_embeddings.embeddings = val_embeddings.embeddings.reshape(
            b, 1, h * w)
    except ValueError:
        b, d = val_embeddings.embeddings.shape
        val_embeddings.embeddings = val_embeddings.embeddings.reshape(b, 1, d)

    query_embeddings.embeddings = query_embeddings.embeddings.astype(
        np.float32)
    val_embeddings.embeddings = val_embeddings.embeddings.astype(np.float32)

    # Normalize embeddings if needed
    query_embeddings_norm = np.linalg.norm(query_embeddings.embeddings, axis=2)
    if not np.allclose(query_embeddings_norm, 1.0, rtol=1e-3):
        query_embeddings.embeddings = query_embeddings.embeddings / \
            query_embeddings_norm[:, :, None]

    val_embeddings_norm = np.linalg.norm(val_embeddings.embeddings, axis=2)
    if not np.allclose(val_embeddings_norm, 1.0, rtol=1e-3):
        val_embeddings.embeddings = val_embeddings.embeddings / \
            val_embeddings_norm[:, :, None]

    # Build FAISS index
    embedding_size = query_embeddings.embeddings[0].shape[1]
    if method == 'IP':
        print("Using IP")
        index_func = faiss.IndexFlatIP
    else:
        index_func = faiss.IndexFlatL2

    search_module = FaissIndex(
        embedding_size=embedding_size, index_func=index_func)
    queries = np.asarray(query_embeddings.embeddings).reshape(
        len(query_embeddings.embeddings), embedding_size)
    database = np.asarray(val_embeddings.embeddings).reshape(
        len(val_embeddings.embeddings), embedding_size)
    search_module.build_index(database)

    # Search for the top-k nearest neighbors
    D, nns_all = search_module.search_nns(queries, topk)

    return D, nns_all


def prepare_predictions_dataframe(q_annotations, v_annotations, val_embeddings, nns_all, D,
                                  label_csv_col, topk=1):
    """Prepare predictions dataframe with all necessary columns."""
    q_annotations['original_img_path'] = q_annotations['img_path']
    q_annotations['true_artist'] = q_annotations['prompt_label']

    v_prompt_labels = np.asarray(v_annotations[label_csv_col].tolist())

    if topk == 1:
        # Single artist prediction case
        preds = v_prompt_labels[nns_all.flatten()].reshape(
            len(q_annotations), 1)
        q_annotations['pred_artist'] = preds[:, 0]
        q_annotations['pred_img_path'] = np.asarray(
            [val_embeddings.filenames[idx] for idx in nns_all[:, 0]])
        q_annotations['pred_img_path'] = [
            str(x) for x in q_annotations['pred_img_path']]
        q_annotations['pred_artist_score'] = D[:, 0]
    else:
        # Multi artist prediction case
        nns_all_pred_topk = nns_all[:, :topk]
        for k in range(topk):
            q_annotations[f'{k}_pred_artist'] = v_prompt_labels[nns_all_pred_topk.flatten(
            )].reshape(len(q_annotations), topk)[:, k]
            q_annotations[f'{k}_pred_img_path'] = np.asarray(
                [val_embeddings.filenames[col] for col in nns_all_pred_topk[:, k]])
            q_annotations[f'{k}_pred_img_path'] = [
                str(x) for x in q_annotations[f'{k}_pred_img_path']]
            q_annotations[f'{k}_pred_artist_score'] = D[:, k]

    q_annotations['true_source'] = q_annotations['source_label']
    q_annotations['true_prompt_type'] = q_annotations['prompt_type']

    # Drop unnecessary columns
    q_annotations = q_annotations.drop(
        ['source_label', 'prompt_type', 'prompt_label', 'img_path', 'index', 'filename'], axis=1)

    return q_annotations
