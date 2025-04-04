import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_prompt_info_from_img_path(img_path):
    """
    Returns prompt_num, prompt_text
    ## example: sdxl_images_sb_prompts-single_artist/larry_poons_seed1/larry_poons_prompt1_seed1.png

    """
    if 'mj_prompts' in img_path or "multi_artist" in img_path:
        prompt_num = int(img_path.split(
            "/")[-1].split("_prompt")[1].split("_seed")[0])
    elif 'easy_prompts' in os.path.dirname(img_path):
        prompt_num = img_path.split("a_picture_of_")[
            1].split("_in_the_style_of")[0]
    else:
        prompt_num = None

    return prompt_num


def vectorized_bootstrap(df, num_iterations=1000, random_state=None,
                         artist_col='true_artist', target_col='artist_correct',
                         block_by_prompt=True, block_by_seed=True):
    """
    Bootstrapping function to calculate the mean and standard error of a target column.
    Can block by artist, prompt, and seed.
    """
    rng = np.random.default_rng(random_state)

    # Preprocess the dataframe: create required columns
    if 'pred_prompt_label' in df.columns:
        df['pred_artist'] = df['pred_prompt_label']
    df['prompt_num'] = df['original_img_path'].apply(
        get_prompt_info_from_img_path)

    df['seed'] = df['original_img_path'].apply(lambda x: int(
        os.path.basename(x).split("_seed")[1].split(".")[0]))
    df['artist_correct'] = (df['pred_artist'] == df['true_artist']).astype(int)

    # Get unique values and number of unique blocks for each grouping variable.
    unique_artists = df[artist_col].unique()
    unique_prompts = df['prompt_num'].unique()
    unique_seeds = df['seed'].unique()

    n_artists = len(unique_artists)
    n_prompts = len(unique_prompts)
    n_seeds = len(unique_seeds)

    # Create mappings from block value to index
    artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
    prompt_to_idx = {prompt: idx for idx, prompt in enumerate(unique_prompts)}
    seed_to_idx = {seed: idx for idx, seed in enumerate(unique_seeds)}

    # Map each observation to its block index
    artist_indices = df[artist_col].map(artist_to_idx).to_numpy()
    prompt_indices = df['prompt_num'].map(prompt_to_idx).to_numpy()
    seed_indices = df['seed'].map(seed_to_idx).to_numpy()
    # print(artist_indices.shape)
    # print(np.unique(artist_indices))

    # Instead of iterating, use np.random.multinomial to get counts for each block in all iterations
    # For each iteration, we sample exactly n_artists times from artists, and similarly for prompts and seeds.
    artist_counts = rng.multinomial(n=n_artists, pvals=np.ones(
        n_artists) / n_artists, size=num_iterations)
    prompt_counts = rng.multinomial(n=n_prompts, pvals=np.ones(
        n_prompts) / n_prompts, size=num_iterations)
    seed_counts = rng.multinomial(n=n_seeds, pvals=np.ones(
        n_seeds) / n_seeds, size=num_iterations)

    # For each iteration and for each observation, the weight is the product of the counts from each block.
    # Using advanced indexing: for each iteration, select the count corresponding to the block the observation belongs to.
    # The resulting weights array has shape (num_iterations, number_of_observations)
    weights = (artist_counts[:, artist_indices] *
               prompt_counts[:, prompt_indices] *
               seed_counts[:, seed_indices])

    # Get the value we are interested in (artist_correct, which is 1 or 0)
    x = df[target_col].to_numpy()

    # Compute the weighted mean for each iteration.
    # Each iteration's mean is sum(weights * x) / sum(weights)
    bootstrap_means = np.sum(weights * x, axis=1) / np.sum(weights, axis=1)

    bootstrap_std_err = np.std(bootstrap_means)
    return bootstrap_means, bootstrap_std_err


def run_bootstrap(l_dataset_dirs,
                  results_dir="RESULTS",
                  random_state=42,
                  is_multi_artist=False):
    """
    Runner function that processes predictions from both single and multi-artist datasets.

    Args:
        l_model_ids: List of model IDs to process
        l_dataset_dirs: List of dataset directories to process
        results_dir: Directory containing results
        random_state: Random seed for reproducibility
        is_multi_artist: If True, process as multi-artist datasets (uses different columns)
    """
    np.random.seed(random_state)
    csv_cols = ["model_id",
                "Seen artists: Complex",
                "Seen artists: Simple",
                "Held-out artists: Complex",
                "Held-out artists: Simple",]

    # Process datasets
    for dataset_dir in tqdm(l_dataset_dirs):
        new_df = pd.DataFrame(columns=csv_cols)

        glob_pattern = f"{results_dir}/**/{dataset_dir}/**/*preds*.csv"
        l_csv_files = glob(glob_pattern)
        l_csv_files.sort(reverse=False)

        for csv_file in l_csv_files:
            split_dir = csv_file.split("/")[-2]
            # Extract model_id from path (assumes results_dir/model_id/...)
            model_id = csv_file.split(
                "/")[1] if csv_file.startswith(results_dir + "/") else csv_file.split("/")[0]

            # Determine if dataset is simple based on dataset name
            is_simple = "simple" in dataset_dir

            if split_dir == "test_artist":
                col_name = "Seen artists: "
            else:
                col_name = "Held-out artists: "

            if is_simple:
                col_name += "Simple"
            else:
                col_name += "Complex"

            print('--'*20)
            print(f"Processing {csv_file}")
            print(model_id, dataset_dir, col_name,
                  f"is_simple: {is_simple}")

            df = pd.read_csv(csv_file)

            # Handle multi-artist specific checks
            if is_multi_artist:
                if 'row_ranked_precision' not in df.columns:
                    print("row_ranked_precision not found in ", csv_file)
                    continue

                # Use vectorized_bootstrap for multi-artist datasets
                # For multi-artist, we use 'all_prompt_labels' as artist_col and 'row_ranked_precision' as target_col
                means, final_std_err = vectorized_bootstrap(df, num_iterations=2000, random_state=random_state,
                                                            artist_col='all_prompt_labels', target_col='row_ranked_precision',
                                                            block_by_prompt=True, block_by_seed=True)

                original_mean = np.mean(df['row_ranked_precision']) * 100
                print(f"Original mean: {original_mean:.1f} %")
            else:
                # Use vectorized_bootstrap for regular datasets
                if dataset_dir == "dataset_journeydb":
                    print("Handling journeydb")
                    # For journeydb, only block by artist since every prompt and seed is unique
                    means, final_std_err = vectorized_bootstrap(df, num_iterations=2000, random_state=random_state,
                                                                artist_col='true_artist', target_col='artist_correct',
                                                                block_by_prompt=False, block_by_seed=False)
                else:
                    means, final_std_err = vectorized_bootstrap(df, num_iterations=2000, random_state=random_state,
                                                                artist_col='true_artist', target_col='artist_correct',
                                                                block_by_prompt=True, block_by_seed=True)

            final_mean = np.mean(means) * 100
            final_std_err *= 100
            print(f"Final mean: {final_mean:.1f} ± {final_std_err:.1f}%")

            # Create new row with appropriate columns
            new_row = {
                "model_id": model_id,
                col_name: f"{final_mean:.1f} ± {final_std_err:.1f}%",
            }

            # Check if model_id already exists
            if len(new_df[(new_df['model_id'] == model_id)]) > 0:
                # update the row
                new_df.loc[(new_df['model_id'] == model_id),
                           col_name] = new_row[col_name]
            else:
                # add a new row
                new_df = pd.concat(
                    [new_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the final results to a CSV file
        print(new_df)
        output_csv = f"{results_dir}/bootstrap_results-{dataset_dir}.csv"
        print(dataset_dir, output_csv)
        new_df.to_csv(output_csv, index=False)


def main():
    # Process single-artist classification results
    l_dataset_dirs = [
        "dataset_sdxl_complex",
        "dataset_sd15_complex",
        "dataset_pixart_complex",
        "dataset_journeydb",
        "dataset_sdxl_simple",
        "dataset_sd15_simple",
        "dataset_pixart_simple",
    ]
    run_bootstrap(l_dataset_dirs, is_multi_artist=False)

    # Process multi-artist classification results
    l_dataset_dirs = [
        "dataset_2artist_sdxl_complex",
        "dataset_2artist_sdxl_simple",
        "dataset_3artist_sdxl_complex",
        "dataset_3artist_sdxl_simple",
    ]

    run_bootstrap(l_dataset_dirs, is_multi_artist=True)


if __name__ == "__main__":
    main()
