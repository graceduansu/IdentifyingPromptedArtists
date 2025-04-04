import argparse


def parse_eval_args(input_args=None):
    """
    Shared argument parser for evaluation scripts to avoid circular imports.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--evaluate_multi_artist_preds", action="store_true",
                        help="Use this flag to evaluate the vanilla classifier on multi-artist classification.")

    # Model
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="The path to the model checkpoint")
    parser.add_argument("--model_type", type=str, default="vanilla_classifier",
                        choices=["vanilla_classifier", "prototypical_network"])

    # If ProtoClip: prototype path
    parser.add_argument("--prototype_path", type=str, default=None)
    parser.add_argument("--model_arch", type=str, default="ViT-L/14")

    # dataset options
    parser.add_argument("--dataset_info_dir",
                        default="dataset_sdxl_simple", type=str)
    parser.add_argument("--label_csv_col", default="prompt_label", type=str)
    parser.add_argument("--eval_split", type=str, default="test_artist",
                        choices=["test_artist", "test_all_unseen_query"],
                        help="The name of the split to evaluate on.")
    parser.add_argument("--artist_mapping_file", type=str, default="map_prompt_label.txt",
                        help="The name of the file under dataset_info_dir containing the artists to classifiy. If none, will automatically choose based on the test_dataset_split.")
    parser.add_argument("--image_prep", default="clip_base_noaug",
                        type=str, help='Image preprocessing')


    # dataloader options
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=8,)

    # save options
    parser.add_argument("--save_dir", type=str, default="RESULTS",
                        help="Parent directory to save embeddings, metrics, and predictions. The model ID will be appended to this directory.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="The name of the directory under embed_dir to save this evaluation run's results")

    return parser.parse_args(input_args) 
