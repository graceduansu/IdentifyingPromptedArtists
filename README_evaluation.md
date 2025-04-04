
## Model Evaluation on Benchmark
Below, we provide an example command for each evaluation script. 
### Evaluating retrieval-based methods (single-artist)
To evaluate one of the retrieval-based methods on single-artist classification, run a command similar to the following:
```bash
python eval_single_artist_retrieval.py --num_gpus 4 --workers 12 --batch_size 1024 \
    --data_dir dataset_sdxl_simple \
    --eval_splits test_all_unseen_query,test_artist \
    --what_to_retrieve sdxl,sd15,pixart,journeydb \
    --train_splits test_all_unseen_support,train \
    --model_type clip_vit_large \
    --embed_dir RESULTS \
    --model_id clip_vit_large
```
The above command will evaluate CLIP ViT-L/14 on the `dataset_sdxl_simple` subset, on both the `test_all_unseen_query` and `test_artist` splits. It will retrieve embeddings from all single-artist prompted images in the dataset (from SDXL, SD15, PixArt, and JourneyDB). <br>
It will extract the embeddings and predictions to `RESULTS/clip_vit_large/nn_embeddings/dataset_sdxl_simple/`, and the evaluation metrics to `RESULTS/metrics_on_dataset_sdxl_simple.csv`. <br>
Please see [eval_single_artist_retrieval.py](eval_single_artist_retrieval.py) `parse_args()` for more details.

### Evaluating retrieval-based methods (multi-artist)
To evaluate one of the retrieval-based methods on multi-artist classification, run a command similar to the following:
```bash
python eval_multi_artist_retrieval.py --num_gpus 4 --workers 12 --batch_size 1024 \
    --data_dir dataset_2artist_sdxl_simple \
    --eval_splits test_all_unseen_query,test_artist \
    --what_to_retrieve sdxl \
    --train_splits test_all_unseen_support,train \
    --model_type clip_vit_large \
    --embed_dir RESULTS \
    --model_id clip_vit_large 
```
The above command will evaluate CLIP ViT-L/14 on the `dataset_2artist_sdxl_simple` subset, on both the `test_all_unseen_query` and `test_artist` splits. It will retrieve embeddings from all single-artist prompted SDXL images. <br>
It will extract the embeddings and predictions to `RESULTS/clip_vit_large/nn_embeddings/dataset_2artist_sdxl_simple/`, and the evaluation metrics to `RESULTS/metrics_on_dataset_2artist_sdxl_simple.csv`. <br>
Please see [eval_multi_artist_retrieval.py](eval_multi_artist_retrieval.py) `parse_args()` for more details.

### Evaluating classification-based methods (single-artist)
To evaluate the classification-based methods on single-artist classification, run a command similar to the following:
```bash
python eval_single_artist_classifiers.py --dataloader_num_workers 12 --batch_size 1024 \
    --dataset_info_dir dataset_sdxl_simple \
    --eval_split test_all_unseen_query \
    --artist_mapping_file test_artists.txt \
    --model_type prototypical_network \
    --ckpt_path trained_models/protoclip_trained_on_dataset_all_single_artist/ckpt/epoch0/model.safetensors \
    --prototype_path dataset_laion/prototype-test_source_real_artist_imgs-clip_vit_large-oneprocess.npy \
    --save_dir RESULTS \
    --model_id protoclip_trained_on_dataset_all_single_artist 
```
The above command will evaluate the prototypical network on the `dataset_sdxl_simple` subset, on the `test_all_unseen_query` split. <br>
It will save the predictions to `RESULTS/protoclip_trained_on_dataset_all_single_artist/dataset_sdxl_simple/test_all_unseen_query/preds.csv`, and the evaluation metrics to `RESULTS/metrics_on_dataset_sdxl_simple.csv`. <br>
Please see [eval_single_artist_classifiers.py](eval_single_artist_classifiers.py) `parse_args()` for more details.

### Evaluating classification-based methods (multi-artist)
To evaluate the prototypical network on multi-artist classification, run a command similar to the following:
```bash
python eval_multi_artist_prototypical.py --dataloader_num_workers 12 --batch_size 1024 \
    --dataset_info_dir dataset_2artist_sdxl_simple \
    --eval_split test_all_unseen_query \
    --artist_mapping_file test_artists.txt \
    --model_type prototypical_network \
    --ckpt_path trained_models/protoclip_trained_on_dataset_all_multi_artist/ckpt/epoch0/model.safetensors \
    --prototype_path dataset_laion/prototype-test_source_real_artist_imgs-clip_vit_large-oneprocess.npy \
    --save_dir RESULTS \
    --model_id protoclip_trained_on_dataset_all_multi_artist 
```
The above command will evaluate the prototypical network on the `dataset_2artist_sdxl_simple` subset, on the `test_all_unseen_query` split. <br>
It will save the predictions to `RESULTS/protoclip_trained_on_dataset_all_multi_artist/dataset_2artist_sdxl_simple/test_all_unseen_query/preds.csv`, and the evaluation metrics to `RESULTS/metrics_on_dataset_2artist_sdxl_simple.csv`. 

To evaluate the vanilla classifier on multi-artist classification, run a command similar to the following:
```bash
python eval_single_artist_classifiers.py --dataloader_num_workers 12 --batch_size 1024 \
    --evaluate_multi_artist_preds \
    --dataset_info_dir dataset_2artist_sdxl_simple \
    --eval_split test_artist \
    --artist_mapping_file map_prompt_label.txt \
    --model_type vanilla_classifier \
    --ckpt_path trained_models/clipclassifier_trained_on_dataset_all_single_artist/ckpt/epoch0/model.safetensors \
    --save_dir RESULTS \
    --model_id clipclassifier_trained_on_dataset_all_single_artist 
```
The above command will evaluate the vanilla classifier on the `dataset_2artist_sdxl_simple` subset, on the `test_artist` split. <br>
It will save the predictions to `RESULTS/clipclassifier_trained_on_dataset_all_single_artist/dataset_2artist_sdxl_simple/test_artist/preds.csv`, and the evaluation metrics to `RESULTS/metrics_on_dataset_2artist_sdxl_simple.csv`. 

### Bootstrapping
To estimate the statistical significance of the results, we provide the bootstrapping script used to obtain the results in the paper.
Run the desired evaluation script first to get model predictions so that the boostrapping script can resample the predictions.
Then, run the bootstrapping script to process all predictions saved under `RESULTS/`:
```bash
python bootstrap_preds.py 
```
Edit `main()` in [bootstrap_preds.py](bootstrap_preds.py) to specify which dataset directories to process.
