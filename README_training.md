## Model Training
### 1. Download the dataset
Follow the instructions in the main [README.md](README.md#download-the-dataset) to download the dataset.

### 2. Configure environment
Initialize the [`accelerate`](https://huggingface.co/docs/accelerate/en/basic_tutorials/install#configuration) environment according to your compute setup: <br>
```bash
accelerate config
```

To monitor training progress with [Weights & Biases](https://docs.wandb.ai/quickstart/), login to your W&B account: <br>
```bash
wandb login
```

### 3. Download prototype embeddings
If you are training the prototypical network, you will need to use a prototype embedding. The prototype embedding is a `.npy` file of a CLIP-ViT-L/14 embedding with dimensions `(num_classes, 1024)` containing the average image embedding of each class. <br>
To download the prototype embeddings computed from our LAION subset, run:
```bash
python scripts/download_prototypes.py
```
The prototype embeddings will be saved to the `dataset_laion/` and `dataset_laion_for_journeydb/` directories. <br>

#### Optional: Compute your own prototype embedding
To compute your own prototype embedding, run a command similar to the following: <br>
```bash
python utils/compute_prototype.py --data_dir dataset_laion \
    --csv_name test_source_real_artist_imgs.csv \
    --artist_mapping_file test_artists.txt \
    --batch_size 1024 \
    --workers 12 \
    --model_arch ViT-L/14
```
The above command will compute the prototype embedding for each class in `test_artists.txt` by using the `test_source_real_artist_imgs` split of the `dataset_laion` directory. <br>
The prototype embedding will be saved to the `dataset_laion/prototype-test_source_real_artist_imgs-clip_vit_large-oneprocess.npy` file.

### 4. Run training command
Below, we provide an example command for each training script. <br>
#### Prototypical network: single-label classification
To train the prototypical network for single-label classification, run a command similar to the following: <br>
```bash
DATASET=dataset_all_sample_sdxl
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 train_single_artist_prototypical.py \
    --tracker_project_name prototypical_network_trained_on_${DATASET} \
    --run_name prototypical_network_trained_on_${DATASET} \
    --image_prep clip_base_randomresizedcrop_hflip_blurplusjpeg0.1 \
    --model_arch ViT-L/14 \
    --prototype_path dataset_laion/prototype-train_imgs-clip_vit_large-oneprocess.npy \
    --temperature 0.07 \
    --dataset_info_dir ${DATASET} \
    --use_weighted_sampler \
    --dataloader_num_workers 12 \
    --test_dataset_split test_artist \
    --learning_rate 1e-6 \
    --train_batch_size 128 \
    --log_every_n_epochs 1 \
    --checkpoint_every_n_epochs 1 \
    --num_training_epochs 1 
```
The above command will save model checkpoints to the `logs/prototypical_network_trained_on_dataset_all_sample_sdxl/` directory. <br>
Please see [train_single_artist_prototypical.py](train_single_artist_prototypical.py) `parse_args()` for more details.

#### Vanilla classifier: single-label classification
To train the vanilla classifier for single-label classification, run a command similar to the following: <br>
```bash
DATASET=dataset_all_sample_sdxl
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 train_single_artist_vanilla_classifier.py \
    --tracker_project_name vanilla_classifier_trained_on_${DATASET} \
    --run_name vanilla_classifier_trained_on_${DATASET} \
    --image_prep clip_base_randomresizedcrop_hflip_blurplusjpeg0.1 \
    --model_arch ViT-L/14 \
    --dataset_info_dir ${DATASET} \
    --use_weighted_sampler \
    --dataloader_num_workers 12 \
    --test_dataset_split test_artist \
    --learning_rate 1e-6 \
    --train_batch_size 128 \
    --log_every_n_epochs 1 \
    --checkpoint_every_n_epochs 1 \
    --num_training_epochs 1 
```
The above command will save model checkpoints to the `logs/vanilla_classifier_trained_on_dataset_all_sample_sdxl/` directory. <br>
Please see [train_single_artist_vanilla_classifier.py](train_single_artist_vanilla_classifier.py) `parse_args()` for more details.

#### Prototypical network: multi-label classification
To train the prototypical network with a multi-label classification objective, run a command similar to the following: <br>
```bash
DATASET=dataset_all_multi_artist
export NCCL_P2P_DISABLE=1
accelerate launch --main_process_port 29501 train_multi_artist_prototypical.py \
    --tracker_project_name multi_artist_prototypical_trained_on_${DATASET} \
    --run_name multi_artist_prototypical_trained_on_${DATASET} \
    --image_prep clip_base_randomresizedcrop_hflip_blurplusjpeg0.1 \
    --model_arch ViT-L/14 \
    --prototype_path dataset_laion/prototype-train_imgs-clip_vit_large-oneprocess.npy \
    --temperature 0.07 \
    --dataset_info_dir ${DATASET} \
    --use_weighted_sampler \
    --dataloader_num_workers 12 \
    --test_dataset_split test_artist \
    --learning_rate 1e-6 \
    --train_batch_size 128 \
    --log_every_n_epochs 1 \
    --checkpoint_every_n_epochs 1 \
    --num_training_epochs 1 
```
The above command will save model checkpoints to the `logs/multi_artist_prototypical_trained_on_dataset_all_multi_artist/` directory. <br>
It uses the same `parse_args()` as the single-label prototypical network training command -- please see [train_single_artist_prototypical.py](train_single_artist_prototypical.py) `parse_args()` for more details.