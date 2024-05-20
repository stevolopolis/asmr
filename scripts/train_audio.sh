#!/usr/bin/bash

seed=42
cuda_id=0
models=("asmr" "siren")

# Train on LibriSpeech audio
for i in {0..100};
do 
    for m in {0..1};
    do
        python train.py -cn train_audio \
        DATASET_CONFIGS.sample_idx=$i \
        TRAIN_CONFIGS.seed="${seed}" \
        model_config="${models[m]}" \
        TRAIN_CONFIGS.device="cuda:${cuda_id}" \
        TRAIN_CONFIGS.out_dir="${models[m]}_audio${i}_exp" 
    done
done