#!/usr/bin/bash

cuda_id=0
models=("siren" "asmr" "ngp" "kilonerf")
uvg_names=("readysteadygo" "honeybee" "shakendry" "yachtride" "jockey" "bosphorus" "beauty")

# Train on UVG dataset
for i in {0..6};
do
    for seed in {43..45};
    do
        for m in {0..3};
        do
            python train.py -cn train_video \
            TRAIN_CONFIGS.out_dir="${models[m]}_zero2one_${uvg_names[i]}_${seed}" \
            TRAIN_CONFIGS.device="cuda:${cuda_id}" \
            TRAIN_CONFIGS.seed="${seed}" \
            DATASET_CONFIGS.file_path="../taco-wacv/data/video/${uvg_names[i]}" \
            model_config="${models[m]}"
        done
    done
done
