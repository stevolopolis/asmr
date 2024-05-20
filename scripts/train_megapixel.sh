#!/usr/bin/bash

cuda_id=0

models=("siren" "asmr" "ngp" "kilonerf")


# Train on Pluto megapixel image
for seed in {43..45};
do
    for m in {0..3};
    do
        if [[ "${models[m]}" == "asmr" ]]; then
            max_coords=262144
        elif [[ "${models[m]}" == "kilonerf" ]]; then
            max_coords=4194304
        elif [[ "${models[m]}" == "ngp" ]]; then
            max_coords=262144
        else
            max_coords=262144
        fi
        python train.py -cn train_megapixel \
        TRAIN_CONFIGS.out_dir="${models[m]}_exp_${seed}" \
        TRAIN_CONFIGS.seed="${seed}" \
        DATASET_CONFIGS.max_coords="${max_coords}" \
        model_config="${models[m]}" \
        TRAIN_CONFIGS.device="cuda:${cuda_id}"
    done
done