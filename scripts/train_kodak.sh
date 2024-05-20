#!/usr/bin/bash

seed=45
cuda_id=2

models=("siren" "asmr" "ngp" "ffn" "wire")
imgs=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")


# Train on Pluto megapixel image
for seed in {43..45};
do
    for img_idx in {0..23};
    do
        for m in {0..4};
        do
            python train.py -cn train_image \
            DATASET_CONFIGS.file_path="../datasets/kodak/kodim${imgs[img_idx]}.png" \
            TRAIN_CONFIGS.seed="${seed}" \
            model_config="${models[m]}" \
            TRAIN_CONFIGS.device="cuda:${cuda_id}" \
            TRAIN_CONFIGS.out_dir="${models[m]}_kodak${imgs[img_idx]}_exp" 
        done
    done
done
