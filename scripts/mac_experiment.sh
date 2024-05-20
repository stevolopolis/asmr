#!/usr/bin/bash

cuda_id=0
seed=42
models=("kilonerf" "asmr" "loe")
loe_dims=("[[4,4],[4,4],[4,4],[2,2],[2,2],[2,2]]" \
            "[[4,4],[4,4],[4,4],[4,4],[2,2]]" \
            "[[8,8],[4,4],[4,4],[4,4]]" \
            "[[8,8], [8,8], [8,8]]" \
            )
asmr_dims=('[[4,2,2,2,2,8], [4,2,2,2,2,8]]' \
            '[[8,2,2,2,8], [2,2,4,4,8]]' \
            '[[4,4,4,8], [4,4,4,8]]' \
            '[[8,8,8], [8,8,8]]'
            )
kilo_dims=("[16,16]" \
            "[16,16]" \
            "[16,16]" \
            "[16,16]"
            )

# Train on Kodak dataset
for model_idx in {1..1};
do
    if [[ "${models[model_idx]}" == "asmr" ]]; then
        dims=${asmr_dims}
        dim_hidden=256
    elif [[ "${models[model_idx]}" == "loe" ]]; then
        dims=${loe_dims}
        dim_hidden=24
    elif [[ "${models[model_idx]}" == "kilonerf" ]]; then
        dims=${kilo_dims}
        dim_hidden=16
    else
        echo "Model not found"
    fi

    for i in {0..3};
    do
        num_layers=$(( 6 - i ))
        echo "${dims[i]}"
        python train.py -cn train_low_mac \
        TRAIN_CONFIGS.out_dir="${models[model_idx]}${num_layers}" \
        TRAIN_CONFIGS.seed="${seed}" \
        model_config="${models[model_idx]}" \
        TRAIN_CONFIGS.device="cuda:${cuda_id}" \
        ++TRAIN_CONFIGS.dim_hidden="${dim_hidden}" \
        ++TRAIN_CONFIGS.num_layers="${num_layers}" \
        ++TRAIN_CONFIGS.dimensions="${dims[i]}"
    done
done