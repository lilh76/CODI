#!/bin/bash

# ========== Configurable Parameters ========== #
# Environment settings
ENV_KEY="mpe:SimpleSpread-v0"         # Environment name
TIME_LIMIT=25                         # Time limit
# Training settings
T_MAX=50000                           # Maximum training steps
INTERVAL=2000                         # Logging interval

# Basic settings
CUDA_DEVICE=0
SEEDS=(1)
BUFFER_SIZES=(20000)
ALGS=("bc" "bc_percent" "omiga" "omar")

DATA_QUALITY="imb2_20k"               # Data quality

OFFLINE_DATA_PATH=''

SYN_DATASET=''

SYN_MAX=20000
COND_RET=0

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Loop through different seeds
for SEED in "${SEEDS[@]}"
do
    for OFFLINE_MAX_BUFFER in "${BUFFER_SIZES[@]}"
    do
        for ALG in "${ALGS[@]}"
        do
            NAME="${ALG}_real_${OFFLINE_MAX_BUFFER}_syn_${SYN_MAX}_ret_${COND_RET}"

            # Build base command
            CMD="python src/main.py --offline --config=$ALG --env-config=gymma_offline \
            --key=\"$ENV_KEY\" --time_limit=$TIME_LIMIT \
            --offline_data_quality=$DATA_QUALITY \
            --t_max=$T_MAX --test_interval=$INTERVAL --log_interval=$INTERVAL \
            --runner_log_interval=$INTERVAL --learner_log_interval=$INTERVAL \
            --save_model_interval=$INTERVAL \
            --seed=$SEED \
            --offline_bottom_data_path=\"$OFFLINE_DATA_PATH\" \
            --offline_max_buffer_size=$OFFLINE_MAX_BUFFER"

            # Add synthetic data parameters (if enabled)
            CMD="$CMD --syn_dataset=\"$SYN_DATASET\" --syn_max=$SYN_MAX"

            # Add experiment name
            CMD="$CMD --name=\"$NAME\""

            sleep 2

            # Execute command in background
            eval $CMD &
        done
    done
done

# Wait for all background processes to complete
wait
echo "All tasks completed"