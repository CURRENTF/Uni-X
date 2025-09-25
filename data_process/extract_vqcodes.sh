#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python data_process/convert_imagepair_cc512_batch_version.py \
        --input_dir ../datasets/long_batch_all  \
        --temp_path ../datasets/uni_data_tai_v1  \
        --vqgan_path data_process/vqgan_ckpts \
        --batch_size 64 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --num_processes 5 &
done
wait


python data_process/packing_imagepairs.py --temp_path ../datasets/uni_data_tai_v1 --save_path ../datasets/uni_data_tai_v1_hf
