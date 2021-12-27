#!/bin/bash

while getopts m:d:l:c: option
do
    case "$option" in
        m) MODEL=${OPTARG};;
        d) DATASET=${OPTARG};;
        l) L_DIM=${OPTARG};;
        c) N_COMP=${OPTARG};;
    esac
done

if [ $MODEL = "RVAE" ]; then
    $N_COMP = 1
fi

python3 run.py --model $MODEL --dataset $DATASET --enc_layers 32 64 --dec_layers 64 32 --latent_dim $L_DIM --num_centers 12 --num_components $N_COMP
