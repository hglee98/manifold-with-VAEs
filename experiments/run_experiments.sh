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

python3 run.py --model $MODEL --dataset $DATASET --enc_layers 64 128 --dec_layers 128 64 --latent_dim $L_DIM --num_centers 32 --num_components $N_COMP
