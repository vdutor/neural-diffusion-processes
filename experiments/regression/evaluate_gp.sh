#!/bin/bash

datasets=("se" "matern")
dim_x_values=(1 2 3)
fullcov_values=("True" "False")

for fullcov in "${fullcov_values[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for dim_x in "${dim_x_values[@]}"
        do
                if [[ "$fullcov" == "True" ]]; then
                    CUDA_VISIBLE_DEVICES="" python eval_gp.py --dim-x="$dim_x" --dataset="$dataset" --fullcov
                else
                    CUDA_VISIBLE_DEVICES="" python eval_gp.py --dim-x="$dim_x" --dataset="$dataset" --no-fullcov
                fi
        done
    done
done