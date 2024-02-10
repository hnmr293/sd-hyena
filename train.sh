#! /usr/bin/env bash

bs=(2 4 8 16)
lr=('1e-1' '1e-2' '1e-3' '1e-4')

for b in "${bs[@]}"
do
    for r in "${lr[@]}"
    do

python train.py \
    --batch_size="${b}" \
    --lr="${r}" \
    --n_epochs=50 \
    --channels=320 \
    --width=64 \
    --height=64 \
    --save_every_n_epochs=10 \
    --log_dir='logs' \
    --out_dir="out/out_${b}_${r}" \
    --name_format="IN01_${b}_${r}_{e:03d}"

    done
done
