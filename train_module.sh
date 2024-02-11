#! /usr/bin/env bash

b=32
r="5e-7"

for t in IN01 IN02
do

python train.py \
    --teacher_model='../models/evbp-anime-half.safetensors' \
    --target="${t}" \
    --batch_size="${b}" \
    --lr="${r}" \
    --n_steps=5000 \
    --width=512 \
    --height=512 \
    --save_every_n_steps=500 \
    --log_dir="logs/2/${t}" \
    --out_dir="out/2/${t}/${b}_${r}" \
    --name_format="${t}_${b}_${r}_{s:05d}"

done
