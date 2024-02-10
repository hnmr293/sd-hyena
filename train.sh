#! /usr/bin/env bash

t='IN04'
b=32
r="1e-6"
python train.py \
    --teacher_model='../models/evbp-anime-half.safetensors' \
    --target="${t}" \
    --batch_size="${b}" \
    --lr="${r}" \
    --n_steps=5000 \
    --channels=640 \
    --width=32 \
    --height=32 \
    --save_every_n_steps=1000 \
    --log_dir="logs/${t}" \
    --out_dir="out/${t}/${b}_${r}" \
    --name_format="${t}_${b}_${r}_{s:05d}"
