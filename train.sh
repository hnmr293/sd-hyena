#! /usr/bin/env bash

b=32
r="1e-6"
python train2.py \
    --teacher_model='../models/evbp-anime-half.safetensors' \
    --batch_size="${b}" \
    --lr="${r}" \
    --n_steps=5000 \
    --channels=320 \
    --width=64 \
    --height=64 \
    --save_every_n_steps=500 \
    --log_dir='logs/out3' \
    --out_dir="out/out3/${b}_${r}" \
    --name_format="IN01_${b}_${r}_{s:05d}"
