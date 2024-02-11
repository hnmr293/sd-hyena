#! /usr/bin/env bash

b=4
r="1e-6"
#t=M00
R=2048

for t in IN01 IN02 IN04 IN05 IN07 IN08 M00 OUT03 OUT04 OUT05 OUT06 OUT07 OUT08 OUT09 OUT10 OUT11
do

python train_module.py \
    --teacher_model='../models/evbp-anime-half.safetensors' \
    --target="${t}" \
    --batch_size="${b}" \
    --lr="${r}" \
    --n_steps=5000 \
    --width="${R}" \
    --height="${R}" \
    --save_every_n_steps=1000 \
    --log_dir="logs/${R}/${b}/${t}" \
    --out_dir="out/${R}/${t}/${b}_${r}" \
    --name_format="${t}_${b}_${r}_{s:05d}"

done
