#! /usr/bin/env bash

PROMPT='close up of a cute girl sitting in the flower garden, clear anime face, insanely frilled white dress, absurdly long brown hair, small silver tiara, long sleeves highneck dress'
NEGATIVE_PROMPT='low quality, worst quality, maid, sleeveless'

b=32
lr="1e-6"

for t in IN01 IN02 IN04 IN05 IN07 IN08 M00 OUT03 OUT04 OUT05 OUT06 OUT07 OUT08 OUT09 OUT10 OUT11
do

HYENAS=''
for s in 1000 2000 3000 4000 5000; do
    HYENAS="${HYENAS},out/1/${t}/${b}_${lr}/${t}_${b}_${lr}_$(printf '%05d' $s).safetensors"
done

python generate.py \
    --model='../models/evbp-anime-half.safetensors' \
    --target="${t}" \
    --hyena="${HYENAS}" \
    --prompt="${PROMPT}" \
    --negative_prompt="${NEGATIVE_PROMPT}" \
    --width=512 \
    --height=512 \
    --num_images=8 \
    --steps=30 \
    --cfg_scale='3.0' \
    --iteration=1 \
    --seed=1 \
    --out_dir="images/1" \
    --name_format="${t}_${b}_${lr}_{h:03d}_{n:03d}"

done
