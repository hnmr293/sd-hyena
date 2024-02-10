#! /usr/bin/env bash

PROMPT='close up of a cute girl sitting in the flower garden, clear anime face, insanely frilled white dress, absurdly long brown hair, small silver tiara, long sleeves highneck dress'
NEGATIVE_PROMPT='low quality, worst quality, maid, sleeveless'

python generate.py \
    --model='../models/evbp-anime-half.safetensors' \
    --hyena='out/out_2_1e-1/IN01_2_1e-1_050.safetensors' \
    --prompt="${PROMPT}" \
    --negative_prompt="${NEGATIVE_PROMPT}" \
    --width=512 \
    --height=512 \
    --num_images=8 \
    --steps=30 \
    --cfg_scale='3.0' \
    --iteration=1 \
    --seed=1 \
    --out_dir='images/hyena' \
    --name_format='image_{n:03d}'
