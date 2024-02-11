import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    '../models/evbp-anime-half.safetensors',
    torch_dtype=torch.float16,
    safety_checker=None,
    cache_dir='.cache',
)

def make_hook(name: str):
    def attn(mod, inputs, output):
        print(name, tuple(inputs[0].shape), tuple(output.shape))
    return attn

for name, mod in pipe.unet.named_modules():
    if 'attn1' not in name:
        continue
    mod.register_forward_hook(make_hook(name))

pipe.to('cuda')
pipe("aaa")

"""
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

mid_block.attentions.0.transformer_blocks.0.attn1.to_q (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_k (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_v (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1 (2, 64, 1280) (2, 64, 1280)

up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
down_blocks.0.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
down_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
down_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

mid_block.attentions.0.transformer_blocks.0.attn1.to_q (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_k (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_v (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 64, 1280) (2, 64, 1280)
mid_block.attentions.0.transformer_blocks.0.attn1 (2, 64, 1280) (2, 64, 1280)

up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 256, 1280) (2, 256, 1280)
up_blocks.1.attentions.2.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)

up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 1024, 640) (2, 1024, 640)
up_blocks.2.attentions.2.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)

up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)

up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.1 (2, 4096, 320) (2, 4096, 320)
up_blocks.3.attentions.2.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
"""
