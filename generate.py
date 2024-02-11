import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import safetensors.torch

from mods import HyenaProcessor, ATTN_MAP


def load_model(model_path: str):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        #local_files_only=True,
        cache_dir='.cache',
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def replace_hyena(target: str, unet, hyena_path: str, image_width: int, image_height: int):
    target = target.upper()
    info = ATTN_MAP[target]
    
    b, a, t = info.diffusers_block_index, info.diffusers_attn_index, info.diffusers_transformer_index
    m, d = info.multiplier, info.input_channels
    assert image_width % m == 0
    assert image_height % m == 0
    
    w = image_width // m
    h = image_height // m
    
    hyena = HyenaProcessor(d, h*w).half()
    safetensors.torch.load_model(hyena, hyena_path)
    hyena = hyena.to('cuda')
    
    blocks = None
    if target.startswith('IN'):
        blocks = unet.down_blocks
    elif target.startswith('M'):
        blocks = [unet.mid_block]
    elif target.startswith('OUT'):
        blocks = unet.up_blocks
    else:
        raise ValueError(f'unknown target: {target}')
    
    mod = blocks[b].attentions[a].transformer_blocks[t]
    mod.attn1.processor = hyena


def generate(iteration: int, pipe, gen_args: dict, seed: int):
    generator = torch.Generator(device='cuda')
    if 0 <= seed:
        generator = generator.manual_seed(seed)
    
    result = []
    for _ in range(iteration):
        images, nsfw = pipe(generator=generator, **gen_args)
        result.append(images)

    return result


if __name__ == '__main__':
    import os
    import argparse
    import pprint
    
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', type=str, required=True)
    p.add_argument('-t', '--target', type=str, choices=list(ATTN_MAP.keys()), default='IN01')
    p.add_argument('-y', '--hyena', type=str, default='')
    p.add_argument('-p', '--prompt', type=str, required=True)
    p.add_argument('-q', '--negative_prompt', type=str, default='')
    p.add_argument('-W', '--width', type=int, default=512)
    p.add_argument('-H', '--height', type=int, default=512)
    p.add_argument('-n', '--num_images', type=int, default=4)
    p.add_argument('-s', '--steps', type=int, default=30)
    p.add_argument('-c', '--cfg_scale', type=float, default=7.0)
    p.add_argument('-i', '--iteration', type=int, default=1)
    p.add_argument('-k', '--seed', type=int, default=-1)
    p.add_argument('-o', '--out_dir', type=str, default='./')
    p.add_argument('-f', '--name_format', type=str, default='image_{n:03d}')

    args = p.parse_args()
    
    gen_args = {
        'prompt': args.prompt,
        'width': args.width,
        'height': args.height,
        'num_inference_steps': args.steps,
        'guidance_scale': 0.0,
        'num_images_per_prompt': args.num_images,
        'return_dict': False,
    }

    if 0.0 < args.cfg_scale and len(args.negative_prompt) != 0:
        gen_args['negative_prompt'] = args.negative_prompt
        gen_args['guidance_scale'] = args.cfg_scale

    
    print('[Hyena] Generation Setting')
    pprint.pprint(gen_args)
    
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = load_model(args.model)
    pipe = pipe.to('cuda')
    
    for k, hyena_path in enumerate(args.hyena.split(',')):
        hyena_path = hyena_path.strip()
        if len(hyena_path) == 0:
            continue
        replace_hyena(args.target, pipe.unet, hyena_path, args.width, args.height)
        
        images = generate(args.iteration, pipe, gen_args, args.seed)
        
        for i, imgs in enumerate(images):
            for j, image in enumerate(imgs):
                n = i * args.num_images + j
                name = args.name_format.format(n=n, b=i, i=j, h=k)
                path = f'{args.out_dir}/{name}.png'
                image.save(path)
