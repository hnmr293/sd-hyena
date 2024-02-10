import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import safetensors.torch

from mods import HyenaProcessor


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

def replace_hyena(unet, hyena_path: str):
    hyena = HyenaProcessor(320, 64*64).half()
    safetensors.torch.load_model(hyena, hyena_path)
    hyena = hyena.to('cuda')
    
    IN01 = unet.down_blocks[0].attentions[0].transformer_blocks[0]
    IN01.attn1.processor = hyena


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
    
    def hyena_set():
        for b in [2, 4, 8, 16]:
            for lr in ['1e-1', '1e-2', '1e-3', '1e-4']:
                for e in [10, 20, 30, 40, 50]:
                    yield b, lr, e
    
    for b, lr, e in hyena_set():
        hyena_path = f'out/out_{b}_{lr}/IN01_{b}_{lr}_{e:03d}.safetensors'
        replace_hyena(pipe.unet, hyena_path)
        
        images = generate(args.iteration, pipe, gen_args, args.seed)
    
        for i, imgs in enumerate(images):
            for j, image in enumerate(imgs):
                n = i * args.num_images + j
                #name = args.name_format.format(n=n, b=i, i=j)
                path = f'{args.out_dir}/{b}_{lr}_{e:03d}_{n}.png'
                image.save(path)
