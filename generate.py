import torch

from utils import init_seed, load_pipeline, replace_hyena, ATTN_MAP


def generate(iteration: int, pipe, gen_args: dict, seed: int):
    init_seed(seed)
    
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

    pipe = load_pipeline(args.model)
    pipe = pipe.to('cuda')
    
    hyenas = args.hyena.split(',')
    if len(hyenas) == 0 or (len(hyenas) == 1 and hyenas[0] == ''):
        hyenas = [None]
    
    for k, hyena_path in enumerate(hyenas):
        if hyena_path is not None:
            hyena_path = hyena_path.strip()
            if len(hyena_path) == 0:
                continue
            replace_hyena(args.target, pipe.unet, hyena_path)
        
        images = generate(args.iteration, pipe, gen_args, args.seed)
        
        for i, imgs in enumerate(images):
            for j, image in enumerate(imgs):
                n = i * args.num_images + j
                name = args.name_format.format(n=n, b=i, i=j, h=k)
                path = f'{args.out_dir}/{name}.png'
                image.save(path)
