import dataclasses

import torch
from diffusers import StableDiffusionPipeline

from utils import load_pipeline, replace_hyena, cuda_profiler, ATTN_MAP
from generate import generate


@dataclasses.dataclass
class Env:
    total_steps: int
    current_step: int = 0

@dataclasses.dataclass
class Result:
    name: str
    step: int
    time: float
    memory: int

def hook(name: str, mod: torch.nn.Module, env: Env, results: list[Result]):
    orig_forward = mod.forward
    def forward(*args, **kwargs):
        with cuda_profiler() as prof:
            x = orig_forward(*args, **kwargs)
        
        result = Result(name, env.current_step, prof['time'], prof['memory'])
        results.append(result)
        
        return x
    
    mod.forward = forward

def run(pipe: StableDiffusionPipeline, n_iter: int, gen_args: dict):
    env = Env(gen_args['num_inference_steps'], 0)
    results: list[Result] = []
    
    for key, attn in ATTN_MAP.items():
        if key.startswith('IN'):
            blocks = pipe.unet.down_blocks
        elif key.startswith('M'):
            blocks = [pipe.unet.mid_block]
        elif key.startswith('OUT'):
            blocks = pipe.unet.up_blocks
        else:
            raise ValueError(f'unknown target: {key}')
        
        b, a, t = attn.diffusers_block_index, attn.diffusers_attn_index, attn.diffusers_transformer_index
        mod = blocks[b].attentions[a].transformer_blocks[t]
        
        hook(key, mod, env, results)
    
    def on_step_end(self: StableDiffusionPipeline, step: int, timestep: int, callback_kwargs: dict):
        env.current_step += 1
        return callback_kwargs
    
    gen_args['callback_on_step_end'] = on_step_end
    
    generate(n_iter, pipe, gen_args, -1)
    
    import pandas as pd
    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    import argparse
    import pprint
    
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', type=str, required=True)
    for attn in ATTN_MAP.keys():
        p.add_argument(f'--{attn.upper()}', type=str, default='')
    p.add_argument('-p', '--prompt', type=str, required=True)
    p.add_argument('-q', '--negative_prompt', type=str, default='')
    p.add_argument('-W', '--width', type=int, default=512)
    p.add_argument('-H', '--height', type=int, default=512)
    p.add_argument('-n', '--num_images', type=int, default=4)
    p.add_argument('-s', '--steps', type=int, default=30)
    p.add_argument('-c', '--cfg_scale', type=float, default=7.0)
    p.add_argument('-i', '--iteration', type=int, default=1)
    p.add_argument('-k', '--seed', type=int, default=-1)

    args = p.parse_args()
    
    gen_args = {
        'prompt': args.prompt,
        'width': args.width,
        'height': args.height,
        'num_inference_steps': args.steps,
        'guidance_scale': 0.0,
        'num_images_per_prompt': args.num_images,
        'return_dict': False,
        'output_type': 'latent',
    }

    if 0.0 < args.cfg_scale and len(args.negative_prompt) != 0:
        gen_args['negative_prompt'] = args.negative_prompt
        gen_args['guidance_scale'] = args.cfg_scale

    
    print('[Hyena] Generation Setting')
    pprint.pprint(gen_args)

    pipe = load_pipeline(args.model)
    pipe = pipe.to('cuda')
    
    data1 = run(pipe, args.iteration, gen_args)
    
    del pipe
    
    pipe = load_pipeline(args.model)
    pipe = pipe.to('cuda')
    
    for attn in ATTN_MAP.keys():
        attn = attn.upper()
        hyena_path = getattr(args, attn)
        if len(hyena_path) != 0:
            replace_hyena(attn, pipe.unet, hyena_path)
    
    data2 = run(pipe, args.iteration, gen_args)
    
    del pipe
    
    import pandas as pd
    data1['type'] = 'sdp'
    data2['type'] = 'hyena'
    data = pd.concat([data1, data2])
    mean = data.groupby(['type', 'name'], as_index=False).mean()
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax00, ax01 = axes
    
    sns.scatterplot(ax=ax00, data=data, x='name', y='time', marker='$\circ$', alpha=0.5, hue='type', hue_order=['sdp', 'hyena'])
    sns.lineplot(ax=ax00, data=mean, x='name', y='time', errorbar=None, markers=False, linewidth=1, hue='type', hue_order=['sdp', 'hyena'])
    
    mean_stack = mean.pivot(index='type', columns=['name'])
    mean_stack.loc[['sdp', 'hyena']].plot.bar(ax=ax01, stacked=True, y='time')
    
    ax00.set_ylabel('time (ms)')
    ax01.set_ylabel('time (ms)')
    
    plt.show()
    