import contextlib
import torch
from mods import ATTN_MAP

def init_seed(seed: int):
    import random
    import numpy
    import torch.cuda
    
    if seed < 0:
        return
    
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def load_pipeline(model_path: str):
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        #local_files_only=True,
        cache_dir='.cache',
    )

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe


def replace_hyena(target: str, unet, hyena_path: str):
    from mods import HyenaProcessor
    import safetensors.torch

    target = target.upper()
    info = ATTN_MAP[target]
    
    b, a, t = info.diffusers_block_index, info.diffusers_attn_index, info.diffusers_transformer_index
    
    with safetensors.safe_open(hyena_path, framework='pt') as f:
        meta = f.metadata()
    w = int(meta['train_width'])
    h = int(meta['train_height'])
    d = int(meta['train_channels'])
    
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


@contextlib.contextmanager
def cuda_profiler(device='cuda'):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)

    obj = {}
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    cuda_start.record()
    
    try:
        yield obj
    finally:
        pass

    cuda_end.record()
    torch.cuda.synchronize()
    obj['time'] = cuda_start.elapsed_time(cuda_end)
    obj['memory'] = torch.cuda.max_memory_allocated(device)


