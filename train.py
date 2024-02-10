import dataclasses

import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import safetensors.torch
#from prodigyopt import Prodigy

from mods import Hyena, AttentionSDP

@dataclasses.dataclass
class TrainConf:
    batch_size: int
    lr: float
    n_steps: int
    save_every_n_steps: int
    log_dir: str
    out_dir: str
    name_format: str
    train_seed: int
    width: int
    height: int
    channels: int


def init_seed(seed: int):
    import random
    import numpy
    
    if seed < 0:
        return
    
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

def save(mod: torch.nn.Module, path: str, n_step: int):
    meta = {
        'global_step': n_step,
    }
    meta = {
        str(k): str(v)
        for k, v
        in meta.items()
    }
    safetensors.torch.save_model(mod, path, meta)

ATTN_MAP = {
    'IN01': 'input_blocks.1',
    'IN02': 'input_blocks.2',
    'IN04': 'input_blocks.4',
    'IN05': 'input_blocks.5',
    'IN07': 'input_blocks.7',
    'IN08': 'input_blocks.8',
    'M00': 'middle_block',
    'OUT03': 'output_blocks.3',
    'OUT04': 'output_blocks.4',
    'OUT05': 'output_blocks.5',
    'OUT06': 'output_blocks.6',
    'OUT07': 'output_blocks.7',
    'OUT08': 'output_blocks.8',
    'OUT09': 'output_blocks.9',
    'OUT10': 'output_blocks.10',
    'OUT11': 'output_blocks.11',
}

def load_teacher_model(path: str, target: str, conf: TrainConf):
    sd = safetensors.torch.load_file(path)

    target = target.upper()
    KEY = f'model.diffusion_model.{ATTN_MAP[target]}.1.transformer_blocks.0.attn1.'
    
    def repr(key: str):
        if not key.startswith(KEY):
            return key
        key = key.replace('.to_out.0.', '.to_out.')
        return key[len(KEY):]
    
    sd = {
        repr(k): v
        for k, v in sd.items()
        if KEY in k
    }
    
    attn = AttentionSDP(conf.channels, conf.channels, conf.channels)
    attn.load_state_dict(sd)
    
    return attn

def train(mod: torch.nn.Module, teacher: torch.nn.Module, conf: TrainConf, logger: SummaryWriter):
    batch_size = conf.batch_size
    lr = conf.lr
    total_steps = conf.n_steps
    save_step = conf.save_every_n_steps
    out_dir = conf.out_dir
    fmt = conf.name_format
    seed = conf.train_seed
    w = conf.width
    h = conf.height
    c = conf.channels
    
    mod = mod.to('cuda')
    mod.requires_grad_(True)
    mod.train()

    teacher = teacher.to('cuda')
    teacher.requires_grad_(False)
    
    params = list(mod.parameters())
    optimizer = torch.optim.AdamW(params, lr)
    #optimizer = Prodigy(params, 1.0)
    #optimizer.zero_grad(set_to_none=True)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1, last_epoch=-1)
    
    pbar = tqdm.tqdm(range(total_steps), smoothing=0, desc='steps')
    
    init_seed(seed)

    for global_steps in range(total_steps):
        optimizer.zero_grad()

        x = torch.randn((batch_size, h*w, c), dtype=torch.float16, device='cuda')
        
        with torch.no_grad():
            y = teacher(x)
        
        y_pred = mod(x)
        
        loss = torch.nn.functional.mse_loss(y_pred, y, reduction='none')
        loss = loss.mean()
        
        loss.backward()

        optimizer.step()
        lr_sched.step()
        
        logger.add_scalar('loss', loss.item() / batch_size, global_steps)
        logger.add_scalar('lr', lr_sched.get_last_lr()[0], global_steps)
        
        pbar.update()
        global_steps += 1
        
        if (global_steps + 1 == total_steps) or (0 < save_step and (global_steps + 1) % save_step == 0):
            outname = fmt.format(s=global_steps+1)
            path = f'{out_dir}/{outname}.safetensors'
            save(mod, path, global_steps+1)


if __name__ == '__main__':
    import os
    import argparse
    import time
    import datetime
    
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--teacher_model', type=str, required=True)
    p.add_argument('-t', '--target', type=str, choices=['IN01', 'IN02', 'IN04', 'IN05', 'IN07', 'IN08', 'M00', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11'], default='IN01')
    p.add_argument('-b', '--batch_size', type=int, default=16)
    p.add_argument('-r', '--lr', type=float, default=1e-5)
    p.add_argument('-n', '--n_steps', type=int, default=200)
    p.add_argument('-c', '--channels', type=int, default=320)
    p.add_argument('-W', '--width', type=int, default=64)
    p.add_argument('-H', '--height', type=int, default=64)
    p.add_argument('-s', '--save_every_n_steps', type=int, default=40)
    p.add_argument('-l', '--log_dir', type=str, default='logs')
    p.add_argument('-o', '--out_dir', type=str, default='out')
    p.add_argument('-f', '--name_format', type=str, default='Hyena-{s:05d}')
    p.add_argument('-d', '--seed', type=int, default=-1)
    p.add_argument('-p', '--pretrained_weight', type=str, default='')
    args = p.parse_args()

    target = args.target
    d = args.channels
    w = args.width
    h = args.height
    
    b = args.batch_size
    lr = args.lr
    n = args.n_steps
    s = args.save_every_n_steps
    seed = args.seed

    now = datetime.datetime.now().strftime('%Y%m%d')
    log_dir=f'{args.log_dir}/{now}.{int(time.time())}'
    out_dir = args.out_dir
    fmt = args.name_format
    
    print(f'[Hyena] Training Setting')
    print(f'  Image Width   = {w}')
    print(f'  Image Height  = {h}')
    print(f'  Latent Ch     = {d}')
    print(f'  Latent Shape  = ({d}, {h*w})')
    print(f'  Target        = {target}')
    print(f'  Batch Size    = {b}')
    print(f'  Lr            = {lr}')
    print(f'  Steps         = {n}')
    print(f'  Saving steps  = {s}')
    print(f'  Log dir       = {log_dir}')
    print(f'  Out dir       = {out_dir}')
    print(f'  Name format   = {fmt}')

    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(out_dir, exist_ok=True)
    
    conf = TrainConf(b, lr, n, s, log_dir, out_dir, fmt, seed, w, h, d)
    
    hyena = Hyena(d, h * w)
    
    if len(args.pretrained_weight) != 0:
        hyena = hyena.half()
        safetensors.torch.load_model(hyena, args.pretrained_weight)
    
    teacher = load_teacher_model(args.teacher_model, target, conf)
    
    logger = SummaryWriter(log_dir=log_dir)
    logger.add_hparams(vars(args), {})
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        train(hyena, teacher, conf, logger)
