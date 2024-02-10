import dataclasses

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import safetensors.torch

from mods import Hyena

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str):
        import glob
        self.files = glob.glob(f'{dir}/*.npy')
        self.sizes = []
        self.total_size = 0
        for file in self.files:
            array = torch.HalfTensor(numpy.load(file))
            # s, 2, n, d
            assert array.ndim == 4
            assert array.size(1) == 2
            s = array.size(0)
            self.sizes.append(s)
            self.total_size += s

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        file = None
        for i, s in enumerate(self.sizes):
            if index < s:
                file = self.files[i]
                break
            index -= s
        assert file is not None

        t = torch.from_numpy(numpy.load(file))
        # s, 2, n, d -> 2, n, d
        t = t[index]
        assert t.dtype == torch.float16
        assert t.ndim == 3
        assert t.size(0) == 2
        i, o = t[0], t[1]

        return i, o


@dataclasses.dataclass
class TrainConf:
    batch_size: int
    lr: float
    n_epochs: int
    save_every_n_epochs: int
    log_dir: str
    out_dir: str
    name_format: str


def save(mod: torch.nn.Module, path: str, n_epoch: int, n_step: int):
    meta = {
        'epoch': n_epoch,
        'global_step': n_step,
    }
    meta = {
        str(k): str(v)
        for k, v
        in meta.items()
    }
    safetensors.torch.save_model(mod, path, meta)

def train(mod: torch.nn.Module, conf: TrainConf, dataset: MyDataset, logger: SummaryWriter):
    batch_size = conf.batch_size
    lr = conf.lr
    epochs = conf.n_epochs
    save_epoch = conf.save_every_n_epochs
    out_dir = conf.out_dir
    fmt = conf.name_format
    
    mod = mod.to('cuda')
    mod.requires_grad_(True)
    mod.train()
    
    loader = DataLoader(dataset, batch_size, shuffle=True)

    params = list(mod.parameters())
    optimizer = torch.optim.AdamW(params, lr)
    #optimizer.zero_grad(set_to_none=True)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1, last_epoch=-1)
    
    pbar = tqdm.tqdm(range(epochs * len(loader)), smoothing=0, desc='steps')
    
    global_steps = 0

    for epoch in range(epochs):
        for iter, (x, y) in enumerate(loader):
            #with torch.enable_grad():
            optimizer.zero_grad()

            x = x.to('cuda')
            y = y.to('cuda')

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
        
        if (epoch == epochs - 1) or (0 < save_epoch and (epoch + 1) % save_epoch == 0):
            outname = fmt.format(e=epoch+1, s=global_steps)
            path = f'{out_dir}/{outname}.safetensors'
            save(mod, path, epoch+1, global_steps)


if __name__ == '__main__':
    import os
    import argparse
    import time
    import datetime
    
    p = argparse.ArgumentParser()
    p.add_argument('-b', '--batch_size', type=int, default=8)
    p.add_argument('-r', '--lr', type=float, default=1e-3)
    p.add_argument('-n', '--n_epochs', type=int, default=50)
    p.add_argument('-c', '--channels', type=int, default=320)
    p.add_argument('-W', '--width', type=int, default=64)
    p.add_argument('-H', '--height', type=int, default=64)
    p.add_argument('-s', '--save_every_n_epochs', type=int, default=1)
    p.add_argument('-l', '--log_dir', type=str, default='logs')
    p.add_argument('-o', '--out_dir', type=str, default='out')
    p.add_argument('-f', '--name_format', type=str, default='Hyena-{e:05d}-{s:05d}')
    args = p.parse_args()
    
    d = args.channels
    w = args.width
    h = args.height
    
    b = args.batch_size
    lr = args.lr
    n = args.n_epochs
    s = args.save_every_n_epochs

    now = datetime.datetime.now().strftime('%Y%m%d')
    log_dir=f'{args.log_dir}/{now}.{int(time.time())}'
    out_dir = args.out_dir
    fmt = args.name_format
    
    print(f'[Hyena] Training Setting')
    print(f'  Image Width   = {w}')
    print(f'  Image Height  = {h}')
    print(f'  Latent Ch     = {d}')
    print(f'  Latent Shape  = ({d}, {h*w})')
    print(f'  Batch Size    = {b}')
    print(f'  Lr            = {lr}')
    print(f'  Epochs        = {n}')
    print(f'  Saving epochs = {s}')
    print(f'  Log dir       = {log_dir}')
    print(f'  Out dir       = {out_dir}')
    print(f'  Name format   = {fmt}')

    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(out_dir, exist_ok=True)
    
    hyena = Hyena(d, h * w)
    dataset = MyDataset('data')
    conf = TrainConf(b, lr, n, s, log_dir, out_dir, fmt)
    
    logger = SummaryWriter(log_dir=log_dir)
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        train(hyena, conf, dataset, logger)
