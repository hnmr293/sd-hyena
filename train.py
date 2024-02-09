import time
import datetime

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

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


def train(mod: torch.nn.Module, batch_size: int, lr: float, epochs: int, dataset: MyDataset):
    mod = mod.to('cuda')
    mod.requires_grad_(True)
    mod.train()
    
    loader = DataLoader(dataset, batch_size, shuffle=True)

    params = list(mod.parameters())
    optimizer = torch.optim.AdamW(params, lr)
    #optimizer.zero_grad(set_to_none=True)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1, last_epoch=-1)
    
    pbar = tqdm.tqdm(range(epochs * len(loader)), smoothing=0, desc='steps')
    
    now = datetime.datetime.now().strftime('%Y%m%d')
    logger = SummaryWriter(log_dir=f'logs/{now}.{int(time.time())}')
    
    global_steps = 0

    for epoch in range(epochs):
        for iter, (x, y) in enumerate(loader):
            #with torch.enable_grad():
            optimizer.zero_grad()

            x = x.to('cuda')
            y = y.to('cuda')

            #x = torch.sigmoid(x)
            y_pred = mod(x)
            loss = torch.nn.functional.mse_loss(y_pred, y, reduction='none')
            loss = loss.mean()
            
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(params, 1.0)
            
            optimizer.step()
            lr_sched.step()
            
            logger.add_scalar('loss', loss.item() / batch_size, global_steps)
            logger.add_scalar('lr', lr_sched.get_last_lr()[0], global_steps)
            
            pbar.update()
            global_steps += 1

if __name__ == '__main__':
    hyena = Hyena(320, 64*64)
    dataset = MyDataset('data')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        train(hyena, 16, 1e-2, 100, dataset)
