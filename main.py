import statistics
import contextlib
import dataclasses

import torch

from mods import Attention, AttentionSDP, AttentionXFormers, Hyena

@dataclasses.dataclass
class Profs:
    time: list[float] = dataclasses.field(default_factory=list)
    memory: list[int] = dataclasses.field(default_factory=list)
    n_params: int = 0

def mean_and_stdev(xs: list):
    mean = statistics.fmean(xs)
    stdev = statistics.stdev(xs)
    return mean, stdev

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


def n_params(mod: torch.nn.Module):
    n = 0
    for param in mod.parameters():
        if param.requires_grad:
            n += param.numel()
    return n


def run(b: int, wh: int, d: int, device, use_fp16: bool, n_iter: int = 100):
    attn = Attention(d, d, d, 8)
    attn_x = AttentionXFormers(d, d, d, 8)
    attn_sdp = AttentionSDP(d, d, d, 8)
    hyena = Hyena(d, wh)

    mods = {
        'original': attn,
        'xformers': attn_x,
        'sdp': attn_sdp,
        'hyena': hyena,
    }
    
    def test(mod):
        with torch.no_grad():
            x = torch.rand((b, wh, d), dtype=torch.float32)
            
            if use_fp16:
                x = x.half()
                mod = mod.half()
            else:
                mod = mod.float32()
            
            x = x.to(device=device)
            mod = mod.to(device=device)
            
            if mod == attn and 128*128 < wh:
                return x, None
            
            if mod == hyena and use_fp16:
                if 2 <= wh and (wh & (wh - 1)) != 0:
                    # cuFFT does not support this manner
                    return x, None
            
            with cuda_profiler(device) as prof:
                x = mod(x)
            
            return x, prof

    # test shape and dtypes
    for name, mod in mods.items():
        x, _ = test(mod)
        assert x.shape == (b, wh, d), f'{name}, {x.shape}, expected = ({b}, {wh}, {d})'
        assert x.dtype == torch.float16 if use_fp16 else torch.float32, f'{name}, {x.dtype}'
        del x
    
    # run
    results = {}
    for name, mod in mods.items():
        # warm-up
        for _ in range(10):
            x, _ = test(mod)
            del x

        profs = Profs()
        profs.n_params = n_params(mod)

        for _ in range(n_iter):
            x, prof = test(mod)
            del x
            
            if prof is None:
                continue
            
            profs.time.append(prof['time'])
            profs.memory.append(prof['memory'])
        
        results[name] = profs

    return results


def main_performance(ax0, ax1, ax2, n_iter: int = 100):
    device = 'cuda:0'
    use_fp16 = True

    b = 8
    w = 64
    h = 64
    d = 320

    results = run(b, w*h, d, device, use_fp16, n_iter=n_iter)

    import pandas as pd
    import seaborn as sns
    
    time = {
        name: prof.time
        for name, prof
        in results.items()
    }

    memory = {
        name: [x / 1024 / 1024 for x in prof.memory]
        for name, prof
        in results.items()
    }

    n_param = {
        name: prof.n_params
        for name, prof
        in results.items()
    }

    names = list(results.keys())

    xlabel = 'attn kind'
    ylabel_time = 'time (ms)'
    ylabel_mem = 'VRAM (MiB)'

    def create_df(data, xlabel, ylabel):
        df = pd.DataFrame(data, columns=names)
        means = df.mean(axis=0).to_list()
        df = pd.melt(frame=df, var_name=xlabel, value_name=ylabel)
        return df, means
    
    time_df, time_means = create_df(time, xlabel, ylabel_time)
    mem_df, mem_means = create_df(memory, xlabel, ylabel_mem)
    
    ax0.plot(names, n_param.values(), marker='o', linestyle='')
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel('parameters')
    
    sns.stripplot(ax=ax1, data=time_df, x=xlabel, y=ylabel_time, marker='$\circ$', alpha=0.5, log_scale=10)
    ax1.plot(names, time_means, color='black', marker='_', markersize=20, linestyle='')

    sns.stripplot(ax=ax2, data=mem_df, x=xlabel, y=ylabel_mem, marker='$\circ$', alpha=0.5)
    ax2.plot(names, mem_means, color='black', marker='_', markersize=20, linestyle='')

def main_performance2(ax0, ax1, ax2, n_iter: int = 10):
    device = 'cuda:0'
    use_fp16 = True

    b = 1
    d = 320
    ws = (256, 512, 768, 1024, 1280, 2048, 4096)

    results = {}
    for w in ws:
        lw = w // 8
        lh = lw
        r = run(b, lw*lh, d, device, use_fp16, n_iter)
        results[w] = r

    import pandas as pd
    import seaborn as sns
    
    rows = []
    for w, series in results.items():
        for attn_kind, prof in series.items():
            for t, m in zip(prof.time, prof.memory):
                rows.append([w, attn_kind, prof.n_params, t, m/1024/1024])
    df = pd.DataFrame(rows, columns=['image size', 'attn kind', 'parameters', 'time (ms)', 'VRAM (MiB)'])
    mean_df = df.groupby(['image size', 'attn kind']).mean()

    names = df['attn kind'].unique()
    
    sns.lineplot(ax=ax0, data=df[['image size', 'attn kind', 'parameters']].drop_duplicates(), x='image size', y='parameters', markers=True, marker='o', dashes=False, hue='attn kind', hue_order=names)
    
    sns.scatterplot(ax=ax1, data=df, x='image size', y='time (ms)', marker='$\circ$', alpha=0.5, hue='attn kind', hue_order=names)
    sns.lineplot(ax=ax1, data=mean_df, x='image size', y='time (ms)', errorbar=None, markers=False, linewidth=1, hue='attn kind', hue_order=names)
    ax1.set(yscale='log')
    
    sns.scatterplot(ax=ax2, data=df, x='image size', y='VRAM (MiB)', marker='$\circ$', alpha=0.5, hue='attn kind', hue_order=names)
    sns.lineplot(ax=ax2, data=mean_df, x='image size', y='VRAM (MiB)', errorbar=None, markers=False, linewidth=1, hue='attn kind', hue_order=names)

def main(n_iter: int = 100):
    # create graphs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=3)
    (ax0, ax1, ax2), (ax3, ax4, ax5) = axes
    
    main_performance2(ax0, ax1, ax2, max(n_iter//10, 5))
    main_performance(ax3, ax4, ax5, n_iter)

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n_iters', type=int, default=100)
    args = p.parse_args()
    main(args.n_iters)
