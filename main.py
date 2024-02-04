import statistics
import contextlib
import dataclasses

import torch
import einops
import xformers
import xformers.ops

from safari import HyenaOperator

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


class Attention(torch.nn.Module):

    def __init__(self, query_dim=320, ctx_dim=320, inner_dim=320, num_heads=8) -> None:
        super().__init__()
        
        assert inner_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.scale = (inner_dim // num_heads) ** -0.5
        
        self.to_q = torch.nn.Linear(in_features=query_dim, out_features=inner_dim, bias=False)
        self.to_k = torch.nn.Linear(in_features=ctx_dim, out_features=inner_dim, bias=False)
        self.to_v = torch.nn.Linear(in_features=ctx_dim, out_features=inner_dim, bias=False)
        self.to_out = torch.nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True)

        self.init_weight(self.to_q)
        self.init_weight(self.to_k)
        self.init_weight(self.to_v)
        self.init_weight(self.to_out)
    
    def init_weight(self, mod):
        if hasattr(mod, 'weight') and mod.weight is not None:
            torch.nn.init.uniform_(mod.weight, -1.0, 1.0)
        if hasattr(mod, 'bias') and mod.bias is not None:
            torch.nn.init.uniform_(mod.bias, -1.0, 1.0)
    
    def get_qkv(self, x, ctx=None):
        ctx = ctx if ctx is not None else x

        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        return q, k, v
    
    def forward(self, x, ctx=None):
        h = self.num_heads
        q_in, k_in, v_in = self.get_qkv(x, ctx)
        
        q = einops.rearrange(q_in, 'b n (h d) -> (b h) n d', h=h)
        k = einops.rearrange(k_in, 'b n (h d) -> (b h) n d', h=h)
        v = einops.rearrange(v_in, 'b n (h d) -> (b h) n d', h=h)

        del q_in, k_in, v_in
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)

        del v, sim

        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out

class AttentionXFormers(Attention):
    def forward(self, x, ctx=None):
        h = self.num_heads
        q_in, k_in, v_in = self.get_qkv(x, ctx)
        
        q = einops.rearrange(q_in, 'b n (h d) -> b n h d', h=h)
        k = einops.rearrange(k_in, 'b n (h d) -> b n h d', h=h)
        v = einops.rearrange(v_in, 'b n (h d) -> b n h d', h=h)
        
        del q_in, k_in, v_in
        
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp)

        del q, k, v
        
        out = einops.rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)

        return out

class AttentionSDP(Attention):
    def forward(self, x, ctx=None):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
            h = self.num_heads
            q_in, k_in, v_in = self.get_qkv(x, ctx)
            
            q = einops.rearrange(q_in, 'b n (h d) -> b h n d', h=h)
            k = einops.rearrange(k_in, 'b n (h d) -> b h n d', h=h)
            v = einops.rearrange(v_in, 'b n (h d) -> b h n d', h=h)
            
            del q_in, k_in, v_in
            
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
            
            del q, k, v
            
            out = einops.rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            
            return out

class Hyena(HyenaOperator):
    def __init__(self, input_dim=320, max_len=64*64, order=2, filter_order=64, num_heads=1, inner_factor=1, num_blocks=1, short_filter_order=3) -> None:
        super().__init__(d_model=input_dim, l_max=max_len, order=order, filter_order=filter_order, num_heads=num_heads, inner_factor=inner_factor, num_blocks=num_blocks, short_filter_order=short_filter_order)
    
    def forward(self, x, ctx=None):
        assert ctx is None
        return super().forward(x)


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
            
            with cuda_profiler(device) as prof:
                x = mod(x)
            
            return x, prof

    # test shape and dtypes
    for name, mod in mods.items():
        x, _ = test(mod)
        assert x.shape == (b, wh, d), f'{name}, {x.shape}'
        assert x.dtype == torch.float16 if use_fp16 else torch.float32, f'{name}, {x.dtype}'
        del x
    
    # run
    results = {}
    for name, mod in mods.items():
        # warm-up
        for _ in range(10):
            test(mod)

        profs = Profs()
        profs.n_params = n_params(mod)

        for _ in range(n_iter):
            _, prof = test(mod)
            profs.time.append(prof['time'])
            profs.memory.append(prof['memory'])
        
        results[name] = profs

    return results


def main(n_iter: int = 100):
    device = 'cuda:0'
    use_fp16 = True

    b = 8
    w = 64
    h = 64
    d = 320

    results = run(b, w*h, d, device, use_fp16, n_iter=n_iter)

    # create graphs
    import pandas as pd
    import matplotlib.pyplot as plt
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
    
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    
    ax0.plot(names, n_param.values(), marker='o', linestyle='')
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel('parameters')
    
    sns.stripplot(ax=ax1, data=time_df, x=xlabel, y=ylabel_time, marker='$\circ$', alpha=0.5, log_scale=10)
    ax1.plot(names, time_means, color='black', marker='_', markersize=20, linestyle='')

    sns.stripplot(ax=ax2, data=mem_df, x=xlabel, y=ylabel_mem, marker='$\circ$', alpha=0.5)
    ax2.plot(names, mem_means, color='black', marker='_', markersize=20, linestyle='')

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n_iters', type=int, default=100)
    args = p.parse_args()
    main(args.n_iters)
