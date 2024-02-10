from typing import Optional

import torch
import einops
import xformers
import xformers.ops
from safari import HyenaOperator

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


class HyenaProcessor(HyenaOperator):
    """for diffusers"""

    def __init__(self, input_dim=320, max_len=64*64, order=2, filter_order=64, num_heads=1, inner_factor=1, num_blocks=1, short_filter_order=3) -> None:
        super().__init__(d_model=input_dim, l_max=max_len, order=order, filter_order=filter_order, num_heads=num_heads, inner_factor=inner_factor, num_blocks=num_blocks, short_filter_order=short_filter_order)
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        assert encoder_hidden_states is None
        return super().forward(hidden_states)
