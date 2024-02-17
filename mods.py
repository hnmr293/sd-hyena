from typing import Optional
import dataclasses

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
    def __init__(self, input_dim=320, max_len=64*64, order=2, filter_order=64, num_heads=1, inner_factor=1, num_blocks=1, short_filter_order=3, fused_bias_fc=False) -> None:
        super().__init__(
            d_model=input_dim,
            l_max=max_len,
            order=order,
            filter_order=filter_order,
            num_heads=num_heads,
            inner_factor=inner_factor,
            num_blocks=num_blocks,
            short_filter_order=short_filter_order,
            fused_bias_fc=fused_bias_fc,
        )
    
    def forward(self, x, ctx=None):
        assert ctx is None
        return super().forward(x)


class HyenaProcessor(HyenaOperator):
    """for diffusers"""

    def __init__(self, input_dim=320, max_len=64*64, order=2, filter_order=64, num_heads=1, inner_factor=1, num_blocks=1, short_filter_order=3, fused_bias_fc=True) -> None:
        super().__init__(
            d_model=input_dim,
            l_max=max_len,
            order=order,
            filter_order=filter_order,
            num_heads=num_heads,
            inner_factor=inner_factor,
            num_blocks=num_blocks,
            short_filter_order=short_filter_order,
            fused_bias_fc=fused_bias_fc,
        )
    
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


@dataclasses.dataclass
class AttnMap:
    diffusers_block_index: int
    diffusers_attn_index: int
    diffusers_transformer_index: int
    multiplier: int
    input_channels: int
    output_channels: int
    name: str

ATTN_MAP: dict[str,AttnMap] = {
    #             v            v                    v
    # down_blocks.0.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
    # down_blocks.0.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
    # down_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)
    # down_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)
    # down_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)
    # down_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)
    # mid_block.attentions.0.transformer_blocks.0.attn1 (2, 64, 1280) (2, 64, 1280)
    # up_blocks.1.attentions.0.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)
    # up_blocks.1.attentions.1.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)
    # up_blocks.1.attentions.2.transformer_blocks.0.attn1 (2, 256, 1280) (2, 256, 1280)
    # up_blocks.2.attentions.0.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)
    # up_blocks.2.attentions.1.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)
    # up_blocks.2.attentions.2.transformer_blocks.0.attn1 (2, 1024, 640) (2, 1024, 640)
    # up_blocks.3.attentions.0.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
    # up_blocks.3.attentions.1.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
    # up_blocks.3.attentions.2.transformer_blocks.0.attn1 (2, 4096, 320) (2, 4096, 320)
    'IN01': AttnMap(0, 0, 0, 8, 320, 320, 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1'),
    'IN02': AttnMap(0, 1, 0, 8, 320, 320, 'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1'),
    'IN04': AttnMap(1, 0, 0, 16, 640, 640, 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1'),
    'IN05': AttnMap(1, 1, 0, 16, 640, 640, 'model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1'),
    'IN07': AttnMap(2, 0, 0, 32, 1280, 1280, 'model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1'),
    'IN08': AttnMap(2, 1, 0, 32, 1280, 1280, 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1'),
    'M00': AttnMap(0, 0, 0, 64, 1280, 1280, 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1'),
    'OUT03': AttnMap(1, 0, 0, 32, 1280, 1280, 'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1'),
    'OUT04': AttnMap(1, 1, 0, 32, 1280, 1280, 'model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1'),
    'OUT05': AttnMap(1, 2, 0, 32, 1280, 1280, 'model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1'),
    'OUT06': AttnMap(2, 0, 0, 16, 640, 640, 'model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1'),
    'OUT07': AttnMap(2, 1, 0, 16, 640, 640, 'model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1'),
    'OUT08': AttnMap(2, 2, 0, 16, 640, 640, 'model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1'),
    'OUT09': AttnMap(3, 0, 0, 8, 320, 320, 'model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1'),
    'OUT10': AttnMap(3, 1, 0, 8, 320, 320, 'model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1'),
    'OUT11': AttnMap(3, 2, 0, 8, 320, 320, 'model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1'),
}