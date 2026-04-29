import torch
from einops import rearrange
from functools import wraps
from modules import shared, sd_hijack_optimizations
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import sgm.modules.attention
import sgm.modules.diffusionmodules.model

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ImportError:
    SAGE_ATTN_AVAILABLE = False
    print("[a1111_amd_sage_attention] sageattention package not found. Sage Attention will be unavailable.")

def sage_attention_forward(self, x, context=None, mask=None, **kwargs):
    from ldm.util import default
    from modules.hypernetworks import hypernetwork

    batch_size, sequence_length, inner_dim = x.shape
    h = self.heads
    head_dim = inner_dim // h

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    target_dtype = torch.float16 if dtype == torch.float32 else dtype

    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2).to(target_dtype)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2).to(target_dtype)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2).to(target_dtype)

    out = sageattn(q, k, v, is_causal=False)

    out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim).to(dtype)

    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

def sage_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    b, c, h, w = q.shape

    if c > 256 or c % 8 != 0:
        q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out

    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))

    dtype = q.dtype
    target_dtype = torch.float16 if dtype == torch.float32 else dtype

    q = q.unsqueeze(2).transpose(1, 2).to(target_dtype)
    k = k.unsqueeze(2).transpose(1, 2).to(target_dtype)
    v = v.unsqueeze(2).transpose(1, 2).to(target_dtype)

    out = sageattn(q, k, v, is_causal=False)

    out = out.transpose(1, 2).squeeze(2).to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    
    out = self.proj_out(out)
    return x + out

class SdOptimizationSageAttentionAMD(sd_hijack_optimizations.SdOptimization):
    name = "Sage Attention 1 (AMD Triton)"
    cmd_opt = "opt_sage_attn_amd"
    priority = 54

    def is_available(self):
        return SAGE_ATTN_AVAILABLE

    def apply(self):
        ldm.modules.attention.CrossAttention.forward = sage_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sage_attnblock_forward
        sgm.modules.attention.CrossAttention.forward = sage_attention_forward
        sgm.modules.diffusionmodules.model.AttnBlock.forward = sage_attnblock_forward

    def undo(self):
        super().undo()

def inject_sage_attention():
    original_func = sd_hijack_optimizations.list_optimizers

    @wraps(original_func)
    def wrapper(res):
        original_func(res)
        if not any(opt.name == "Sage Attention 1 (AMD Triton)" for opt in res):
            res.append(SdOptimizationSageAttentionAMD())

    sd_hijack_optimizations.list_optimizers = wrapper


inject_sage_attention()
