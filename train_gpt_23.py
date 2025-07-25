import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path
from tqdm import tqdm 
import wandb
import argparse
# 添加TensorBoard支持
from torch.utils.tensorboard import SummaryWriter

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--wandb", type = str, default="no", choices=["yes", "no"], help="Whether to use wandb for logging")
parser.add_argument("--muon_momentum_0", type=float, default=0.95, help="Muon momentum[0] for nesterov")
parser.add_argument("--muon_momentum_1", type=float, default=0.95, help="Muon momentum[1] for buf")
parser.add_argument("--muon_type", type=str, default="default", help="Muon optimizer type")
parser.add_argument("--adaptive_beta", type=str, default="no", choices=["cos","NS_cos", "no"], help="Whether to use adaptive beta based on cosine similarity")
parser.add_argument("--tensorboard_path", type=str, default="logs_tensorboard_train23", help="Tensorboard path")
parser.add_argument("--wandb_project", type=str, default="train_gpt_23", help="Wandb project")
cli_args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        weight_decay: Weight decay factor for regularization.
        momentum: The momentum used by the internal SGD (can be single value or tuple for double_momentum).
        ns_steps: The number of Newton-Schulz iteration steps to use.
        type: Momentum type - 'default', 'double_momentum', 'svd_momentum', 'svd_momentum_v2'
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, ns_steps=5, rank=0, world_size=1, type='default', adaptive_beta="no"):
        self.rank = rank
        self.world_size = world_size
        self.type = type
        self.adaptive_beta = cli_args.adaptive_beta
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    def compute_cosine_similarity(self, g, buf, version):
        """计算g和buf之间的余弦相似度"""
        if version == "cos":
            g_flat = g.flatten()
            buf_flat = buf.flatten()
            
            # 计算余弦相似度
            dot_product = torch.dot(g_flat, buf_flat)
            g_norm = torch.norm(g_flat)
            buf_norm = torch.norm(buf_flat)
            
            # 避免除零
            cosine_sim = dot_product / (g_norm * buf_norm + 1e-8) 
            # 确保相似度在合理范围内，使用绝对值
            # 返回cosine_sim和0的较大值
            return max(cosine_sim, 0)
        elif version == "NS_cos":
            _ortho = zeropower_via_newtonschulz5(g, steps=self.ns_steps).flatten()
            buf_ortho = zeropower_via_newtonschulz5(buf, steps=self.ns_steps).flatten()
            g_norm = torch.norm(g_ortho)
            buf_norm = torch.norm(buf_ortho)
            dot_product = torch.dot(g_ortho.flatten(), buf_ortho.flatten())
            cosine_sim = dot_product / (g_norm * buf_norm + 1e-8) 
            return max(cosine_sim, 0)
    


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * 0.2 * max(p_world.size(-2), p_world.size(-1))**0.5) # 按照月之暗面的paper的learning rate scaling
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]

                    # 处理4D卷积滤波器
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                        buf = buf.view(len(buf), -1)

                    # 根据 type 选择不同的动量和正交化策略
                    if self.type == 'default':
                        # 计算adaptive momentum
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g, 1 - momentum) # 0.98
                        g_muon = g.lerp_(buf, momentum)
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                    elif self.type == 'double_momentum':
                        # momentum[1] 用于 buf，momentum[0] 用于 nesterov
                        momentum_0 = cli_args.muon_momentum_0
                        momentum_1 = cli_args.muon_momentum_1
                        
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum_0 = momentum_0 + (1-momentum_0) * cosine_sim
                            momentum_1 = momentum_1 + (1-momentum_1) * cosine_sim
                        
                        buf.lerp_(g, 1 - momentum_1) # 0.98
                        g_muon = g.lerp_(buf, momentum_0) # 0.95
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                    
                    elif self.type == 'double_momentum_reverse':
                        # momentum[1] 用于 buf，momentum[0] 用于 nesterov
                        momentum_0 = cli_args.muon_momentum_0
                        momentum_1 = cli_args.muon_momentum_1
                        
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum_0 = momentum_0 + (1-momentum_0) * cosine_sim
                            momentum_1 = momentum_1 + (1-momentum_1) * cosine_sim
                        
                        g_muon = buf * momentum_1 + g * (1 - momentum_1)
                        #g_muon = g.lerp_(g_muon, momentum_1) # 0.98
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                        buf.lerp_(g, 1 - momentum_1) # 0.98
                    elif self.type == 'svd_momentum':
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g, 1 - momentum)
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                        buf_ortho = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                        g_muon = g_ortho.lerp_(buf_ortho, momentum).flatten()
                    elif self.type == 'NS_momentum':
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g_ortho, 1 - momentum)
                        g_muon = g_ortho.lerp_(buf, cli_args.muon_momentum_1).flatten().to(dtype=torch.bfloat16)
                    
                    elif self.type == "doubleNS_momentum_v1":
                        # 再NS_momentum的基础上，对nesterov的更新也进行NS正交化,保证update的NS正交性
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g_ortho, 1 - momentum)
                        g_muon = g_ortho.lerp_(buf, cli_args.muon_momentum_1).to(dtype=torch.bfloat16)
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                    
                    elif self.type == "doubleNS_momentum_v2":
                        # 计算adaptive momentum
                        '''
                        momentum的更新方式不变, update的时候,先对G进行NS，然后以0.5的beta对nesterov步骤，最后对update进行NS正交化
                        '''
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":

                            
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g, 1 - momentum)
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        g_muon = g_ortho.lerp_(buf, cli_args.muon_momentum_1).to(dtype=torch.bfloat16)
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()
                    
                    elif self.type == "tribleNS_v2":
                        momentum = group["momentum"]
                        if self.adaptive_beta != "no":

                            
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum = momentum + (1-momentum) * cosine_sim
                        
                        buf.lerp_(g, 1 - momentum)
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        buf_ortho = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"]).to(dtype=torch.float32)
                        g_muon = g_ortho.lerp_(buf_ortho, cli_args.muon_momentum_1)
                        g_muon = zeropower_via_newtonschulz5(g_muon, steps=group["ns_steps"]).flatten()

                    elif self.type == 'mix': # NS_momentum + double_momentum
                        g_ortho = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).to(dtype=torch.float32)
                        
                        momentum_0 = group["momentum"][0]
                        momentum_1 = group["momentum"][1]
                        if self.adaptive_beta != "no":
                            cosine_sim = self.compute_cosine_similarity(g, buf, self.adaptive_beta)
                            momentum_0 = momentum_0 + (1-momentum_0) * cosine_sim
                            momentum_1 = momentum_1 + (1-momentum_1) * cosine_sim
                        
                        buf.lerp_(g_ortho, 1 - momentum_0)
                        g_muon = g_ortho.lerp_(buf, momentum_1).flatten().to(dtype=torch.bfloat16)
                    else:
                        raise ValueError(f"Unknown Muon type: {self.type}")
                    
                    g = g_muon
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                if g.dtype != update_buffer.dtype:
                    g = g.to(update_buffer.dtype)
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=False, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.to(torch.int32).argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
#assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# 创建TensorBoard writer
name = cli_args.muon_type + str([cli_args.muon_momentum_0, cli_args.muon_momentum_1])
tensorboard_writer = None
if master_process:
    tensorboard_writer = SummaryWriter(cli_args.tensorboard_path)

# begin logging
logfile = None
if cli_args.wandb == "yes":
    if master_process:
        config_dict = vars(args).copy()
        config_dict.update(vars(cli_args))
        
        # 创建唯一的run名称，避免冲突
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建更详细的run名称，包含所有关键参数
        name_parts = [cli_args.muon_type]
        name_parts.append(f"m0_{cli_args.muon_momentum_0}")
        name_parts.append(f"m1_{cli_args.muon_momentum_1}")
        if cli_args.adaptive_beta != "no":
            name_parts.append(cli_args.adaptive_beta)
        name_parts.append(timestamp)
        
        run_name = "_".join(name_parts)
        import uuid
        run_id = run_name + "_" + str(uuid.uuid4())[:8]  # 确保run_id唯一
        wandb.init(
            project=cli_args.wandb_project,
            id=run_name,
            name=run_id,
            config=config_dict,
            resume="never",
            reinit="finish_previous"
        )

def print0(s, console=False):
    if master_process:
        if logfile:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)
        else:
            if console:
                print(s)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):  
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# 配置 momentum 参数
cfg = {
    "default": cli_args.muon_momentum_0,
    "double_momentum": [cli_args.muon_momentum_0, cli_args.muon_momentum_1],
    "double_momentum_reverse": [cli_args.muon_momentum_1, cli_args.muon_momentum_0],
    "NS_momentum": cli_args.muon_momentum_0,
    "doubleNS_momentum_v1": cli_args.muon_momentum_0,
    "doubleNS_momentum_v2": cli_args.muon_momentum_0,
    "tribleNS_v2": cli_args.muon_momentum_0, 
    "mix": [cli_args.muon_momentum_0, cli_args.muon_momentum_1],
    
}

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)

# 根据muon_type选择不同的优化器配置
optimizer2 = Muon(hidden_matrix_params, lr=0.05, weight_decay=0.01, momentum=cfg[cli_args.muon_type],
                      rank=rank, world_size=world_size, type=cli_args.muon_type, adaptive_beta=cli_args.adaptive_beta)


optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

# 记录训练配置信息
if master_process:
    print0("="*100, console=True)
    print0(f"Muon Type: {cli_args.muon_type}", console=True)
    print0(f"Muon Momentum 0: {cli_args.muon_momentum_0}", console=True)
    print0(f"Muon Momentum 1: {cli_args.muon_momentum_1}", console=True)
    print0(f"Adaptive Beta: {cli_args.adaptive_beta}", console=True)
    print0("="*100, console=True)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#      Overlap Communication Setup     #
########################################

# Create parameter buckets for better overlap
def create_buckets(params, bucket_size_mb=25):
    """Group parameters into buckets of approximately bucket_size_mb MB each"""
    buckets = []
    current_bucket = []
    current_size = 0

    # Sort parameters by size (largest first) for better bucketing
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

    for param in sorted_params:
        param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

        if current_size + param_size_mb > bucket_size_mb and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size_mb
        else:
            current_bucket.append(param)
            current_size += param_size_mb

    if current_bucket:
        buckets.append(current_bucket)

    return buckets

# Create buckets for all parameters
all_params = [p for p in model.parameters() if p.requires_grad]
param_buckets = create_buckets(all_params)

print0(f"Created {len(param_buckets)} gradient buckets")
for i, bucket in enumerate(param_buckets):
    total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
    print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB")

# Bucket state tracking
bucket_ready_count = [0] * len(param_buckets)
bucket_handles = [None] * len(param_buckets)
param_to_bucket = {}

# Map each parameter to its bucket index
for bucket_idx, bucket in enumerate(param_buckets):
    for param in bucket:
        param_to_bucket[param] = bucket_idx

def _gradient_hook(param: Tensor):
    """Called when a parameter's gradient is ready"""
    if param.grad is None:
        return

    bucket_idx = param_to_bucket[param]
    bucket_ready_count[bucket_idx] += 1

    # Check if all parameters in this bucket are ready
    if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
        # All-reduce this bucket
        bucket_grads = [p.grad for p in param_buckets[bucket_idx]]

        # For multi-tensor operations, we can reduce them together
        if len(bucket_grads) == 1:
            handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
        else:
            # Use multi-tensor all-reduce for efficiency
            handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)

        bucket_handles[bucket_idx] = handle

# Register hooks for all parameters
print0("Registering bucketed gradient hooks...")
for param in all_params:
    param.register_post_accumulate_grad_hook(_gradient_hook)

def wait_for_gradients():
    """Wait for all gradient reductions to complete and reset bucket state"""
    for handle in bucket_handles:
        if handle is not None:
            handle.wait()

    # Reset state for next iteration
    for i in range(len(bucket_ready_count)):
        bucket_ready_count[i] = 0
        bucket_handles[i] = None

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in tqdm(range(train_steps + 1), desc=f"Training {cli_args.muon_type}", unit="step", disable=not master_process):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        ## Make sure all gradient reductions from the previous training step are finished before validation
        #wait_for_gradients()

        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len #总共需要处理的batch size, 均摊到每个GPU上
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        
        # 记录到wandb
        if cli_args.wandb == "yes" and master_process:
            wandb.log({"val_loss": float(val_loss), "step": step})
        
        # 记录到TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Validation/Loss', float(val_loss), step)
            tensorboard_writer.add_scalar('Training/Time_ms', training_time_ms, step)
            tensorboard_writer.add_scalar('Training/Step_avg_ms', training_time_ms/max(step, 1), step)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs("logs", exist_ok=True)
            torch.save(log, f"logs/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    train_loss = model(inputs, targets, get_window_size_blocks(step))
    train_loss.backward()
    #for param in model.parameters():
    #    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    wait_for_gradients() # does the same thing as commented two lines above, but faster

    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    if  last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        normalized_train_loss = train_loss/args.train_seq_len
        print0(f"step:{step+1}/{train_steps} train_loss:{normalized_train_loss:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
        
        # 记录到wandb
        if cli_args.wandb == "yes" and master_process:
            wandb.log({"train_loss": float(normalized_train_loss), "step": step})
        
        # 记录到TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Training/Loss', float(normalized_train_loss), step)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

# 关闭TensorBoard writer
if master_process and tensorboard_writer is not None:
    tensorboard_writer.close()
    print0("TensorBoard writer closed", console=True)

dist.destroy_process_group()

