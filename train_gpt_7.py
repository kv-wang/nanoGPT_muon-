import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import wandb
import argparse
# 添加TensorBoard支持
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--wandb", type = str, default="no", choices=["yes", "no"], help="Whether to use wandb for logging")
parser.add_argument("--muon_momentum_0", type=float, default=0.95, help="Muon momentum[0] for nesterov")
parser.add_argument("--muon_momentum_1", type=float, default=0.95, help="Muon momentum[1] for buf")
parser.add_argument("--muon_type", type=str, default="default", help="Muon optimizer type")
parser.add_argument("--adaptive_beta", type=str, default="no", choices=["cos","NS_cos", "no"], help="Whether to use adaptive beta based on cosine similarity")
parser.add_argument("--tensorboard_path", type=str, default="logs_tensorboard_train7_6e-4_0.1_zkti", help="Tensorboard path")
parser.add_argument("--wandb_project", type=str, default="train_gpt_7_6e-4_0.1_zkti", help="Wandb project")
cli_args = parser.parse_args()

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
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


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

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
            _ortho = zeropower_via_newtonschulz5(g, steps=5).flatten()
            buf_ortho = zeropower_via_newtonschulz5(buf, steps=5).flatten()
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
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 32 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 10200 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 500 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()



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

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

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

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)) #模型包装
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)

# 根据muon_type选择不同的优化器配置
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, weight_decay=0.1, momentum=cfg[cli_args.muon_type],
                  rank=ddp_rank, world_size=ddp_world_size, type=cli_args.muon_type, adaptive_beta=cli_args.adaptive_beta)

optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# 记录训练配置信息
if master_process:
    print0("="*100, console=True)
  
    print0(f"Muon Type: {cli_args.muon_type}", console=True)
    print0(f"Muon Momentum 0: {cli_args.muon_momentum_0}", console=True)
    print0(f"Muon Momentum 1: {cli_args.muon_momentum_1}", console=True)
    print0(f"Adaptive Beta: {cli_args.adaptive_beta}", console=True)
    print0("="*100, console=True)

training_time_ms = 0

# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in tqdm(range(args.num_iterations + 1), desc=f"Training {cli_args.muon_type}", unit="step", disable=not master_process):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
            
            # 记录到wandb
            if cli_args.wandb == "yes":
                wandb.log({"val_loss": float(val_loss), "step": step})
            
            # 记录到TensorBoard
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('Validation/Loss', float(val_loss), step)
                tensorboard_writer.add_scalar('Training/Time_ms', training_time_ms, step)
                tensorboard_writer.add_scalar('Training/Step_avg_ms', training_time_ms/(timed_steps-1), step)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        #log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        # torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again

        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
   
    
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            tensorboard_writer.add_scalar('Training/Loss', float(train_loss), step)
            print(f"step:{step}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
            print0(f"step:{step}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)

# 关闭TensorBoard writer
if master_process and tensorboard_writer is not None:
    tensorboard_writer.close()
    print0("TensorBoard writer closed", console=True)

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()