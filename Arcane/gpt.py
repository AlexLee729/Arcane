import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be multiple of n_head"
        # parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # projections
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # RoPE precomputed inv frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def _apply_rope(self, x):
        # x: (B, nh, T, head_dim)
        B, nh, T, hd = x.shape
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (T, hd/2)
        cos = torch.cat((freqs, freqs), dim=-1).cos()[None, None, :, :]
        sin = torch.cat((freqs, freqs), dim=-1).sin()[None, None, :, :]
        # rotate pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rot * sin

    def forward(self, x):
        B, T, C = x.size()
        # qkv projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        # reshape to (B, nh, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # apply RoPE
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        # scaled dot-product attention (causal)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # merge heads
        y = attn_out.transpose(1, 2).reshape(B, T, C)
        # output projection + dropout
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3) 
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) 
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False) 
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False) 

    def forward(self, x):
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd, eps=1e-8),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)\
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # exclude embeddings, layernorm, and bias from weight decay
            if (
                param.dim() < 2 or
                'wte' in name or
                'wpe' in name or
                'embedding' in name or
                'ln' in name.lower() or
                'norm' in name.lower()
            ):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
            fused=True,
        )
        return optimizer

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None):
        assert isinstance(tokens, list), "Input tokens must be a list of ints"
        device = next(self.parameters()).device  # get model device

        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim

        for _ in range(max_tokens):
            logits, _ = self(ids)  # forward pass (B, T, vocab_size)
            logits = logits[:, -1, :]  # take last token logits (B, vocab_size)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sampling or greedy
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()