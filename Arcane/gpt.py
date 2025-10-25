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
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # Removed dropout-related attributes
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # RoPE precomputed inverse frequencies (unchanged, for compatibility)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def _apply_rope(self, x, cos, sin):
        # Modified to use precomputed cos/sin instead of recomputing
        # x: (B, nh, T, head_dim)
        B, nh, T, hd = x.shape
        # Slice cos/sin to match sequence length T
        cos = cos[:, :T, :, :]  # (1, T, 1, head_dim)
        sin = sin[:, :T, :, :]  # (1, T, 1, head_dim)
        # Rotate pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rot * sin

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        # qkv projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape to (B, nh, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Apply RoPE using precomputed cos/sin
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        # Scaled dot-product attention (causal), no dropout
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Merge heads
        y = attn_out.transpose(1, 2).reshape(B, T, C)
        # Output projection, no dropout
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # Removed dropout
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        # Pass cos/sin to attention
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.rotary_seq_len = config.block_size * 10  # Support 10x block_size, like Model 2
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd, eps=1e-8),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # Precompute RoPE embeddings
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.config.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.config.n_layer))

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)  # Use bfloat16 like Model 2
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # Slice cos/sin for the current sequence length
        cos = self.cos[:, :t, :, :]
        sin = self.sin[:, :t, :, :]
        # Token embeddings
        x = self.transformer.wte(idx)  # (b, t, n_embd)
        # Removed pos_emb and dropout
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_ids=None,
        enc=None,
    ):
        eos_ids = set(eos_token_ids or [])
        if enc and "<|endoftext|>" in enc._special_tokens:
            eos_ids.add(enc._special_tokens["<|endoftext|>"])
        stop_after = int(0.7 * max_new_tokens)
        for step in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    mask = cdf > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits[mask] = -float("inf")
                    logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            if idx_next.item() in eos_ids and step + 1 >= stop_after:
                break
        return idx