import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    """Configuration class for GPT model parameters."""
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention with Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Linear projections for query, key, value
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Precomputed inverse frequencies for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embeddings to the input tensor."""
        B, nh, T, hd = x.shape
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (T, hd/2)
        cos = torch.cat((freqs, freqs), dim=-1).cos()[None, None, :, :]  # (1, 1, T, hd)
        sin = torch.cat((freqs, freqs), dim=-1).sin()[None, None, :, :]  # (1, 1, T, hd)
        
        # Rotate pairs of coordinates
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rot * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention."""
        B, T, C = x.size()
        
        # Compute query, key, value projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        
        # Apply RoPE to query and key
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        
        # Scaled dot-product attention (causal)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Merge heads and project
        y = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    """Multi-Layer Perceptron module for GPT."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP."""
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """GPT model implementation."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None, "Vocabulary size must be specified"
        assert config.block_size is not None, "Block size must be specified"
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.RMSNorm(config.n_embd, eps=1e-8),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        """Initialize weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for GPT model."""
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Forward pass
        x = self.transformer.wte(idx)  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float) -> torch.optim.AdamW:
        """Configure AdamW optimizer with parameter-specific weight decay."""
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (param.dim() < 2 or
                any(s in name for s in ['wte', 'wpe', 'embedding', 'ln', 'norm'])):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
            fused=True,
        )

    @torch.inference_mode()
    def generate(self, tokens: List[int], max_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """Generate tokens using the GPT model."""
        assert isinstance(tokens, list), "Input tokens must be a list of integers"
        device = next(self.parameters()).device
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits, _ = self(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample or take argmax
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()