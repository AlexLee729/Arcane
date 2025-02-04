from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # Ensure that embedding dimension is divisible by the number of heads
        
        # Linear layers for Q, K, V projections, and output projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Attention parameters
        self.n_head = config.n_head  # Number of attention heads
        self.n_embd = config.n_embd  # Embedding dimension
        self.head_dim = config.n_embd // config.n_head  # Dimension per head

        # Cache for key and value tensors for inference
        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)

        # Rotary positional embedding (RoPE) frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.bfloat16) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
    
    def rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split even and odd dimensions
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)  # Rotate halves and concatenate
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        q_rot = q * cos + self.rotate_half(q) * sin  # Apply RoPE to query
        k_rot = k * cos + self.rotate_half(k) * sin  # Apply RoPE to key
        return q_rot, k_rot

    def compute_rope(self, seq_len, device):
        t = torch.arange(seq_len, dtype=torch.bfloat16, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        return cos, sin
    
    def forward(self, x, attention_mask=None, use_cache=False):
        B, T, C = x.size()  # Batch size, sequence length, embedding size

        # Get QKV projections from the input
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute rotary positional embeddings (RoPE)
        cos, sin = self.compute_rope(seq_len=T, device=x.device)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if use_cache and self.cache_k is not None:
            k = torch.cat((self.cache_k, k), dim=2)
            v = torch.cat((self.cache_v, v), dim=2)
        if use_cache:
            self.cache_k = k
            self.cache_v = v

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, 1024]
            attention_mask = attention_mask.expand(B, self.n_head, T, T)  # Expand to [B, n_head, 1024, 1024]
            # Convert the mask to boolean
            attention_mask = attention_mask.to(torch.bool)
        # Perform scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.SiLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer.
        mlp (nn.Module): Feed-forward network (MLP).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = MLP(config)
        self.attn_norm = nn.RMSNorm(config.n_embd)
        self.ffn_norm = nn.RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = False):
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask, use_cache=use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass
class GPTConfig:
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50257
    block_size: int = 1024

class GPT(nn.Module):
    """
    GPT-style Transformer language model.
    Uses token embeddings (with weight tying for output logits),
    LayerNorm, and rotary embeddings within the attention mechanism.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head with weight tying.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (B, T).
            targets: Optional target indices for loss computation.
            attention_mask: Optional attention mask.
            use_cache: Whether to enable caching (for generation).
            
        Returns:
            A tuple of (logits, loss), where loss is None if targets are not provided.
        """
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        # Token embedding.
        x = self.wte(idx)  # (B, T, n_embd)

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, use_cache=use_cache)

        # Final layer normalization.
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float):
        """
        Set up the optimizer with separate parameter groups for weight decay.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 32,
        num_return_sequences: int = 1,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        device: str = "cpu",
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from a given prompt using top-k and nucleus (top-p) sampling.
        
        Args:
            prompt: The text prompt.
            max_length: Maximum sequence length (including prompt).
            num_return_sequences: Number of sequences to generate.
            top_k: The number of top tokens to consider for sampling.
            top_p: The cumulative probability threshold for nucleus sampling.
            temperature: Temperature for scaling logits.
            device: Device to run the generation on.
            eos_token_id: Optional end-of-sequence token id to stop generation early.
            
        Returns:
            Generated token indices of shape (num_return_sequences, sequence_length).
        """
        self.eval()
        enc = tiktoken.get_encoding("o200k_base")
        tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        tokens = tokens.repeat(num_return_sequences, 1)
        generated = tokens

        # Clear caches for all blocks.
        for block in self.blocks:
            block.attn.clear_cache()

        for _ in range(max_length - tokens.size(1)):
            logits, _ = self(generated, use_cache=True)
            next_logits = logits[:, -1, :] / temperature

            # Top-k sampling.
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(next_logits, top_k, dim=-1)
                topk_probs = F.softmax(topk_probs, dim=-1)
                next_token = topk_indices.gather(-1, torch.multinomial(topk_probs, num_samples=1))
            # Nucleus (top-p) sampling.
            elif top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumulative_probs > top_p] = -float("Inf")
                probs = F.softmax(sorted_logits, dim=-1)
                next_token = sorted_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            generated = torch.cat((generated, next_token), dim=1)
        return generated