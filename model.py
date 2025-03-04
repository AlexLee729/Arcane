from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.
    Uses activation checkpointing to recompute the forward pass
    during backpropagation to save memory.
    """
    def __init__(self, config):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        # Linear projections for QKV and the final projection.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Buffers for caching keys and values (used during generation).
        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)

        # Precompute the inverse frequency for RoPE.
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        # Precompute cosine and sine tables for RoPE up to max_seq_len.
        max_seq_len = getattr(config, "block_size", 2048)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [max_seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))  # [1, 1, max_seq_len, head_dim]
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def clear_cache(self) -> None:
        """Clears the cached key and value tensors."""
        self.cache_k = None
        self.cache_v = None

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional embeddings.
        x: [B, n_head, T, head_dim]
        cos, sin: [1, 1, T, head_dim]
        """
        x = x.view(*x.shape[:-1], self.head_dim // 2, 2)
        cos = cos.view(cos.shape[0], cos.shape[1], cos.shape[2], self.head_dim // 2, 2)
        sin = sin.view(sin.shape[0], sin.shape[1], sin.shape[2], self.head_dim // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rotated = torch.stack((x1 * cos[..., 0] - x2 * sin[..., 0],
                                  x2 * cos[..., 1] + x1 * sin[..., 1]), dim=-1)
        return x_rotated.reshape(*x.shape[:-2], self.head_dim)

    def _forward_attn(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        """
        Core attention computation.
        This helper method is wrapped via checkpointing when use_cache is False.
        """
        B, T, _ = x.size()
        # Compute QKV.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Slice the precomputed cosine and sine tables.
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]

        # Apply rotary embeddings.
        q = self.apply_rotary_pos_emb(q, cos, sin)
        k = self.apply_rotary_pos_emb(k, cos, sin)

        # Handle caching (if used for autoregressive generation).
        if use_cache and self.cache_k is not None:
            k = torch.cat((self.cache_k, k), dim=2)
            v = torch.cat((self.cache_v, v), dim=2)
        if use_cache:
            self.cache_k = k
            self.cache_v = v

        # Scaled dot-product attention with causal masking.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        return y

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass for the attention module.
        When not using caching, wraps the core computation in a checkpoint to save memory.
        """
        if use_cache:
            # If caching is needed, bypass checkpointing.
            return self._forward_attn(x, use_cache=True)
        else:
            # When training without caching, use checkpointing to recompute intermediate
            # activations during backpropagation (saving memory).
            return torch.utils.checkpoint.checkpoint(
                partial(self._forward_attn, use_cache=False),
                x,
                use_reentrant=False
            )

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x: torch.Tensor):
        return self.c_proj(F.gelu(self.c_fc(x)) * self.c_fc2(x))

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
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=1e-8)
        self.ffn_norm = nn.RMSNorm(config.n_embd, eps=1e-8)

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        x = x + self.attn(self.attn_norm(x), use_cache=use_cache)
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
    RMSNorm, and rotary embeddings within the attention mechanism.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=1e-8)

        # Output head with weight tying.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear and embedding layers with improved schemes."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (B, T).
            targets: Optional target indices for loss computation.
            use_cache: Whether to enable caching (for generation).
            
        Returns:
            A tuple of (logits, loss), where loss is None if targets are not provided.
        """
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        # Token embedding.
        x = self.wte(idx)  # (B, T, n_embd)

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, use_cache=use_cache)

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
        print(f"Total number of parameters: {total_params}\n")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 32,
        num_return_sequences: int = 1,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.1,  # new parameter
        device: str = "cpu",
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from a given prompt using combined top-k, nucleus (top-p) sampling,
        and applies a repetition penalty to discourage repeated tokens.
        
        Args:
            prompt: The text prompt.
            max_length: Maximum sequence length (including prompt).
            num_return_sequences: Number of sequences to generate.
            top_k: The number of top tokens to consider for sampling. If <= 0, disabled.
            top_p: The cumulative probability threshold for nucleus sampling. If >= 1.0, disabled.
            temperature: Temperature for scaling logits.
            repetition_penalty: Factor to penalize already generated tokens (>1.0 penalizes).
            device: Device to run the generation on.
            eos_token_id: Optional end-of-sequence token id to stop generation early.
            
        Returns:
            Generated token indices of shape (num_return_sequences, sequence_length).
        """
        self.eval()
        tokens = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0).repeat(num_return_sequences, 1)
        generated = tokens

        # Clear caches for all blocks.
        for block in self.blocks:
            block.attn.clear_cache()

        for _ in range(max_length - tokens.size(1)):
            logits, _ = self(generated, use_cache=True)
            next_logits = logits[:, -1, :] / temperature

            # Clone logits for filtering.
            filtered_logits = next_logits.clone()

            # Apply top-k filtering if enabled.
            if top_k > 0:
                kth_values = torch.topk(filtered_logits, top_k, dim=-1)[0][..., -1, None]
                filtered_logits = torch.where(
                    filtered_logits < kth_values,
                    torch.tensor(-float("Inf"), device=filtered_logits.device),
                    filtered_logits,
                )

            # Apply nucleus (top-p) filtering if enabled.
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Shift the mask to always keep at least one token.
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, -float("Inf"))
                filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

            # Apply repetition penalty: reduce logits for tokens already generated.
            if repetition_penalty > 1.0:
                for i in range(generated.size(0)):
                    # Get unique tokens generated for each sequence
                    unique_tokens = set(generated[i].tolist())
                    for token_id in unique_tokens:
                        filtered_logits[i, token_id] /= repetition_penalty

            # Compute probabilities. With valid filtering, softmax should be safe.
            probs = F.softmax(filtered_logits, dim=-1)
            if torch.isnan(probs).any() or (probs < 0).any():
                raise ValueError("Invalid probabilities generated after filtering.")

            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token (if all sequences have produced it, stop early).
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            generated = torch.cat((generated, next_token), dim=1)
        return generated