# LLM Model Code Explanation

## Introduction
This document provides a concise explanation of the LLM model code, breaking down each section and explaining its functionality.

## 1. CasualSelfAttention Class
### Initialization
```python
def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
	# Multi-head attention projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, dtype=torch.bfloat16)
        
	# Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, dtype=torch.bfloat16)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # Store configuration parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Key and value cache for inference
        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)
        
        # Initialize Rotary Positional Embedding (RoPE) parameters
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
```
- **Multi-head Attention**: `self.c_attn` projects input embeddings into key, query, and value tensors for multi-head attention
- **Output Projection**: `self.c_proj` transforms the concatenated multi-head attention outputs back to the original embedding size
### Rotate half
```python
def rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)
```
- **Purpose**: Performs rotation on input tensor `x` for positional encoding
### Rotary Positional Embedding
```python
def apply_rotary_pos_emb(self, q, k, cos, sin):
        q_cos = q * cos - self.rotate_half(q) * sin
        q_sin = q * sin + self.rotate_half(q) * cos
        k_cos = k * cos - self.rotate_half(k) * sin
        k_sin = k * sin + self.rotate_half(k) * cos
        return q_cos + q_sin, k_cos + k_sin
```
- **Purpose**: Applies rotary positional embeddings (`cos` and `sin`) to queries (`q`) and keys (`k`) for enhancing model performance.
### Forward
```python
def forward(self, x, use_cache=False):
        x = x.to(dtype=torch.bfloat16)
        B, T, C = x.size() 
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        # Apply RoPE
        seq_len = k.shape[-2]
        t = torch.arange(seq_len, device=k.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16)
        
        if use_cache and self.cache_k is not None:
            k = torch.cat((self.cache_k, k), dim=2)
            v = torch.cat((self.cache_v, v), dim=2)
        if use_cache:
            self.cache_k = k
            self.cache_v = v
            
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
```
- **Input Transformation**: Converts input `x` to `torch.bfloat16` and retrieves batch size (`B`), sequence length (`T`), and embedding size (`C`).
- **Multi-head Attention**: Projects input through `self.c_attn` to obtain `q`, `k`, and `v` tensors, reshaping them for multi-head attention calculations.
- **RoPE Application**: Applies rotary position embeddings (`cos` and `sin`) to enhance attention mechanism.
- **Caching**: Optionally caches key (`cache_k`) and value (`cache_v`) tensors for efficient inference
- **Attention Computation**: Uses scaled dot product attention (`F.scaled_dot_product_attention`) with casual masking (`is_casual=True`) to compute `y`. *Requires PyTorch >= 2.0*
- **Output Projection**: Projects the concatenated attention outputs back to the original embedding size usinf `self.c_proj`.
