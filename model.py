from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
    
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
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, use_cache=False):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask, use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, attention_mask=None, use_cache=False):
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Position indices
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask, use_cache=use_cache)
            
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters in the model: {total_params}")
        
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

    @torch.no_grad()
    def generate(self, prompt, max_length=32, num_return_sequences=1, top_k=50, top_p=0.95, temperature=1.0, device='cpu', eos_token_id=None):
        self.eval()  # Set model to evaluation mode
        enc = tiktoken.get_encoding('o200k_base')

        # Encode the prompt into token ids
        tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        tokens = tokens.repeat(num_return_sequences, 1)  # Repeat for num_return_sequences

        # Initialize the generated sequence tensor
        xgen = tokens

        with torch.no_grad():  # Disable gradients for inference to save memory
            # Clear cache in transformer blocks
            for block in self.transformer.h:
                block.attn.clear_cache()

            # Generate tokens
            for _ in range(max_length - tokens.size(1)):  # Loop until max_length is reached
                logits, _ = self(xgen, use_cache=True)  # (B, T, vocab_size)
                logits = logits[:, -1, :] / temperature  # Scale logits by temperature

                # Apply top-k sampling
                if top_k > 0:
                    topk_probs, topk_indices = torch.topk(logits, top_k, dim=-1)
                    topk_probs = F.softmax(topk_probs, dim=-1)
                    sampled_index = torch.multinomial(topk_probs, 1)  # Sample from top-k
                    sampled_token = topk_indices.gather(-1, sampled_index)  # Get corresponding token
                # Apply top-p (nucleus) sampling
                elif top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_logits[sorted_indices_to_remove] = -float('Inf')  # Remove tokens that exceed top-p

                    probs = F.softmax(sorted_logits, dim=-1)
                    sampled_token = torch.multinomial(probs, 1)  # Sample token
                    sampled_token = sorted_indices.gather(-1, sampled_token)
                # Default to full sampling (no top-k, no top-p)
                else:
                    probs = F.softmax(logits, dim=-1)  # Full distribution
                    sampled_token = torch.multinomial(probs, 1)  # Sample token

                # Handle EOS token (stop early if reached)
                if eos_token_id is not None and sampled_token == eos_token_id:
                    break

                # Append the sampled token to the generated sequence
                xgen = torch.cat((xgen, sampled_token), dim=1)

        return xgen