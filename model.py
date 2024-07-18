from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma
    
class MultiQueryCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Number of heads and dimensions per head
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Query projections
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, dtype=torch.bfloat16)
        # Shared key and value projections
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd // config.n_head, dtype=torch.bfloat16)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, dtype=torch.bfloat16)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # kv-cache for inference
        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
        
    def forward(self, x, use_cache=False):
        x = x.to(dtype=torch.bfloat16)
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values
        q = self.c_attn_q(x)  # Queries
        kv = self.c_attn_kv(x)  # Keys and values
        
        # Split keys and values
        k, v = kv.split(self.head_dim, dim=2)
        
        # Reshape queries
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        # Reshape keys and values (single shared key-value pair per head)
        k = k.view(B, 1, T, self.head_dim).repeat(1, self.n_head, 1, 1) # (B, nh, T, hs)
        v = v.view(B, 1, T, self.head_dim).repeat(1, self.n_head, 1, 1) # (B, nh, T, hs)
        
        if use_cache and self.cache_k is not None:
            k = torch.cat((self.cache_k, k), dim=2)
            v = torch.cat((self.cache_v, v), dim=2)
        if use_cache:
            self.cache_k = k
            self.cache_v = v
            
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y) # output projection
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, dtype=torch.bfloat16)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, dtype=torch.bfloat16)
        self.layer_scale = LayerScale(config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = x.to(dtype=torch.bfloat16)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.layer_scale(x)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiQueryCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # Position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # Final layer norm
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=torch.bfloat16)
        
        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None, use_cache=False):
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Position indices
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache)
            
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
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Model size: {num_decay_params + num_nodecay_params}")
        
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
    
    def generate(self, prompt, max_length=32, num_return_sequences=1, top_k=50, device='cpu'):
        self.eval()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        
        print(prompt, end="", flush=True)
        xgen = tokens
        
        with torch.no_grad():
            # Clear the cache before generating
            for block in self.transformer.h:
                block.attn.clear_cache()

            while xgen.size(1) < max_length:
                # forward the model to get the logits
                logits, _ = self(xgen, use_cache=True)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # take the logits at the last position
                probs = F.softmax(logits, dim=-1)# get the probabilities
                
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                ix = torch.multinomial(topk_probs, 1)  # select a token from the top-k probabilities
                xcol = torch.gather(topk_indices, -1, ix)  # gather the corresponding indices
                xgen = torch.cat((xgen, xcol), dim=1) # append to the sequence
                
                # Decode and print the last generated word
                last_token = xgen[0, -1].item()
                last_word = enc.decode([last_token])
                print(last_word, end="", flush=True)
                
                # Check if generated length exceeds 70% of max_length
                if xgen.size(1) > 0.7 * max_length and (last_word.endswith('.') or last_word.endswith('!') or last_word.endswith('?')):
                    break
        print()