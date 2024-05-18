import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        head_size = config.n_embd // config.n_head
        # Linear layers for key, query, and value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_head * head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_head * head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_head * head_size, bias=config.bias)
        # Output projection layer
        self.c_proj = nn.Linear(config.n_head * head_size, config.n_embd)
        # Dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        # Check if flash attention is available (supported in PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.num_heads = config.n_head
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimension
        
        # Linear transformation and split into multiple heads
        k = self.key(x).view(B, T, self.num_heads, self.head_size)
        q = self.query(x).view(B, T, self.num_heads, self.head_size)
        v = self.value(x).view(B, T, self.num_heads, self.head_size)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout, is_causal=True)
        else:
            # Transpose to prepare for matrix multiplication
            k = k.transpose(1, 2)  # (B, num_heads, T, head_size)
            q = q.transpose(1, 2)  # (B, num_heads, T, head_size)
            v = v.transpose(1, 2)  # (B, num_heads, T, head_size)

            # Compute attention scores
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)  # (B, num_heads, T, T)

            # Mask out upper triangular elements
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)  # (1, T, T)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

            # Apply softmax to get attention weights and apply dropout (Converts attention scores to probabilities)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)

            # Weighted sum of values
            y = torch.matmul(attention_weights, v)  # (B, num_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, num_heads * head_size)

        # Project back to the original dimension and apply dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # each token directly reads off the logits for the next token from a lookup table (gets probability for next token)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("Number of parameters: %.2fM" % (self.get_num_params()))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
        # Find and exclude parameters from embedding layers
            embedding_params = sum(p.numel() for module in self.modules() if isinstance(module, nn.Embedding) for p in module.parameters())
            n_params -= embedding_params
        return n_params / 1e6

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        # Transformer blocks
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        # Language model head
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # Flatten the logits and targets for cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        enc = tiktoken.get_encoding("gpt2")
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            result = enc.decode(idx)

        return result