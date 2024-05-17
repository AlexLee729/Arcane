import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
import config
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Linear layers for key, query, and value projections for all heads
        self.key = nn.Linear(config.n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(config.n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(config.n_embd, num_heads * head_size, bias=False)
        # Output projection layer
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        # Dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        # Check if flash attention is available (supported in PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.num_heads = num_heads
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimension
        
        # Linear transformation and split into multiple heads
        k = self.key(x).view(B, T, self.num_heads, self.head_size)
        q = self.query(x).view(B, T, self.num_heads, self.head_size)
        v = self.value(x).view(B, T, self.num_heads, self.head_size)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=config.dropout, is_causal=True)
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
        y = self.resid_dropout(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear = nn.Linear(n_embd, 4 * n_embd)
        self.activation = nn.GELU()
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.mlp = MLP(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table (gets probability for next token)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        num_params = sum(p.numel() for p in self.parameters()) / 1e6
        return num_params

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T,C)
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
    def generate(self, prompt, max_new_tokens, temperature=1, batch_size=10):
        self.eval()
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(prompt)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.token_embedding_table.weight.device)

        generated_tokens = []

        while len(generated_tokens) < max_new_tokens:
            batch_tokens = []
            for _ in range(batch_size):
                idx_cond = idx[:, -config.block_size:]  # Crop context to last block_size tokens
                logits, _ = self(idx_cond)  # Get logits for next token
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)  # Calculate probabilities
                idx_next = torch.multinomial(probs, num_samples=1)
                batch_tokens.append(idx_next.item())

            idx_next_batch = torch.tensor(batch_tokens, dtype=torch.long, device=idx.device).unsqueeze(0)
            idx = torch.cat((idx, idx_next_batch), dim=1)
            generated_tokens.extend(batch_tokens)

        result = enc.decode(generated_tokens)
        return result