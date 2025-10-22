import torch

class KVCache:
    def __init__(self, n_layer, max_seq_len, n_head, head_dim, device, dtype=torch.bfloat16):
        self.n_layer = n_layer 
        self.max_seq_len = max_seq_len 
        self.n_head = n_head 
        self.head_dim = head_dim
        self.device = device 
        self.dtype = dtype 
        self.cache = [None] * n_layer 
        self.current_pos = 0 

    # Inserts new keys and values into the cache for a given layer
    def insert_kv(self, layer_idx, k, v):
        if self.cache[layer_idx] is None:
            B, _, T, _ = k.shape
            # Initialize cache with zeros for batch, heads, max_seq_len, head_dim
            k_cache = torch.zeros(B, self.n_head, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            v_cache = torch.zeros(B, self.n_head, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            self.cache[layer_idx] = (k_cache, v_cache)
        k_cache, v_cache = self.cache[layer_idx]
        # Update cache with new keys and values at current position
        k_cache[:, :, self.current_pos:self.current_pos+k.size(2), :] = k
        v_cache[:, :, self.current_pos:self.current_pos+k.size(2), :] = v
        # Return cached keys and values up to current position
        return k_cache[:, :, :self.current_pos+k.size(2), :], v_cache[:, :, :self.current_pos+k.size(2), :]

    # Returns current sequence position
    def get_pos(self):
        return self.current_pos

    # Updates position after adding new tokens
    def update_pos(self, num_new_tokens):
        self.current_pos += num_new_tokens

    # Resets cache for new generation
    def reset(self):
        self.cache = [None] * self.n_layer
        self.current_pos = 0