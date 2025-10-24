import torch

class KVCache:
    def __init__(self, n_layer, max_seq_len, n_kv_head, head_dim, device, dtype=torch.bfloat16):
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.cache = [None] * n_layer
        self.current_pos = 0

    def insert_kv(self, layer_idx, k, v):
        B, _, T, _ = k.shape
        if self.cache[layer_idx] is None:
            k_cache = torch.zeros(B, self.n_kv_head, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            v_cache = torch.zeros(B, self.n_kv_head, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            self.cache[layer_idx] = (k_cache, v_cache)
        k_cache, v_cache = self.cache[layer_idx]
        k_cache[:, :, self.current_pos:self.current_pos + T, :] = k
        v_cache[:, :, self.current_pos:self.current_pos + T, :] = v
        self.current_pos += T
        return k_cache[:, :, :self.current_pos, :], v_cache[:, :, :self.current_pos, :]

    def get_pos(self):
        return self.current_pos

    def update_pos(self, num_new_tokens):
        self.current_pos += num_new_tokens

    def reset(self):
        self.cache = [None] * self.n_layer
        self.current_pos = 0