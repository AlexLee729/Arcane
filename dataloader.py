# import numpy as np
# import os
# import torch

# def load_tokens(filename):
#     npt = np.load(filename)
#     npt = npt.astype(np.int32)
#     ptt = torch.tensor(npt, dtype=torch.long)
#     return ptt

# class DataLoader:
#     def __init__(self, B, T, split):
#         self.B = B
#         self.T = T
#         assert split in {'train', 'val'}
        
#         # get the shard filenames
#         data_root = "edu_fineweb10B"
#         shards = os.listdir(data_root)
#         shards = [s for s in shards if split in s]
#         shards = sorted(shards)
#         shards = [os.path.join(data_root, s) for s in shards]
#         self.shards = shards
#         assert len(shards) > 0, f"no shards found for split {split}"
#         print(f"found {len(shards)} shards for split {split}")
#         self.reset()
        
#     def reset(self):
#         # state, init at shard zero
#         self.current_shard = 0
#         self.tokens = load_tokens(self.shards[self.current_shard])
#         self.current_position = 0
    
#     def set_state(self, shard):
#         self.current_shard = shard
#         self.tokens = load_tokens(self.shards[self.current_shard])
#         self.current_position = 0

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position : self.current_position+B*T+1]
#         x = (buf[:-1]).view(B, T) # inputs
#         y = (buf[1:]).view(B, T) # targets
#         # advance the position in the tensor
#         self.current_position += B * T
#         # if loading the next batch would be out of bounds, advance to next shard
#         if self.current_position + (B * T + 1) > len(self.tokens):
#             self.current_shard = (self.current_shard + 1) % len(self.shards)
#             self.tokens = load_tokens(self.shards[self.current_shard])
#             self.current_position = 0
#         return x, y

import json
import os
import random
import torch
import tiktoken
from torch.nn.utils.rnn import pad_sequence

enc = tiktoken.get_encoding('gpt2')

# Define EOS token ID (GPT models often reserve a special ID for this)
EOS_TOKEN_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]  # Assuming GPT-2-style tokenization

def load_jsonl_data(filename):
    """Load data from a JSONL file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return [(enc.encode(entry["prompt"]) + [EOS_TOKEN_ID], enc.encode(entry["response"]) + [EOS_TOKEN_ID]) for entry in data]

def shuffle_jsonl_file(input_filename):
    """Shuffle the data within a JSONL file."""
    # Load the JSON data from the input file
    with open(input_filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    
    # Shuffle the data
    random.shuffle(data)

    # Overwrite the original file with shuffled data
    with open(input_filename, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

class JSONDataLoader:
    def __init__(self, data_dir, B, split, max_seq_length=1024):
        self.B = B
        self.max_seq_length = max_seq_length
        assert split in {'train', 'val'}
        
        # Gather all JSONL files in the directory
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl')])
        if not self.files:
            raise ValueError(f"No JSONL files found in directory: {data_dir}")
        
        self.split = split
        print(f"Found {len(self.files)} files for split {split}")
        self.reset()
        
    def reset(self):
        self.current_file_idx = 0
        self._load_current_file_data()
        self.current_position = 0

    def _load_current_file_data(self):
        current_file = self.files[self.current_file_idx]
        shuffle_jsonl_file(current_file)  # Shuffle the file before loading
        data = load_jsonl_data(current_file)
        split_idx = int(0.8 * len(data))
        
        if self.split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

    def set_batch_size(self, new_B):
        """Dynamically sets a new batch size."""
        self.B = new_B

    def next_batch(self):
        B = self.B
        remaining_samples = len(self.data) - self.current_position

        # Adjust batch size if fewer samples are remaining than required for a full batch
        effective_B = min(B, remaining_samples) if remaining_samples > 0 else 1

        batch = []
        while len(batch) < effective_B and self.current_position < len(self.data):
            prompt_tokens, response_tokens = self.data[self.current_position]
            combined_tokens = prompt_tokens + response_tokens
            if len(combined_tokens) <= self.max_seq_length:
                batch.append((prompt_tokens, response_tokens))
            self.current_position += 1

        x, y, attention_mask = [], [], []

        for prompt_tokens, response_tokens in batch:
            # Combine prompt and response
            combined_tokens = prompt_tokens + response_tokens
            pad_token_id = 0
            
            pad_length = self.max_seq_length - len(combined_tokens)
            padded_combined_tokens = combined_tokens + [pad_token_id] * pad_length
            padded_attention_mask = [1] * len(combined_tokens) + [0] * pad_length
            
            prompt_padded = padded_combined_tokens[:len(prompt_tokens) + 1]
            response_padded = padded_combined_tokens[len(prompt_tokens) + 1:]
            
            prompt_padded = prompt_padded + [pad_token_id] * (self.max_seq_length - len(prompt_padded))
            response_padded = response_padded + [pad_token_id] * (self.max_seq_length - len(response_padded))

            padded_attention_mask = [1] * len(prompt_tokens) + [1] * len(response_tokens) + [0] * (self.max_seq_length - len(combined_tokens))
            
            x.append(torch.tensor(prompt_padded, dtype=torch.long))
            y.append(torch.tensor(response_padded, dtype=torch.long))
            attention_mask.append(torch.tensor(padded_attention_mask, dtype=torch.long))

            if self.current_position >= len(self.data):
                self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
                self._load_current_file_data()
                self.current_position = 0
        
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        return x, y, attention_mask