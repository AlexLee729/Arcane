import torch

# Model Hyperparameters
batch_size = 2  # Number of sequences run in parallel
block_size = 512  # Context size used for prediction
n_embd = 768
n_head = 12
n_layer = 8
dropout = 0.0
vocab_size = 40478  # GPT-2 vocabulary size

# Training Hyperparameters
max_iters = 10000
eval_interval = 500
learning_rate = 6e-4
eval_iters = 200

# File path for saving/loading the model
gpt_model_path = 'Models/vtuber_chat.pth'

# Determine device (use 'mps' if available, otherwise default to 'cpu')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'