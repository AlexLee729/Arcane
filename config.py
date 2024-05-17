import torch

# Model Hyperparameters
batch_size = 2  # Number of sequences run in parallel
block_size = 1024  # Context size used for prediction
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.2
vocab_size = 50304  # GPT-2 vocabulary size - 50304

# Training Hyperparameters
max_iters = 50000
eval_interval = 500
learning_rate = 6e-4
eval_iters = 200

# Finetuning Hyperparameters
# max_iters = 40
# eval_interval = 5
# learning_rate = 6e-5
# eval_iters = 200

# File path for saving/loading the model
gpt_model_path = 'Models/Arcane_small.pth'

# Determine device (use 'mps' if available, otherwise default to 'cpu')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'