# 123.74M Model

# Model Hyperparameters
batch_size = 4  # Number of sequences run in parallel
block_size = 1024  # Context size used for prediction
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0
vocab_size = 50304  # GPT-2 vocabulary size
bias = False

# Training Hyperparameters
max_iters = 3000
eval_interval = 300
learning_rate = 6e-4
eval_iters = 200

# Finetuning Hyperparameters
# max_iters = 40
# eval_interval = 5
# learning_rate = 3e-5
# eval_iters = 200

# Macbook GPU
device = 'mps'

# if computer doesnt have gpu
# device = 'cpu'